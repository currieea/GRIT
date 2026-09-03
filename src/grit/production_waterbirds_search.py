"""Waterbirds-CF adapter for the shared production local-search lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from grit.config import LinearProbeTrainingConfig
from grit.projection import FittedLinearProjection
from grit.schemas import SeedStage
from grit.search import (
    SearchCandidate,
    SearchPlan,
    WaterbirdsProductionSearchConfig,
    current_code_provenance,
    current_environment_provenance,
)
from grit.search_outputs import (
    WaterbirdsPairedSummaryArtifact,
    WaterbirdsProductionMethodSummary,
    WaterbirdsProductionSummary,
)
from grit.search_scheduler import (
    CompletedStageRun,
    LocalRunScheduler,
    SearchRunTask,
    StageExecutor,
    WaterbirdsCompletedStageRun,
    make_final_search_task,
    make_search_task,
)
from grit.selection import CheckpointIdentity
from grit.training import (
    PersistedLinearCheckpointStore,
    persist_selected_linear_checkpoint,
)
from grit.waterbirds import (
    WaterbirdsAdjustedWeightSpec,
    WaterbirdsDatasetManifest,
    mint_waterbirds_adjusted_weight_spec,
)
from grit.waterbirds_features import (
    WaterbirdsFeatureCache,
    WaterbirdsTuningFeatureCache,
    fit_waterbirds_oracle_projection,
    load_waterbirds_feature_cache,
    load_waterbirds_tuning_feature_cache,
)
from grit.waterbirds_pairs import (
    WaterbirdsOraclePairManifest,
    WaterbirdsOraclePairSet,
)
from grit.waterbirds_run_contracts import (
    WaterbirdsArtifactReference,
    WaterbirdsCandidateConfig,
    WaterbirdsCheckpointArtifactReference,
    WaterbirdsDatasetArtifactReference,
    WaterbirdsFeatureArtifactReference,
    WaterbirdsPairArtifactReference,
    WaterbirdsProjectionArtifactReference,
    WaterbirdsRunResult,
    make_waterbirds_metric_summary,
)
from grit.waterbirds_selection import (
    FrozenWaterbirdsCandidate,
    WaterbirdsTuningFinalists,
    freeze_waterbirds_candidate,
    freeze_waterbirds_final_checkpoint,
    make_waterbirds_tuning_finalists,
    select_confirmed_waterbirds_candidate,
    select_waterbirds_checkpoint,
)
from grit.waterbirds_training import (
    TrainedWaterbirdsRun,
    WaterbirdsMethod,
    restore_waterbirds_checkpoint,
    train_waterbirds_linear_probe,
)

if TYPE_CHECKING:
    from grit.production_search import (
        ProductionExecutionLimits,
        ProductionSearchStatus,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeCandidate:
    planned: SearchCandidate
    config: WaterbirdsCandidateConfig
    projection: FittedLinearProjection | None


def run_waterbirds_production_search(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
) -> WaterbirdsProductionSummary | ProductionSearchStatus:
    """Run or continue the approved Waterbirds ERM/oracle-GRIT search."""

    from grit.production_search import (
        limited_tuning_candidates,
        persist_canonical_artifact,
        waterbirds_status_from_plan,
        write_experiment_index,
    )

    config = _search_config(plan)
    output_root = Path(plan.resolved_config.output_root)
    dataset = _dataset_manifest(plan)
    pairs = WaterbirdsOraclePairSet(manifest=_pair_manifest(plan))
    cache = _load_cache(
        plan, tuning_only=limits is not None and limits.stop_after == "tuning"
    )
    weights = mint_waterbirds_adjusted_weight_spec(dataset)
    if weights.canonical_digest() != (
        plan.resolved_config.lineage.adjusted_weight_spec_digest
    ):
        raise ValueError("Waterbirds adjusted weights changed after planning")
    projections: dict[int, FittedLinearProjection] = {}
    scheduler = LocalRunScheduler(output_root, plan)
    remaining = None if limits is None else limits.max_new_runs

    def run_stage(
        tasks: tuple[SearchRunTask, ...],
        execute: StageExecutor,
    ) -> tuple[CompletedStageRun, ...]:
        nonlocal remaining
        if limits is None:
            return scheduler.run_tasks(tasks, execute)
        allowance = len(tasks) if remaining is None else remaining
        results, newly_executed = scheduler.run_tasks_bounded(
            tasks, execute, max_new_runs=allowance
        )
        if remaining is not None:
            remaining -= newly_executed
        return results

    def runtime_candidate(candidate: SearchCandidate) -> _RuntimeCandidate:
        projection: FittedLinearProjection | None = None
        if candidate.method_id == "grit":
            rank = candidate.requested_rank
            if rank is None:
                raise AssertionError("planned Waterbirds GRIT candidate lacks a rank")
            projection = projections.get(rank)
            if projection is None:
                projection = fit_waterbirds_oracle_projection(
                    cache,
                    pairs,
                    requested_rank=rank,
                    relative_singular_value_tolerance=(
                        config.relative_singular_value_tolerance
                    ),
                )
                persist_canonical_artifact(
                    output_root / "projections" / f"grit-rank-{rank}.json",
                    projection.diagnostics,
                )
                projections[rank] = projection
        resolved = materialize_waterbirds_candidate_config(
            plan, candidate, projection
        )
        return _RuntimeCandidate(candidate, resolved, projection)

    def execute_pre_final(task: SearchRunTask, _run_root: Path) -> CompletedStageRun:
        runtime = runtime_candidate(task.candidate)
        trained = _train_task(cache, weights, runtime, task)
        decision = select_waterbirds_checkpoint(trained.validation_metrics)
        return WaterbirdsCompletedStageRun(
            schema_version="grit.waterbirds-search-stage-run/v1",
            dataset="waterbirds_cf",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            checkpoint_decision=decision,
        )

    tuning_candidates = limited_tuning_candidates(plan, limits)
    tuning_seeds = (
        config.seeds.stages.tuning
        if limits is None or limits.tuning_seed is None
        else (limits.tuning_seed,)
    )
    tuning_tasks = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in tuning_candidates
        for seed in tuning_seeds
    )
    tuning_runs = cast(
        tuple[WaterbirdsCompletedStageRun, ...],
        run_stage(tuning_tasks, execute_pre_final),
    )
    if limits is not None and (
        limits.stop_after == "tuning" or len(tuning_runs) != len(tuning_tasks)
    ):
        return waterbirds_status_from_plan(plan)
    finalists = _finalists(plan, tuning_runs, output_root)
    candidates_by_id = {item.candidate_id: item for item in plan.candidates}
    confirmation_tasks = tuple(
        make_search_task(
            plan,
            candidates_by_id[item.candidate_id],
            SeedStage.CONFIRMATION,
            seed,
        )
        for method in ("erm", "grit")
        for item in finalists[method].ordered_candidates
        for seed in config.seeds.stages.confirmation
    )
    confirmation_runs = cast(
        tuple[WaterbirdsCompletedStageRun, ...],
        run_stage(confirmation_tasks, execute_pre_final),
    )
    if limits is not None and len(confirmation_runs) != len(confirmation_tasks):
        return waterbirds_status_from_plan(plan)
    winners = _freeze_winners(
        plan, finalists, confirmation_runs, output_root
    )
    if isinstance(cache, WaterbirdsTuningFeatureCache):
        raise AssertionError("tuning-only Waterbirds execution reached final stage")

    def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
        frozen = winners[task.candidate.method_id]
        runtime = runtime_candidate(task.candidate)
        trained = _train_task(cache, weights, runtime, task)
        decision = select_waterbirds_checkpoint(trained.validation_metrics)
        frozen_checkpoint = freeze_waterbirds_final_checkpoint(decision, frozen)
        selected = trained.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
        checkpoint_root = run_root / "selected-checkpoint"
        checkpoint_manifest = persist_selected_linear_checkpoint(
            selected, checkpoint_root
        )
        persisted = PersistedLinearCheckpointStore(checkpoint_root)
        restoration = restore_waterbirds_checkpoint(
            frozen_checkpoint, persisted, trained.algorithm
        )
        handle = cache.issue_final_handle(
            run_id=trained.run_id,
            candidate_id=task.candidate.candidate_id,
            method_id=task.candidate.method_id,
            scientific_config_digest=task.candidate.scientific_config_digest,
            seed=task.seed,
            projection_rank=task.candidate.requested_rank,
        )
        final_view = handle.open(frozen, frozen_checkpoint, restoration)
        final_table = cache.verify_final_view(final_view)
        predictions = trained.algorithm.predict(final_table.features)
        from grit.waterbirds_selection import compute_waterbirds_final_metric

        final_metric = compute_waterbirds_final_metric(
            final_view,
            predictions,
            adjusted_weights=weights,
            record_id=f"metric:{trained.run_id}:test",
        )
        result = WaterbirdsRunResult(
            schema_version="grit.waterbirds-run-result/v2",
            result_kind="ordinary_waterbirds",
            status="succeeded",
            run_id=trained.run_id,
            final_seed=task.seed,
            resolved_config=runtime.config,
            resolved_config_digest=runtime.config.canonical_digest(),
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            candidate_selection=frozen,
            checkpoint_selection=frozen_checkpoint,
            restoration=restoration,
            final_test_metric=final_metric,
            selected_checkpoint_manifest_digest=(
                checkpoint_manifest.canonical_digest()
            ),
            artifacts=_result_artifacts(
                plan,
                task,
                checkpoint_manifest.canonical_digest(),
                runtime,
                frozen_checkpoint.checkpoint,
            ),
        )
        persist_canonical_artifact(run_root / "final-result.json", result)
        completed = WaterbirdsCompletedStageRun(
            schema_version="grit.waterbirds-search-stage-run/v1",
            dataset="waterbirds_cf",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            checkpoint_decision=decision,
            final_result_relative_path="final-result.json",
            final_result_digest=result.canonical_digest(),
            final_result=result,
        )
        persist_canonical_artifact(run_root / "result.json", completed)
        return completed

    final_tasks = tuple(
        make_final_search_task(
            plan,
            candidates_by_id[winners[method].candidate_id],
            seed,
            winners[method],
        )
        for method in ("erm", "grit")
        for seed in config.seeds.stages.final
    )
    final_runs = cast(
        tuple[WaterbirdsCompletedStageRun, ...],
        run_stage(final_tasks, execute_final),
    )
    if limits is not None:
        return waterbirds_status_from_plan(plan)
    summary = _summary(plan, finalists, winners, final_runs)
    persist_canonical_artifact(
        output_root / "summaries" / "waterbirds-summary.json", summary
    )
    persist_canonical_artifact(
        output_root / "summaries" / "waterbirds-paired-differences.json",
        WaterbirdsPairedSummaryArtifact(
            schema_version="grit.waterbirds-paired-summary/v1",
            configured_final_seeds=plan.seeds.stages.final,
            paired_worst_group_by_seed=summary.paired_worst_group_by_seed,
            paired_worst_group_summary=summary.paired_worst_group_summary,
        ),
    )
    write_experiment_index(plan, output_root)
    return summary


def _finalists(
    plan: SearchPlan,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    output_root: Path,
) -> dict[WaterbirdsMethod, WaterbirdsTuningFinalists]:
    from grit.production_search import persist_canonical_artifact

    artifacts = compute_waterbirds_finalists(plan, runs)
    for method, finalists in artifacts.items():
        persist_canonical_artifact(
            output_root / "selection" / method / "tuning-finalists.json",
            finalists,
        )
    return artifacts


def compute_waterbirds_finalists(
    plan: SearchPlan,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> dict[WaterbirdsMethod, WaterbirdsTuningFinalists]:
    records = tuple(item for run in runs for item in run.validation_metrics)
    artifacts: dict[WaterbirdsMethod, WaterbirdsTuningFinalists] = {}
    for method in ("erm", "grit"):
        finalists = make_waterbirds_tuning_finalists(
            tuple(item for item in records if item.method_id == method),
            plan.seeds.stages,
        )
        artifacts[method] = finalists
    return artifacts


def _freeze_winners(
    plan: SearchPlan,
    finalists: dict[WaterbirdsMethod, WaterbirdsTuningFinalists],
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    output_root: Path,
) -> dict[WaterbirdsMethod, FrozenWaterbirdsCandidate]:
    from grit.production_search import persist_canonical_artifact

    winners = compute_waterbirds_winners(plan, finalists, runs)
    for method, frozen in winners.items():
        persist_canonical_artifact(
            output_root / "selection" / method / "winner.json", frozen
        )
    return winners


def compute_waterbirds_winners(
    plan: SearchPlan,
    finalists: dict[WaterbirdsMethod, WaterbirdsTuningFinalists],
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> dict[WaterbirdsMethod, FrozenWaterbirdsCandidate]:
    records = tuple(item for run in runs for item in run.validation_metrics)
    winners: dict[WaterbirdsMethod, FrozenWaterbirdsCandidate] = {}
    for method in ("erm", "grit"):
        artifact = finalists[method]
        ids = {item.candidate_id for item in artifact.ordered_candidates}
        decision = select_confirmed_waterbirds_candidate(
            tuple(
                item
                for item in records
                if item.method_id == method and item.candidate_id in ids
            ),
            artifact,
            plan.seeds.stages,
        )
        frozen = freeze_waterbirds_candidate(
            decision, artifact, plan.seeds.stages
        )
        winners[method] = frozen
    return winners


def materialize_waterbirds_candidate_config(
    plan: SearchPlan,
    candidate: SearchCandidate,
    projection: FittedLinearProjection | None,
) -> WaterbirdsCandidateConfig:
    config = _search_config(plan)
    lineage = plan.resolved_config.lineage
    weights = lineage.adjusted_weight_spec_digest
    if weights is None:
        raise ValueError("Waterbirds plan lacks adjusted-weight lineage")
    if candidate.method_id == "erm":
        pair_digest = None
        projection_digest = None
        rank = None
        tolerance = None
    else:
        if projection is None or candidate.requested_rank is None:
            raise ValueError("Waterbirds GRIT candidate lacks fitted projection")
        pair_digest = lineage.pair_manifest_digest
        projection_digest = projection.diagnostics.canonical_digest()
        rank = candidate.requested_rank
        tolerance = float(config.relative_singular_value_tolerance)
    resolved = WaterbirdsCandidateConfig(
        schema_version="grit.waterbirds-candidate/v2",
        protocol_id="waterbirds_cf/v1",
        non_reportable=False,
        method_id=candidate.method_id,
        dataset_profile="production",
        dataset_manifest_digest=lineage.dataset_manifest_digest,
        feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
        normalization=config.normalization,
        adjusted_weight_spec_digest=weights,
        pair_manifest_digest=pair_digest,
        projection_diagnostics_digest=projection_digest,
        projection_rank=rank,
        relative_singular_value_tolerance=tolerance,
        training=LinearProbeTrainingConfig(
            optimizer="adam",
            batch_size=config.batch_size,
            learning_rate=float(candidate.learning_rate),
            weight_decay=float(candidate.weight_decay),
            max_epochs=config.max_epochs,
        ),
        seed_sets=config.seeds.stages,
    )
    if resolved.scientific_config_digest() != candidate.scientific_config_digest:
        raise ValueError("planned Waterbirds candidate digest cannot be materialized")
    return resolved


def _train_task(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    weights: WaterbirdsAdjustedWeightSpec,
    runtime: _RuntimeCandidate,
    task: SearchRunTask,
) -> TrainedWaterbirdsRun:
    return train_waterbirds_linear_probe(
        cache.training_table(),
        cache.validation_table(),
        weights,
        runtime.config.training,
        run_id=f"run:{task.task_id}",
        candidate_id=task.candidate.candidate_id,
        method_id=task.candidate.method_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        seed_stage=task.stage,
        seed=task.seed,
        projection=runtime.projection,
        projection_rank=task.candidate.requested_rank,
    )


def _result_artifacts(
    plan: SearchPlan,
    task: SearchRunTask,
    checkpoint_digest: str,
    runtime: _RuntimeCandidate,
    checkpoint: CheckpointIdentity,
) -> tuple[WaterbirdsArtifactReference, ...]:
    root = Path(plan.resolved_config.output_root)
    inputs = {item.kind: item for item in plan.resolved_config.input_artifacts}
    config = runtime.config
    values: list[WaterbirdsArtifactReference] = [
        WaterbirdsDatasetArtifactReference(
            artifact_id="waterbirds-cf-dataset",
            kind="dataset_manifest",
            relative_uri=os.path.relpath(inputs["dataset_manifest"].path, root),
            digest=inputs["dataset_manifest"].digest,
        ),
        WaterbirdsFeatureArtifactReference(
            artifact_id="waterbirds-feature-cache",
            kind="feature_manifest",
            relative_uri=os.path.relpath(inputs["feature_manifest"].path, root),
            digest=inputs["feature_manifest"].digest,
            dataset_manifest_digest=config.dataset_manifest_digest,
            normalization=config.normalization,
        ),
        WaterbirdsCheckpointArtifactReference(
            artifact_id=f"checkpoint:{task.task_id}",
            kind="selected_linear_checkpoint",
            relative_uri=(
                f"{task.relative_directory}/selected-checkpoint/manifest.json"
            ),
            digest=checkpoint_digest,
            checkpoint=checkpoint,
        ),
    ]
    if task.candidate.method_id == "grit":
        projection = runtime.projection
        if projection is None or config.pair_manifest_digest is None:
            raise AssertionError("Waterbirds GRIT artifact lineage disappeared")
        values.extend(
            (
                WaterbirdsPairArtifactReference(
                    artifact_id="waterbirds-oracle-pairs",
                    kind="pair_manifest",
                    relative_uri=os.path.relpath(inputs["pair_manifest"].path, root),
                    digest=config.pair_manifest_digest,
                    dataset_manifest_digest=config.dataset_manifest_digest,
                ),
                WaterbirdsProjectionArtifactReference(
                    artifact_id=f"projection:grit:{task.candidate.requested_rank}",
                    kind="projection_diagnostics",
                    relative_uri=(
                        "projections/"
                        f"grit-rank-{task.candidate.requested_rank}.json"
                    ),
                    digest=projection.diagnostics.canonical_digest(),
                    pair_manifest_digest=config.pair_manifest_digest,
                    feature_cache_manifest_digest=(
                        config.feature_cache_manifest_digest
                    ),
                    normalization=config.normalization,
                    requested_rank=projection.diagnostics.requested_rank,
                ),
            )
        )
    return tuple(values)


def _summary(
    plan: SearchPlan,
    finalists: dict[WaterbirdsMethod, WaterbirdsTuningFinalists],
    winners: dict[WaterbirdsMethod, FrozenWaterbirdsCandidate],
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsProductionSummary:
    by_identity = {
        (run.task.candidate.method_id, run.task.seed): run for run in runs
    }
    methods: list[WaterbirdsProductionMethodSummary] = []
    for method in ("erm", "grit"):
        results: list[WaterbirdsRunResult] = []
        paths: list[tuple[int, str]] = []
        for seed in plan.seeds.stages.final:
            run = by_identity[(method, seed)]
            result = run.final_result
            if result is None:
                raise AssertionError("completed Waterbirds final run lacks result")
            results.append(result)
            paths.append(
                (seed, f"{run.task.relative_directory}/final-result.json")
            )
        worst = tuple(
            (item.final_seed, float(item.final_test_metric.worst_group_accuracy))
            for item in results
        )
        adjusted = tuple(
            (
                item.final_seed,
                float(item.final_test_metric.adjusted_average_accuracy),
            )
            for item in results
        )
        raw = tuple(
            (item.final_seed, float(item.final_test_metric.raw_average_accuracy))
            for item in results
        )
        top = finalists[method].ordered_candidates
        methods.append(
            WaterbirdsProductionMethodSummary(
                method_id=method,
                lineage=plan.resolved_config.lineage,
                selected_candidate_id=winners[method].candidate_id,
                finalist_candidate_ids=(
                    top[0].candidate_id,
                    top[1].candidate_id,
                    top[2].candidate_id,
                ),
                configured_final_seeds=plan.seeds.stages.final,
                result_paths_by_seed=tuple(paths),
                worst_group_by_seed=worst,
                adjusted_average_by_seed=adjusted,
                raw_average_by_seed=raw,
                worst_group_summary=make_waterbirds_metric_summary(
                    "worst_group_accuracy", tuple(value for _, value in worst)
                ),
                adjusted_average_summary=make_waterbirds_metric_summary(
                    "adjusted_average_accuracy",
                    tuple(value for _, value in adjusted),
                ),
                raw_average_summary=make_waterbirds_metric_summary(
                    "raw_average_accuracy", tuple(value for _, value in raw)
                ),
            )
        )
    erm_values = dict(methods[0].worst_group_by_seed)
    grit_values = dict(methods[1].worst_group_by_seed)
    paired = tuple(
        (seed, float(grit_values[seed]) - float(erm_values[seed]))
        for seed in plan.seeds.stages.final
    )
    return WaterbirdsProductionSummary(
        schema_version="grit.waterbirds-production-summary/v1",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        lineage=plan.resolved_config.lineage,
        methods=(methods[0], methods[1]),
        paired_worst_group_by_seed=paired,
        paired_worst_group_summary=make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(value for _, value in paired),
        ),
    )


def _dataset_manifest(plan: SearchPlan) -> WaterbirdsDatasetManifest:
    path = _input_path(plan, "dataset_manifest")
    manifest = WaterbirdsDatasetManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        manifest.canonical_digest()
        != plan.resolved_config.lineage.dataset_manifest_digest
    ):
        raise ValueError("Waterbirds dataset manifest changed after planning")
    return manifest


def _pair_manifest(plan: SearchPlan) -> WaterbirdsOraclePairManifest:
    path = _input_path(plan, "pair_manifest")
    manifest = WaterbirdsOraclePairManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        manifest.canonical_digest()
        != plan.resolved_config.lineage.pair_manifest_digest
    ):
        raise ValueError("Waterbirds pair manifest changed after planning")
    return manifest


def _load_cache(
    plan: SearchPlan,
    *,
    tuning_only: bool = False,
) -> WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache:
    loader = (
        load_waterbirds_tuning_feature_cache
        if tuning_only
        else load_waterbirds_feature_cache
    )
    cache = loader(
        _input_path(plan, "feature_manifest").parent,
        expected_dataset_manifest_digest=(
            plan.resolved_config.lineage.dataset_manifest_digest
        ),
        expected_normalization=plan.normalization,
    )
    if (
        cache.manifest.canonical_digest()
        != plan.resolved_config.lineage.feature_cache_manifest_digest
    ):
        raise ValueError("Waterbirds feature manifest changed after planning")
    return cache


def _input_path(
    plan: SearchPlan,
    kind: Literal["dataset_manifest", "feature_manifest", "pair_manifest"],
) -> Path:
    return next(
        Path(item.path)
        for item in plan.resolved_config.input_artifacts
        if item.kind == kind
    )


def _search_config(plan: SearchPlan) -> WaterbirdsProductionSearchConfig:
    config = plan.resolved_config.config
    if not isinstance(config, WaterbirdsProductionSearchConfig):
        raise TypeError("Waterbirds production runner received another dataset")
    return config
