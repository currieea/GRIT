"""Waterbirds-CF adapter for the shared production local-search lifecycle.

One search tree runs one selector: the ordinary validation worst-group selector, or the
separately labeled `test_oracle` track that reproduces the paper's test-selected
protocol. Every implemented method binds to the same trainer through
`_waterbirds_method`; the dataset supplies only the environment, group, pair, and
validation-loss bindings the method definitions require.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

from grit.config import (
    FishAlgorithmConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    LisaAlgorithmConfig,
    MatchDgAlgorithmConfig,
    RexAlgorithmConfig,
    SwadAlgorithmConfig,
)
from grit.data.waterbirds import (
    WaterbirdsAdjustedWeightSpec,
    WaterbirdsDatasetManifest,
    mint_waterbirds_adjusted_weight_spec,
)
from grit.data.waterbirds_pairs import (
    WaterbirdsOraclePairManifest,
    WaterbirdsOraclePairSet,
)
from grit.features.waterbirds import (
    WaterbirdsFeatureCache,
    WaterbirdsTuningFeatureCache,
    fit_waterbirds_oracle_projection,
    load_waterbirds_feature_cache,
    load_waterbirds_tuning_feature_cache,
    waterbirds_oracle_pair_features,
)
from grit.methods.baselines import (
    FishLinearProbeMethod,
    LisaLinearProbeMethod,
    MatchDgLinearProbeMethod,
    SwadLinearProbeMethod,
)
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    GroupDroLinearProbeMethod,
    IrmLinearProbeMethod,
    LinearProbeAlgorithm,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    persist_selected_linear_checkpoint,
)
from grit.methods.waterbirds_training import (
    DiagnosticEpochHook,
    TrainedWaterbirdsRun,
    WaterbirdsMethod,
    restore_waterbirds_checkpoint,
    train_waterbirds_linear_probe,
)
from grit.schemas import SeedStage
from grit.search.lifecycle import (
    ProductionLifecycleHooks,
    ProductionStatusHooks,
    production_lifecycle_status,
    run_production_lifecycle,
)
from grit.search.outputs import (
    WaterbirdsPairedSummaryArtifact,
    WaterbirdsProductionMethodSummary,
    WaterbirdsProductionSummary,
)
from grit.search.plan import (
    SearchCandidate,
    SearchPlan,
    WaterbirdsProductionSearchConfig,
    current_code_provenance,
    current_environment_provenance,
    waterbirds_candidate_algorithm,
)
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    persist_canonical_artifact,
)
from grit.search.scheduler import (
    CompletedStageRun,
    SearchRunTask,
    StageExecutor,
    WaterbirdsCompletedStageRun,
    make_final_search_task,
    make_search_task,
)
from grit.search.waterbirds_contracts import (
    WaterbirdsArtifactReference,
    WaterbirdsCandidateConfig,
    WaterbirdsCheckpointArtifactReference,
    WaterbirdsDatasetArtifactReference,
    WaterbirdsFeatureArtifactReference,
    WaterbirdsFinalResult,
    WaterbirdsPairArtifactReference,
    WaterbirdsProjectionArtifactReference,
    WaterbirdsRunResult,
    WaterbirdsTestOracleRunResult,
    make_waterbirds_metric_summary,
)
from grit.selection.cmnist import CheckpointIdentity
from grit.selection.waterbirds import (
    ORDINARY_WATERBIRDS_SELECTOR,
    FrozenWaterbirdsCandidate,
    WaterbirdsSelector,
    WaterbirdsSelectorRecord,
    WaterbirdsTuningFinalists,
    compute_waterbirds_diagnostic_metric,
    compute_waterbirds_final_metric,
    freeze_waterbirds_candidate,
    freeze_waterbirds_final_checkpoint,
    make_waterbirds_tuning_finalists,
    select_confirmed_waterbirds_candidate,
    select_waterbirds_checkpoint,
)

WATERBIRDS_GROUP_COUNT = 4
WATERBIRDS_ENVIRONMENT_COUNT = 2

WaterbirdsFinalists = dict[WaterbirdsMethod, WaterbirdsTuningFinalists]
WaterbirdsWinners = dict[WaterbirdsMethod, FrozenWaterbirdsCandidate]


@dataclass(frozen=True, slots=True)
class _RuntimeCandidate:
    planned: SearchCandidate
    config: WaterbirdsCandidateConfig
    projection: FittedLinearProjection | None
    pair_differences: torch.Tensor | None


def waterbirds_selector(config: WaterbirdsProductionSearchConfig) -> WaterbirdsSelector:
    return config.selectors[0]


def finalists_filename(selector: WaterbirdsSelector) -> str:
    return (
        "test-oracle-tuning-finalists.json"
        if selector == "test_oracle"
        else "tuning-finalists.json"
    )


def winner_filename(selector: WaterbirdsSelector) -> str:
    return "test-oracle-winner.json" if selector == "test_oracle" else "winner.json"


def run_waterbirds_production_search(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
) -> WaterbirdsProductionSummary | ProductionSearchStatus:
    """Run or continue one Waterbirds search tree under its configured selector."""

    config = _search_config(plan)
    selector = waterbirds_selector(config)
    output_root = Path(plan.resolved_config.output_root)
    dataset = _dataset_manifest(plan)
    pairs = WaterbirdsOraclePairSet(manifest=_pair_manifest(plan))
    # The test-oracle track scores every epoch on the test split, so it always
    # needs the full cache; the ordinary track can tune from the redacted one.
    tuning_only = limits is not None and limits.stop_after == "tuning"
    cache = _load_cache(plan, tuning_only=tuning_only and not config.test_oracle)
    weights = mint_waterbirds_adjusted_weight_spec(dataset)
    if weights.canonical_digest() != (
        plan.resolved_config.lineage.adjusted_weight_spec_digest
    ):
        raise ValueError("Waterbirds adjusted weights changed after planning")
    projections: dict[int, FittedLinearProjection] = {}
    pair_bank: dict[str, torch.Tensor] = {}

    def runtime_candidate(candidate: SearchCandidate) -> _RuntimeCandidate:
        projection: FittedLinearProjection | None = None
        differences: torch.Tensor | None = None
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
        elif candidate.method_id == "matchdg":
            differences = pair_bank.get("land_minus_water")
            if differences is None:
                land, water = waterbirds_oracle_pair_features(cache, pairs)
                differences = land - water
                pair_bank["land_minus_water"] = differences
        resolved = materialize_waterbirds_candidate_config(
            plan, candidate, projection, selector
        )
        return _RuntimeCandidate(candidate, resolved, projection, differences)

    def execute_pre_final(task: SearchRunTask, _run_root: Path) -> CompletedStageRun:
        runtime = runtime_candidate(task.candidate)
        trained = _train_task(cache, weights, runtime, task)
        decision = select_waterbirds_checkpoint(
            trained.selector_records(selector), selector
        )
        return WaterbirdsCompletedStageRun(
            schema_version="grit.waterbirds-search-stage-run/v1",
            dataset="waterbirds_cf",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            diagnostic_metrics=trained.diagnostic_metrics,
            checkpoint_decision=decision,
        )

    candidates_by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[WaterbirdsCompletedStageRun, ...]:
        if any(not isinstance(run, WaterbirdsCompletedStageRun) for run in runs):
            raise ValueError("Waterbirds lifecycle received another dataset result")
        return cast(tuple[WaterbirdsCompletedStageRun, ...], runs)

    def confirmation_tasks(
        finalists: WaterbirdsFinalists,
    ) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_search_task(
                plan,
                candidates_by_id[item.candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for item in finalists[method].ordered_candidates
            for seed in config.seeds.stages.confirmation
        )

    def make_winners(
        finalists: WaterbirdsFinalists,
        runs: tuple[WaterbirdsCompletedStageRun, ...],
        root: Path,
    ) -> WaterbirdsWinners:
        return _freeze_winners(plan, finalists, runs, root)

    def final_tasks(winners: WaterbirdsWinners) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                candidates_by_id[winners[method].candidate_id],
                seed,
                winners[method],
            )
            for method in plan.methods
            for seed in config.seeds.stages.final
        )

    def final_executor(winners: WaterbirdsWinners) -> StageExecutor:
        if isinstance(cache, WaterbirdsTuningFeatureCache):
            raise AssertionError("tuning-only Waterbirds execution reached final stage")
        full_cache = cache

        def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
            frozen = winners[task.candidate.method_id]
            runtime = runtime_candidate(task.candidate)
            trained = _train_task(full_cache, weights, runtime, task)
            decision = select_waterbirds_checkpoint(
                trained.selector_records(selector), selector
            )
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
            artifacts = _result_artifacts(
                plan,
                task,
                checkpoint_manifest.canonical_digest(),
                runtime,
                frozen_checkpoint.checkpoint,
            )
            result: WaterbirdsFinalResult
            if selector == "test_oracle":
                diagnostics = trained.diagnostic_metrics
                if diagnostics is None:
                    raise AssertionError("test-oracle final run lacks test records")
                result = WaterbirdsTestOracleRunResult(
                    schema_version="grit.waterbirds-run-result/v3",
                    result_kind="waterbirds_test_oracle_diagnostic",
                    status="succeeded",
                    run_id=trained.run_id,
                    final_seed=task.seed,
                    resolved_config=runtime.config,
                    resolved_config_digest=runtime.config.canonical_digest(),
                    code=current_code_provenance(),
                    environment=current_environment_provenance(),
                    validation_metrics=trained.validation_metrics,
                    diagnostic_metrics=diagnostics,
                    test_oracle_candidate_selection=frozen,
                    test_oracle_checkpoint_selection=frozen_checkpoint,
                    restoration=restoration,
                    selected_checkpoint_manifest_digest=(
                        checkpoint_manifest.canonical_digest()
                    ),
                    artifacts=artifacts,
                )
            else:
                handle = full_cache.issue_final_handle(
                    run_id=trained.run_id,
                    candidate_id=task.candidate.candidate_id,
                    method_id=task.candidate.method_id,
                    scientific_config_digest=task.candidate.scientific_config_digest,
                    seed=task.seed,
                    projection_rank=task.candidate.requested_rank,
                )
                final_view = handle.open(frozen, frozen_checkpoint, restoration)
                final_table = full_cache.verify_final_view(final_view)
                predictions = trained.algorithm.predict(final_table.features)
                final_metric = compute_waterbirds_final_metric(
                    final_view,
                    predictions,
                    adjusted_weights=weights,
                    record_id=f"metric:{trained.run_id}:test",
                )
                result = WaterbirdsRunResult(
                    schema_version="grit.waterbirds-run-result/v3",
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
                    artifacts=artifacts,
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
                diagnostic_metrics=trained.diagnostic_metrics,
                checkpoint_decision=decision,
                final_result_relative_path="final-result.json",
                final_result_digest=result.canonical_digest(),
                final_result=result,
            )
            persist_canonical_artifact(run_root / "result.json", completed)
            return completed

        return execute_final

    def persist_summary(summary: WaterbirdsProductionSummary, root: Path) -> None:
        persist_canonical_artifact(
            root / "summaries" / "waterbirds-summary.json", summary
        )
        if (
            summary.paired_worst_group_by_seed is not None
            and summary.paired_worst_group_summary is not None
        ):
            persist_canonical_artifact(
                root / "summaries" / "waterbirds-paired-differences.json",
                WaterbirdsPairedSummaryArtifact(
                    schema_version="grit.waterbirds-paired-summary/v1",
                    configured_final_seeds=plan.seeds.stages.final,
                    paired_worst_group_by_seed=summary.paired_worst_group_by_seed,
                    paired_worst_group_summary=summary.paired_worst_group_summary,
                ),
            )

    hooks = ProductionLifecycleHooks(
        coerce_runs=coerce_runs,
        execute_pre_final=execute_pre_final,
        make_finalists=lambda runs, root: _finalists(plan, runs, root),
        confirmation_tasks=confirmation_tasks,
        make_winners=make_winners,
        final_tasks=final_tasks,
        final_executor=final_executor,
        make_summary=lambda finalists, winners, runs: _summary(
            plan, finalists, winners, runs
        ),
        persist_summary=persist_summary,
        status=lambda: waterbirds_status_from_plan(plan),
    )
    return run_production_lifecycle(plan, limits, hooks)


def _finalists(
    plan: SearchPlan,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    output_root: Path,
) -> WaterbirdsFinalists:
    artifacts = compute_waterbirds_finalists(plan, runs)
    selector = waterbirds_selector(_search_config(plan))
    for method, finalists in artifacts.items():
        persist_canonical_artifact(
            output_root / "selection" / method / finalists_filename(selector),
            finalists,
        )
    return artifacts


def _selector_records(
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    method: WaterbirdsMethod,
    selector: WaterbirdsSelector,
) -> tuple[WaterbirdsSelectorRecord, ...]:
    return tuple(
        record
        for run in runs
        if run.task.candidate.method_id == method
        for record in run.selector_records(selector)
    )


def compute_waterbirds_finalists(
    plan: SearchPlan,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsFinalists:
    selector = waterbirds_selector(_search_config(plan))
    artifacts: WaterbirdsFinalists = {}
    for method in plan.methods:
        artifacts[method] = make_waterbirds_tuning_finalists(
            _selector_records(runs, method, selector), plan.seeds.stages, selector
        )
    return artifacts


def _freeze_winners(
    plan: SearchPlan,
    finalists: WaterbirdsFinalists,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
    output_root: Path,
) -> WaterbirdsWinners:
    winners = compute_waterbirds_winners(plan, finalists, runs)
    selector = waterbirds_selector(_search_config(plan))
    for method, frozen in winners.items():
        persist_canonical_artifact(
            output_root / "selection" / method / winner_filename(selector), frozen
        )
    return winners


def compute_waterbirds_winners(
    plan: SearchPlan,
    finalists: WaterbirdsFinalists,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsWinners:
    selector = waterbirds_selector(_search_config(plan))
    winners: WaterbirdsWinners = {}
    for method in plan.methods:
        artifact = finalists[method]
        finalist_ids = {item.candidate_id for item in artifact.ordered_candidates}
        records = tuple(
            item
            for item in _selector_records(runs, method, selector)
            if item.candidate_id in finalist_ids
        )
        decision = select_confirmed_waterbirds_candidate(
            records, artifact, plan.seeds.stages
        )
        winners[method] = freeze_waterbirds_candidate(
            decision, artifact, plan.seeds.stages
        )
    return winners


def materialize_waterbirds_candidate_config(
    plan: SearchPlan,
    candidate: SearchCandidate,
    projection: FittedLinearProjection | None,
    selector: WaterbirdsSelector = ORDINARY_WATERBIRDS_SELECTOR,
) -> WaterbirdsCandidateConfig:
    config = _search_config(plan)
    lineage = plan.resolved_config.lineage
    weights = lineage.adjusted_weight_spec_digest
    if weights is None:
        raise ValueError("Waterbirds plan lacks adjusted-weight lineage")
    pair_digest = (
        lineage.pair_manifest_digest
        if candidate.method_id in ("grit", "matchdg")
        else None
    )
    projection_digest = None
    rank = None
    tolerance = None
    if candidate.method_id == "grit":
        if projection is None or candidate.requested_rank is None:
            raise ValueError("Waterbirds GRIT candidate lacks fitted projection")
        projection_digest = projection.diagnostics.canonical_digest()
        rank = candidate.requested_rank
        tolerance = float(config.relative_singular_value_tolerance)
    elif projection is not None:
        raise ValueError("only Waterbirds GRIT candidates carry a projection")
    resolved = WaterbirdsCandidateConfig(
        schema_version="grit.waterbirds-candidate/v3",
        protocol_id="waterbirds_cf/v1",
        non_reportable=False,
        method_id=candidate.method_id,
        selector=selector,
        dataset_profile="production",
        dataset_manifest_digest=lineage.dataset_manifest_digest,
        feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
        normalization=config.normalization,
        adjusted_weight_spec_digest=weights,
        pair_manifest_digest=pair_digest,
        projection_diagnostics_digest=projection_digest,
        projection_rank=rank,
        relative_singular_value_tolerance=tolerance,
        algorithm=waterbirds_candidate_algorithm(
            config, candidate.method_id, candidate.settings()
        ),
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
    run_id = f"run:{task.task_id}"
    diagnostic_hook: DiagnosticEpochHook | None = None
    if runtime.config.test_oracle:
        if isinstance(cache, WaterbirdsTuningFeatureCache):
            raise AssertionError("test-oracle training requires the full cache")
        test_view = cache.open_test_oracle_table(runtime.config, run_id=run_id)

        def score_test(algorithm: LinearProbeAlgorithm, identity: CheckpointIdentity):
            return compute_waterbirds_diagnostic_metric(
                test_view,
                algorithm.predict(test_view.table.features),
                adjusted_weights=weights,
                record_id=f"metric:{identity.checkpoint_id}:test",
                candidate_id=task.candidate.candidate_id,
                method_id=task.candidate.method_id,
                scientific_config_digest=task.candidate.scientific_config_digest,
                checkpoint_id=identity.checkpoint_id,
                epoch=identity.epoch,
                seed_stage=task.stage,
                seed=task.seed,
                projection_rank=task.candidate.requested_rank,
            )

        diagnostic_hook = score_test
    return train_waterbirds_linear_probe(
        cache.training_table(),
        cache.validation_table(),
        weights,
        runtime.config.training,
        run_id=run_id,
        candidate_id=task.candidate.candidate_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        seed_stage=task.stage,
        seed=task.seed,
        method=_waterbirds_method(cache, runtime, task),
        diagnostic_hook=diagnostic_hook,
    )


def _waterbirds_method(
    cache: WaterbirdsFeatureCache | WaterbirdsTuningFeatureCache,
    runtime: _RuntimeCandidate,
    task: SearchRunTask,
) -> LinearProbeTrainingMethod:
    """Bind one planned candidate's algorithm config to the shared trainer.

    Backgrounds and groups are read through the cache's method-definition accessors;
    ERM, GRIT, SWAD, and MatchDG never touch them.
    """

    algorithm = runtime.config.algorithm
    if isinstance(
        algorithm, RexAlgorithmConfig | IrmAlgorithmConfig | FishAlgorithmConfig
    ):
        environment_ids = cache.training_environment_ids()
        if isinstance(algorithm, RexAlgorithmConfig):
            return RexLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=WATERBIRDS_ENVIRONMENT_COUNT,
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
        if isinstance(algorithm, IrmAlgorithmConfig):
            return IrmLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=WATERBIRDS_ENVIRONMENT_COUNT,
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
        return FishLinearProbeMethod(
            environment_ids=environment_ids,
            environment_count=WATERBIRDS_ENVIRONMENT_COUNT,
            meta_step_size=float(algorithm.meta_step_size),
        )
    if isinstance(algorithm, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
        group_ids = cache.training_group_ids()
        if isinstance(algorithm, GroupDroAlgorithmConfig):
            return GroupDroLinearProbeMethod(
                group_ids=group_ids,
                group_count=WATERBIRDS_GROUP_COUNT,
                step_size=float(algorithm.adversarial_step_size),
            )
        return LisaLinearProbeMethod(
            group_ids=group_ids,
            group_count=WATERBIRDS_GROUP_COUNT,
            selection_prob=float(algorithm.selection_prob),
        )
    if isinstance(algorithm, SwadAlgorithmConfig):
        validation = cache.validation_table()
        return SwadLinearProbeMethod(
            loss_features=validation.features,
            loss_targets=validation.labels,
            tolerance_ratio=float(algorithm.tolerance_ratio),
            segment_updates=int(algorithm.segment_updates),
            n_converge=int(algorithm.n_converge),
            n_tolerance=int(algorithm.n_tolerance),
        )
    if isinstance(algorithm, MatchDgAlgorithmConfig):
        if runtime.pair_differences is None:
            raise AssertionError("MatchDG candidate lacks its oracle pair bank")
        return MatchDgLinearProbeMethod(
            pair_differences=runtime.pair_differences,
            latent_dim=int(algorithm.latent_dim),
            penalty_weight=float(algorithm.penalty_weight),
        )
    return OrdinaryLinearProbeMethod(
        method_id=task.candidate.method_id,
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
    if config.pair_manifest_digest is not None:
        values.append(
            WaterbirdsPairArtifactReference(
                artifact_id="waterbirds-oracle-pairs",
                kind="pair_manifest",
                relative_uri=os.path.relpath(inputs["pair_manifest"].path, root),
                digest=config.pair_manifest_digest,
                dataset_manifest_digest=config.dataset_manifest_digest,
            )
        )
    if task.candidate.method_id == "grit":
        projection = runtime.projection
        if projection is None or config.pair_manifest_digest is None:
            raise AssertionError("Waterbirds GRIT artifact lineage disappeared")
        values.append(
            WaterbirdsProjectionArtifactReference(
                artifact_id=f"projection:grit:{task.candidate.requested_rank}",
                kind="projection_diagnostics",
                relative_uri=(
                    f"projections/grit-rank-{task.candidate.requested_rank}.json"
                ),
                digest=projection.diagnostics.canonical_digest(),
                pair_manifest_digest=config.pair_manifest_digest,
                feature_cache_manifest_digest=config.feature_cache_manifest_digest,
                normalization=config.normalization,
                requested_rank=projection.diagnostics.requested_rank,
            )
        )
    return tuple(values)


def _summary(
    plan: SearchPlan,
    finalists: WaterbirdsFinalists,
    winners: WaterbirdsWinners,
    runs: tuple[WaterbirdsCompletedStageRun, ...],
) -> WaterbirdsProductionSummary:
    selector = waterbirds_selector(_search_config(plan))
    by_identity = {(run.task.candidate.method_id, run.task.seed): run for run in runs}
    methods: list[WaterbirdsProductionMethodSummary] = []
    for method in plan.methods:
        paths: list[tuple[int, str]] = []
        worst: list[tuple[int, float]] = []
        adjusted: list[tuple[int, float]] = []
        raw: list[tuple[int, float]] = []
        for seed in plan.seeds.stages.final:
            run = by_identity[(method, seed)]
            result = run.final_result
            if result is None:
                raise AssertionError("completed Waterbirds final run lacks result")
            metric = result.reported_test_metric
            paths.append((seed, f"{run.task.relative_directory}/final-result.json"))
            worst.append((seed, float(metric.worst_group_accuracy)))
            adjusted.append((seed, float(metric.adjusted_average_accuracy)))
            raw.append((seed, float(metric.raw_average_accuracy)))
        top = finalists[method].ordered_candidates
        methods.append(
            WaterbirdsProductionMethodSummary(
                method_id=method,
                selector=selector,
                lineage=plan.resolved_config.lineage,
                selected_candidate_id=winners[method].candidate_id,
                finalist_candidate_ids=(
                    top[0].candidate_id,
                    top[1].candidate_id,
                    top[2].candidate_id,
                ),
                configured_final_seeds=plan.seeds.stages.final,
                result_paths_by_seed=tuple(paths),
                worst_group_by_seed=tuple(worst),
                adjusted_average_by_seed=tuple(adjusted),
                raw_average_by_seed=tuple(raw),
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
    by_method = {item.method_id: item for item in methods}
    paired: tuple[tuple[int, float], ...] | None = None
    paired_summary = None
    if "erm" in by_method and "grit" in by_method:
        erm_values = dict(by_method["erm"].worst_group_by_seed)
        grit_values = dict(by_method["grit"].worst_group_by_seed)
        paired = tuple(
            (seed, float(grit_values[seed]) - float(erm_values[seed]))
            for seed in plan.seeds.stages.final
        )
        paired_summary = make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(value for _, value in paired),
        )
    return WaterbirdsProductionSummary(
        schema_version="grit.waterbirds-production-summary/v2",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        lineage=plan.resolved_config.lineage,
        selector=selector,
        methods=tuple(methods),
        paired_worst_group_by_seed=paired,
        paired_worst_group_summary=paired_summary,
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
    if manifest.canonical_digest() != plan.resolved_config.lineage.pair_manifest_digest:
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


def waterbirds_status_from_plan(plan: SearchPlan) -> ProductionSearchStatus:
    config = _search_config(plan)
    selector = waterbirds_selector(config)
    by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[WaterbirdsCompletedStageRun, ...]:
        if any(not isinstance(run, WaterbirdsCompletedStageRun) for run in runs):
            raise ValueError("Waterbirds status received another dataset result")
        return cast(tuple[WaterbirdsCompletedStageRun, ...], runs)

    def finalists_complete(finalists: WaterbirdsFinalists, root: Path) -> bool:
        count = 0
        for method, expected in finalists.items():
            path = root / "selection" / method / finalists_filename(selector)
            if not path.exists():
                continue
            count += 1
            observed = WaterbirdsTuningFinalists.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    "Waterbirds finalist artifact does not match canonical tuning "
                    f"results: {path}"
                )
        return count == len(plan.methods)

    def confirmation_tasks(
        finalists: WaterbirdsFinalists,
    ) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_search_task(
                plan,
                by_id[item.candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for item in finalists[method].ordered_candidates
            for seed in config.seeds.stages.confirmation
        )

    def winner_count(winners: WaterbirdsWinners, root: Path) -> int:
        count = 0
        for method, expected in winners.items():
            path = root / "selection" / method / winner_filename(selector)
            if not path.exists():
                continue
            count += 1
            observed = FrozenWaterbirdsCandidate.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    "Waterbirds frozen winner does not match canonical confirmation "
                    f"results: {path}"
                )
        return count

    def final_tasks(winners: WaterbirdsWinners) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(plan, by_id[winner.candidate_id], seed, winner)
            for winner in winners.values()
            for seed in config.seeds.stages.final
        )

    hooks = ProductionStatusHooks(
        coerce_runs=coerce_runs,
        compute_finalists=lambda runs: compute_waterbirds_finalists(plan, runs),
        finalists_complete=finalists_complete,
        confirmation_tasks=confirmation_tasks,
        compute_winners=lambda finalists, runs: compute_waterbirds_winners(
            plan, finalists, runs
        ),
        winner_count=winner_count,
        expected_winner_count=len(plan.methods),
        final_tasks=final_tasks,
    )
    return production_lifecycle_status(plan, hooks)
