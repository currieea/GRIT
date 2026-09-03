"""CMNIST-specific search orchestration, selection, and summaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
    CmnistArtifactLineageConfig,
    CmnistDatasetConfig,
    CmnistSourceCounts,
    CpuRuntimeConfig,
    DisabledPairsConfig,
    DisabledProjectionConfig,
    ErmAlgorithmConfig,
    FrozenFeatureConfig,
    GritAlgorithmConfig,
    LinearProbeTrainingConfig,
    LinearProjectionConfig,
    OraclePairsConfig,
    OrdinaryExperimentConfig,
    OrdinarySelectionConfig,
)
from grit.data.cmnist import CmnistOraclePairManifest
from grit.features.cmnist import (
    CmnistFeatureCache,
    CmnistTuningFeatureCache,
    load_cmnist_feature_cache,
    load_cmnist_tuning_feature_cache,
)
from grit.lifecycle import open_final_test, record_final_accuracy
from grit.methods.checkpoints import restore_checkpoint
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import (
    MethodId,
    PersistedLinearCheckpointStore,
    TrainedLinearProbeRun,
    evaluate_accuracy,
    persist_selected_linear_checkpoint,
    train_linear_probe,
)
from grit.results import ArtifactReference, OrdinaryRunResult, SucceededStatus
from grit.schemas import CmnistSelector, SeedStage
from grit.search.outputs import (
    CmnistFinalSeedObservation,
    CmnistMethodSelectorSummary,
    CmnistPairedSeedDifference,
    CmnistPairedSelectorSummary,
    CmnistProductionSummary,
    make_cmnist_accuracy_summary,
)
from grit.search.plan import (
    CmnistProductionSearchConfig,
    SearchCandidate,
    SearchPlan,
    current_code_provenance,
    current_environment_provenance,
)
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    complete_outputs_valid,
    limited_tuning_candidates,
    persist_canonical_artifact,
    planned_candidate,
    write_experiment_index,
)
from grit.search.scheduler import (
    CmnistCompletedStageRun,
    CompletedStageRun,
    LocalRunScheduler,
    SearchRunTask,
    StageExecutor,
    make_final_search_task,
    make_search_task,
)
from grit.selection.cmnist import (
    FinalistUnion,
    FrozenCandidateSelection,
    TuningFinalistsArtifact,
    freeze_candidate,
    freeze_final_checkpoint,
    make_finalist_union,
    make_tuning_finalists,
    select_checkpoint,
    select_confirmed_candidate,
)

NonEmptyStr: TypeAlias = str


@dataclass(frozen=True, slots=True)
class _CmnistRuntimeCandidate:
    planned: SearchCandidate
    config: OrdinaryExperimentConfig
    projection: FittedLinearProjection | None

def run_cmnist_search(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
) -> CmnistProductionSummary | ProductionSearchStatus:
    config = _cmnist_search_config(plan)
    output_root = Path(plan.resolved_config.output_root)
    cache = _load_cmnist_cache(
        plan, tuning_only=limits is not None and limits.stop_after == "tuning"
    )
    pair_manifest = _cmnist_pair_manifest(plan)
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
            tasks,
            execute,
            max_new_runs=allowance,
        )
        if remaining is not None:
            remaining -= newly_executed
        return results

    def runtime_candidate(candidate: SearchCandidate) -> _CmnistRuntimeCandidate:
        projection: FittedLinearProjection | None = None
        if candidate.method_id == "grit":
            rank = candidate.requested_rank
            if rank is None:
                raise AssertionError("planned GRIT candidate lacks a rank")
            projection = projections.get(rank)
            if projection is None:
                red, green = cache.pair_tables()
                # Pairs are stored in seeded hash order, so the first N form the
                # N-pair bank for the same seed.
                projection = fit_linear_projection(
                    red.features[: config.pair_count],
                    green.features[: config.pair_count],
                    requested_rank=rank,
                    pair_manifest_digest=pair_manifest.canonical_digest(),
                    feature_cache_manifest_digest=cache.manifest.canonical_digest(),
                    relative_singular_value_tolerance=(
                        config.relative_singular_value_tolerance
                    ),
                )
                persist_canonical_artifact(
                    output_root / "projections" / f"grit-rank-{rank}.json",
                    projection.diagnostics,
                )
                projections[rank] = projection
        resolved = materialize_cmnist_candidate_config(
            plan,
            candidate,
            CmnistSelector.PRIMARY_ROBUST,
        )
        return _CmnistRuntimeCandidate(candidate, resolved, projection)

    def execute_pre_final(task: SearchRunTask, _run_root: Path) -> CompletedStageRun:
        runtime = runtime_candidate(task.candidate)
        trained = _train_cmnist_task(cache, runtime, task)
        decisions = (
            select_checkpoint(
                trained.validation_metrics, CmnistSelector.PRIMARY_ROBUST
            ),
            select_checkpoint(
                trained.validation_metrics, CmnistSelector.SECONDARY_SOURCE
            ),
        )
        return CmnistCompletedStageRun(
            schema_version="grit.cmnist-search-stage-run/v1",
            dataset="cmnist",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            checkpoint_decisions=decisions,
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
        tuple[CmnistCompletedStageRun, ...],
        run_stage(tuning_tasks, execute_pre_final),
    )
    if limits is not None and (
        limits.stop_after == "tuning" or len(tuning_runs) != len(tuning_tasks)
    ):
        return cmnist_status_from_plan(plan)
    finalists, unions = _cmnist_finalists(plan, tuning_runs, output_root)
    candidates_by_id = {item.candidate_id: item for item in plan.candidates}
    confirmation_ids = {
        method: union.confirmation_candidate_ids for method, union in unions.items()
    }
    confirmation_tasks = tuple(
        make_search_task(
            plan,
            candidates_by_id[candidate_id],
            SeedStage.CONFIRMATION,
            seed,
        )
        for method in ("erm", "grit")
        for candidate_id in confirmation_ids[method]
        for seed in config.seeds.stages.confirmation
    )
    confirmation_runs = cast(
        tuple[CmnistCompletedStageRun, ...],
        run_stage(confirmation_tasks, execute_pre_final),
    )
    if limits is not None and len(confirmation_runs) != len(confirmation_tasks):
        return cmnist_status_from_plan(plan)
    winners = _freeze_cmnist_winners(
        plan, finalists, confirmation_runs, output_root
    )
    if isinstance(cache, CmnistTuningFeatureCache):
        raise AssertionError("tuning-only CMNIST execution reached final stage")

    def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
        selector = CmnistSelector(task.selector)
        frozen = winners[(task.candidate.method_id, selector)]
        runtime = runtime_candidate(task.candidate)
        runtime = _CmnistRuntimeCandidate(
            runtime.planned,
            materialize_cmnist_candidate_config(plan, task.candidate, selector),
            runtime.projection,
        )
        trained = _train_cmnist_task(cache, runtime, task)
        decision = select_checkpoint(trained.validation_metrics, selector)
        frozen_checkpoint = freeze_final_checkpoint(decision, frozen)
        selected = trained.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
        checkpoint_root = run_root / "selected-checkpoint"
        checkpoint_manifest = persist_selected_linear_checkpoint(
            selected, checkpoint_root
        )
        persisted = PersistedLinearCheckpointStore(checkpoint_root)
        restoration = restore_checkpoint(
            frozen_checkpoint,
            persisted,
            trained.algorithm.restore_inference_state,
        )
        handle = cache.issue_final_handle(
            run_id=trained.run_id,
            candidate_id=task.candidate.candidate_id,
            scientific_config_digest=task.candidate.scientific_config_digest,
        )
        final_view = open_final_test(
            handle, frozen, frozen_checkpoint, restoration
        )
        final_table = cache.open_final_table(final_view)
        final_metric = record_final_accuracy(
            final_view,
            record_id=f"metric:{trained.run_id}:test_ood",
            value=evaluate_accuracy(trained.algorithm, final_table),
            sample_count=len(final_table.source_ids),
        )
        result = OrdinaryRunResult(
            schema_version="grit.run-result/v1",
            result_kind="ordinary",
            run_id=trained.run_id,
            resolved_config=runtime.config,
            resolved_config_digest=runtime.config.canonical_digest(),
            status=SucceededStatus(kind="succeeded"),
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            candidate_selection=frozen,
            checkpoint_selection=frozen_checkpoint,
            restoration=restoration,
            final_test_metrics=(final_metric,),
            artifacts=_cmnist_result_artifacts(
                plan,
                task,
                checkpoint_manifest.canonical_digest(),
                runtime,
            ),
        )
        persist_canonical_artifact(run_root / "final-result.json", result)
        completed = CmnistCompletedStageRun(
            schema_version="grit.cmnist-search-stage-run/v1",
            dataset="cmnist",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            checkpoint_decisions=(decision,),
            final_result_relative_path="final-result.json",
            final_result_digest=result.canonical_digest(),
            final_result=result,
        )
        persist_canonical_artifact(run_root / "result.json", completed)
        return completed

    final_tasks = tuple(
        make_final_search_task(
            plan,
            candidates_by_id[frozen.candidate_id],
            seed,
            frozen,
        )
        for method in ("erm", "grit")
        for selector in (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        )
        for frozen in (winners[(method, selector)],)
        for seed in config.seeds.stages.final
    )
    final_runs = cast(
        tuple[CmnistCompletedStageRun, ...],
        run_stage(final_tasks, execute_final),
    )
    if limits is not None:
        return cmnist_status_from_plan(plan)
    summary = _cmnist_summary(plan, finalists, winners, final_runs)
    persist_canonical_artifact(
        output_root / "summaries" / "cmnist-summary.json", summary
    )
    for paired in summary.paired_selectors:
        persist_canonical_artifact(
            output_root
            / "summaries"
            / f"cmnist-{paired.selector.value}-paired-differences.json",
            paired,
        )
    write_experiment_index(plan, output_root)
    return summary

def _cmnist_finalists(
    plan: SearchPlan,
    runs: tuple[CmnistCompletedStageRun, ...],
    output_root: Path,
) -> tuple[
    dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    dict[MethodId, FinalistUnion],
]:
    artifacts, unions = compute_cmnist_finalists(plan, runs)
    for method in ("erm", "grit"):
        primary = artifacts[(method, CmnistSelector.PRIMARY_ROBUST)]
        secondary = artifacts[(method, CmnistSelector.SECONDARY_SOURCE)]
        union = unions[method]
        root = output_root / "selection" / method
        persist_canonical_artifact(
            root / "primary-tuning-finalists.json", primary
        )
        persist_canonical_artifact(
            root / "secondary-tuning-finalists.json", secondary
        )
        persist_canonical_artifact(root / "confirmation-union.json", union)
    return artifacts, unions

def compute_cmnist_finalists(
    plan: SearchPlan,
    runs: tuple[CmnistCompletedStageRun, ...],
) -> tuple[
    dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    dict[MethodId, FinalistUnion],
]:
    metrics = tuple(metric for run in runs for metric in run.validation_metrics)
    artifacts: dict[
        tuple[MethodId, CmnistSelector], TuningFinalistsArtifact
    ] = {}
    unions: dict[MethodId, FinalistUnion] = {}
    for method in ("erm", "grit"):
        method_metrics = tuple(item for item in metrics if item.method_id == method)
        primary = make_tuning_finalists(
            method_metrics,
            CmnistSelector.PRIMARY_ROBUST,
            plan.seeds.stages,
        )
        secondary = make_tuning_finalists(
            method_metrics,
            CmnistSelector.SECONDARY_SOURCE,
            plan.seeds.stages,
        )
        union = make_finalist_union(primary, secondary)
        artifacts[(method, CmnistSelector.PRIMARY_ROBUST)] = primary
        artifacts[(method, CmnistSelector.SECONDARY_SOURCE)] = secondary
        unions[method] = union
    return artifacts, unions

def _freeze_cmnist_winners(
    plan: SearchPlan,
    finalists: dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    runs: tuple[CmnistCompletedStageRun, ...],
    output_root: Path,
) -> dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection]:
    winners = compute_cmnist_winners(plan, finalists, runs)
    for (method, selector), frozen in winners.items():
        persist_canonical_artifact(
            output_root
            / "selection"
            / method
            / f"{selector.value}-winner.json",
            frozen,
        )
    return winners

def compute_cmnist_winners(
    plan: SearchPlan,
    finalists: dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    runs: tuple[CmnistCompletedStageRun, ...],
) -> dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection]:
    metrics = tuple(metric for run in runs for metric in run.validation_metrics)
    winners: dict[
        tuple[MethodId, CmnistSelector], FrozenCandidateSelection
    ] = {}
    for method in ("erm", "grit"):
        for selector in (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            artifact = finalists[(method, selector)]
            finalist_ids = {
                item.candidate_id for item in artifact.ordered_candidates
            }
            records = tuple(
                item
                for item in metrics
                if item.method_id == method and item.candidate_id in finalist_ids
            )
            decision = select_confirmed_candidate(
                records, artifact, plan.seeds.stages
            )
            frozen = freeze_candidate(decision, artifact, plan.seeds.stages)
            winners[(method, selector)] = frozen
    return winners

def materialize_cmnist_candidate_config(
    plan: SearchPlan,
    candidate: SearchCandidate,
    selector: CmnistSelector,
) -> OrdinaryExperimentConfig:
    config = _cmnist_search_config(plan)
    lineage = plan.resolved_config.lineage
    training = LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=config.batch_size,
        learning_rate=float(candidate.learning_rate),
        weight_decay=float(candidate.weight_decay),
        max_epochs=config.max_epochs,
    )
    if candidate.method_id == "erm":
        pairs = DisabledPairsConfig(kind="disabled")
        projection = DisabledProjectionConfig(kind="disabled")
        algorithm = ErmAlgorithmConfig(kind="erm")
        pair_digest = None
    else:
        rank = candidate.requested_rank
        if rank is None:
            raise AssertionError("planned GRIT candidate lacks a rank")
        pairs = OraclePairsConfig(
            kind="oracle",
            construction_id="cmnist-clean-oracle-pairs-v1",
            source_partition_ids=("train_e01_sources", "train_e02_sources"),
            pair_count=config.pair_count,
            pair_seed=config.seeds.pairs,
            orientation="red_minus_green",
        )
        projection = LinearProjectionConfig(
            kind="linear_pair_difference",
            requested_rank=rank,
            center_differences=False,
            relative_singular_value_tolerance=(
                config.relative_singular_value_tolerance
            ),
        )
        algorithm = GritAlgorithmConfig(kind="grit")
        pair_digest = lineage.pair_manifest_digest
    resolved = OrdinaryExperimentConfig(
        schema_version="grit.experiment/v1",
        run_kind="ordinary",
        experiment_name=config.experiment_name,
        protocol_id="cmnist/v1",
        reportable=True,
        dataset=CmnistDatasetConfig(
            dataset_id="cmnist",
            construction_method_id="cmnist-stratified-hash-v1",
            construction_seed=config.seeds.construction,
            label_flip_prob=0.25,
            source_counts=CmnistSourceCounts(
                train_e01=25_000,
                train_e02=25_000,
                validation=10_000,
                test=10_000,
            ),
            training_split_names=("train_e01", "train_e02"),
            validation_split_names=("val_e01", "val_e02", "val_e05"),
            final_test_split_name="test_ood",
        ),
        representation=FrozenFeatureConfig(
            kind="frozen_features",
            encoder_id="openai-clip-vit-b32",
            encoder_revision=OPENAI_CLIP_REVISION,
            weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
            preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
            feature_dimension=512,
            normalization=config.normalization,
        ),
        pairs=pairs,
        projection=projection,
        algorithm=algorithm,
        training=training,
        runtime=CpuRuntimeConfig(device="cpu", deterministic_algorithms=True),
        seed_sets=config.seeds.stages,
        artifact_lineage=CmnistArtifactLineageConfig(
            dataset_manifest_digest=lineage.dataset_manifest_digest,
            feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
            pair_manifest_digest=pair_digest,
        ),
        selection=OrdinarySelectionConfig(selector=selector),
    )
    if resolved.scientific_config_digest() != candidate.scientific_config_digest:
        raise ValueError("planned CMNIST candidate digest cannot be materialized")
    return resolved

def _train_cmnist_task(
    cache: CmnistFeatureCache | CmnistTuningFeatureCache,
    runtime: _CmnistRuntimeCandidate,
    task: SearchRunTask,
) -> TrainedLinearProbeRun:
    return train_linear_probe(
        cache.training_tables(),
        cache.validation_tables(),
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

def _cmnist_result_artifacts(
    plan: SearchPlan,
    task: SearchRunTask,
    checkpoint_manifest_digest: str,
    runtime: _CmnistRuntimeCandidate,
) -> tuple[ArtifactReference, ...]:
    root = Path(plan.resolved_config.output_root)
    inputs = {item.kind: item for item in plan.resolved_config.input_artifacts}
    values = [
        ArtifactReference(
            artifact_id="cmnist-dataset-manifest",
            kind="dataset_manifest",
            relative_uri=os.path.relpath(inputs["dataset_manifest"].path, root),
            digest=inputs["dataset_manifest"].digest,
        ),
        ArtifactReference(
            artifact_id="cmnist-feature-manifest",
            kind="feature_manifest",
            relative_uri=os.path.relpath(inputs["feature_manifest"].path, root),
            digest=inputs["feature_manifest"].digest,
        ),
        ArtifactReference(
            artifact_id=f"checkpoint:{task.task_id}",
            kind="selected_linear_checkpoint",
            relative_uri=(
                f"{task.relative_directory}/selected-checkpoint/manifest.json"
            ),
            digest=checkpoint_manifest_digest,
        ),
    ]
    if task.candidate.method_id == "grit":
        projection = runtime.projection
        if projection is None:
            raise AssertionError("GRIT result lacks projection")
        rank = task.candidate.requested_rank
        values.extend(
            (
                ArtifactReference(
                    artifact_id="cmnist-oracle-pair-manifest",
                    kind="pair_manifest",
                    relative_uri=os.path.relpath(
                        inputs["pair_manifest"].path, root
                    ),
                    digest=inputs["pair_manifest"].digest,
                ),
                ArtifactReference(
                    artifact_id=f"projection:grit:{rank}",
                    kind="projection_diagnostics",
                    relative_uri=f"projections/grit-rank-{rank}.json",
                    digest=projection.diagnostics.canonical_digest(),
                ),
            )
        )
    return tuple(values)

def _cmnist_summary(
    plan: SearchPlan,
    finalists: dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    winners: dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection],
    runs: tuple[CmnistCompletedStageRun, ...],
) -> CmnistProductionSummary:
    by_identity = {
        (
            run.task.candidate.method_id,
            CmnistSelector(run.task.selector),
            run.task.seed,
        ): run
        for run in runs
    }
    methods: list[CmnistMethodSelectorSummary] = []
    for method in ("erm", "grit"):
        for selector in (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            observations: list[CmnistFinalSeedObservation] = []
            for seed in plan.seeds.stages.final:
                run = by_identity[(method, selector, seed)]
                result = run.final_result
                if result is None or result.final_test_metrics is None:
                    raise AssertionError("completed final run lacks CMNIST metrics")
                metric = result.final_test_metrics[0]
                observations.append(
                    CmnistFinalSeedObservation(
                        seed=seed,
                        method_id=method,
                        selector=selector,
                        result_path=(
                            f"{run.task.relative_directory}/final-result.json"
                        ),
                        metric_record_id=metric.record_id,
                        test_ood_accuracy=metric.value,
                    )
                )
            values = tuple(
                float(item.test_ood_accuracy) for item in observations
            )
            top = finalists[(method, selector)].ordered_candidates
            methods.append(
                CmnistMethodSelectorSummary(
                    method_id=method,
                    selector=selector,
                    lineage=plan.resolved_config.lineage,
                    selected_candidate_id=winners[(method, selector)].candidate_id,
                    finalist_candidate_ids=(
                        top[0].candidate_id,
                        top[1].candidate_id,
                        top[2].candidate_id,
                    ),
                    configured_final_seeds=plan.seeds.stages.final,
                    final_observations=tuple(observations),
                    accuracy_summary=make_cmnist_accuracy_summary(
                        "test_ood_accuracy", values
                    ),
                )
            )
    paired: list[CmnistPairedSelectorSummary] = []
    method_map = {(item.method_id, item.selector): item for item in methods}
    for selector in (
        CmnistSelector.PRIMARY_ROBUST,
        CmnistSelector.SECONDARY_SOURCE,
    ):
        erm = method_map[("erm", selector)]
        grit = method_map[("grit", selector)]
        erm_values = {
            item.seed: float(item.test_ood_accuracy)
            for item in erm.final_observations
        }
        grit_values = {
            item.seed: float(item.test_ood_accuracy)
            for item in grit.final_observations
        }
        differences = tuple(
            CmnistPairedSeedDifference(
                seed=seed,
                selector=selector,
                grit_minus_erm_test_ood_accuracy=(
                    grit_values[seed] - erm_values[seed]
                ),
            )
            for seed in plan.seeds.stages.final
        )
        paired.append(
            CmnistPairedSelectorSummary(
                selector=selector,
                configured_final_seeds=plan.seeds.stages.final,
                paired_differences=differences,
                difference_summary=make_cmnist_accuracy_summary(
                    "grit_minus_erm_test_ood_accuracy",
                    tuple(
                        float(item.grit_minus_erm_test_ood_accuracy)
                        for item in differences
                    ),
                ),
            )
        )
    return CmnistProductionSummary(
        schema_version="grit.cmnist-production-summary/v1",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        lineage=plan.resolved_config.lineage,
        methods=(methods[0], methods[1], methods[2], methods[3]),
        paired_selectors=(paired[0], paired[1]),
    )

def _load_cmnist_cache(
    plan: SearchPlan,
    *,
    tuning_only: bool = False,
) -> CmnistFeatureCache | CmnistTuningFeatureCache:
    paths = {
        item.kind: Path(item.path)
        for item in plan.resolved_config.input_artifacts
    }
    loader = (
        load_cmnist_tuning_feature_cache
        if tuning_only
        else load_cmnist_feature_cache
    )
    cache = loader(
        paths["feature_manifest"].parent,
        expected_source_manifest_digest=(
            plan.resolved_config.lineage.dataset_manifest_digest
        ),
        expected_pair_manifest_digest=plan.resolved_config.lineage.pair_manifest_digest,
        expected_normalization=plan.normalization,
    )
    if (
        cache.manifest.canonical_digest()
        != plan.resolved_config.lineage.feature_cache_manifest_digest
    ):
        raise ValueError("CMNIST feature manifest changed after planning")
    return cache

def _cmnist_pair_manifest(plan: SearchPlan) -> CmnistOraclePairManifest:
    path = next(
        Path(item.path)
        for item in plan.resolved_config.input_artifacts
        if item.kind == "pair_manifest"
    )
    manifest = CmnistOraclePairManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if (
        manifest.canonical_digest()
        != plan.resolved_config.lineage.pair_manifest_digest
    ):
        raise ValueError("CMNIST pair manifest changed after planning")
    return manifest

def _cmnist_search_config(plan: SearchPlan) -> CmnistProductionSearchConfig:
    config = plan.resolved_config.config
    if not isinstance(config, CmnistProductionSearchConfig):
        raise TypeError("CMNIST production runner received another dataset")
    return config

def cmnist_status_from_plan(plan: SearchPlan) -> ProductionSearchStatus:
    config = _cmnist_search_config(plan)
    scheduler = LocalRunScheduler(Path(plan.resolved_config.output_root), plan)
    tuning = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in plan.candidates
        for seed in config.seeds.stages.tuning
    )
    status = scheduler.status(tuning)
    complete = len(status.complete_task_ids)
    if complete != len(tuning):
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="cmnist",
            plan_digest=plan.canonical_digest(),
            phase="tuning",
            tuning_expected=len(tuning),
            tuning_complete=complete,
            confirmation_expected=0,
            confirmation_complete=0,
            frozen_winner_count=0,
            final_expected=0,
            final_complete=0,
        )
    tuning_runs_untyped = scheduler.completed_results(tuning)
    if any(
        not isinstance(run, CmnistCompletedStageRun)
        for run in tuning_runs_untyped
    ):
        raise ValueError("CMNIST tuning stage contains another dataset result")
    tuning_runs = cast(tuple[CmnistCompletedStageRun, ...], tuning_runs_untyped)
    expected_finalists, expected_unions = compute_cmnist_finalists(
        plan, tuning_runs
    )
    root = Path(plan.resolved_config.output_root)
    finalist_artifact_count = 0
    for method in ("erm", "grit"):
        selection_root = root / "selection" / method
        finalist_paths = (
            (
                selection_root / "primary-tuning-finalists.json",
                TuningFinalistsArtifact,
                expected_finalists[(method, CmnistSelector.PRIMARY_ROBUST)],
            ),
            (
                selection_root / "secondary-tuning-finalists.json",
                TuningFinalistsArtifact,
                expected_finalists[(method, CmnistSelector.SECONDARY_SOURCE)],
            ),
            (
                selection_root / "confirmation-union.json",
                FinalistUnion,
                expected_unions[method],
            ),
        )
        for path, artifact_type, expected in finalist_paths:
            if not path.exists():
                continue
            finalist_artifact_count += 1
            observed = artifact_type.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    f"CMNIST selection artifact does not match canonical tuning "
                    f"results: {path}"
                )
    if finalist_artifact_count != 6:
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="cmnist",
            plan_digest=plan.canonical_digest(),
            phase="confirmation",
            tuning_expected=len(tuning),
            tuning_complete=complete,
            confirmation_expected=0,
            confirmation_complete=0,
            frozen_winner_count=0,
            final_expected=0,
            final_complete=0,
        )
    by_id = {item.candidate_id: item for item in plan.candidates}
    confirmation = tuple(
        make_search_task(
            plan,
            planned_candidate(by_id, candidate_id),
            SeedStage.CONFIRMATION,
            seed,
        )
        for method in ("erm", "grit")
        for candidate_id in expected_unions[method].confirmation_candidate_ids
        for seed in config.seeds.stages.confirmation
    )
    confirmation_status = scheduler.status(confirmation)
    confirmation_complete = len(confirmation_status.complete_task_ids)
    if confirmation_complete != len(confirmation):
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="cmnist",
            plan_digest=plan.canonical_digest(),
            phase="confirmation",
            tuning_expected=len(tuning),
            tuning_complete=complete,
            confirmation_expected=len(confirmation),
            confirmation_complete=confirmation_complete,
            frozen_winner_count=0,
            final_expected=0,
            final_complete=0,
        )
    confirmation_runs_untyped = scheduler.completed_results(confirmation)
    if any(
        not isinstance(run, CmnistCompletedStageRun)
        for run in confirmation_runs_untyped
    ):
        raise ValueError("CMNIST confirmation stage contains another dataset result")
    confirmation_runs = cast(
        tuple[CmnistCompletedStageRun, ...], confirmation_runs_untyped
    )
    expected_winners = compute_cmnist_winners(
        plan, expected_finalists, confirmation_runs
    )
    winner_count = 0
    for method in ("erm", "grit"):
        for selector in (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            path = root / "selection" / method / f"{selector.value}-winner.json"
            if path.exists():
                winner_count += 1
                observed = FrozenCandidateSelection.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if observed != expected_winners[(method, selector)]:
                    raise ValueError(
                        "CMNIST frozen winner does not match canonical confirmation "
                        f"results: {path}"
                    )
    if winner_count != 4:
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="cmnist",
            plan_digest=plan.canonical_digest(),
            phase="final",
            tuning_expected=len(tuning),
            tuning_complete=complete,
            confirmation_expected=len(confirmation),
            confirmation_complete=confirmation_complete,
            frozen_winner_count=winner_count,
            final_expected=0,
            final_complete=0,
        )
    final = tuple(
        make_final_search_task(
            plan,
            planned_candidate(by_id, winner.candidate_id),
            seed,
            winner,
        )
        for winner in expected_winners.values()
        for seed in config.seeds.stages.final
    )
    final_status = scheduler.status(final)
    final_complete = len(final_status.complete_task_ids)
    phase: Literal["final", "complete"] = "final"
    if final_complete == len(final) and complete_outputs_valid(plan):
        phase = "complete"
    return ProductionSearchStatus(
        schema_version="grit.production-search-status/v1",
        dataset="cmnist",
        plan_digest=plan.canonical_digest(),
        phase=phase,
        tuning_expected=len(tuning),
        tuning_complete=complete,
        confirmation_expected=len(confirmation),
        confirmation_complete=confirmation_complete,
        frozen_winner_count=4,
        final_expected=len(final),
        final_complete=final_complete,
    )
