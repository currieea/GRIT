"""CMNIST-specific search orchestration, selection, and summaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

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
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LinearProbeTrainingConfig,
    LinearProjectionConfig,
    OraclePairsConfig,
    OrdinaryExperimentConfig,
    OrdinarySelectionConfig,
    RexAlgorithmConfig,
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
from grit.methods.groupdro import CMNIST_GROUP_COUNT, cmnist_group_ids
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import (
    GroupDroLinearProbeMethod,
    IrmLinearProbeMethod,
    MethodId,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    TrainedLinearProbeRun,
    evaluate_accuracy,
    persist_selected_linear_checkpoint,
    train_linear_probe,
)
from grit.results import ArtifactReference, OrdinaryRunResult, SucceededStatus
from grit.schemas import CmnistSelector, SeedStage
from grit.search.lifecycle import (
    ProductionLifecycleHooks,
    ProductionStatusHooks,
    production_lifecycle_status,
    run_production_lifecycle,
)
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
    persist_canonical_artifact,
)
from grit.search.scheduler import (
    CmnistCompletedStageRun,
    CompletedStageRun,
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
    pair_manifest = _cmnist_pair_manifest(plan) if "grit" in plan.methods else None
    projections: dict[int, FittedLinearProjection] = {}

    def runtime_candidate(candidate: SearchCandidate) -> _CmnistRuntimeCandidate:
        projection: FittedLinearProjection | None = None
        if candidate.method_id == "grit":
            rank = candidate.requested_rank
            if rank is None:
                raise AssertionError("planned GRIT candidate lacks a rank")
            if pair_manifest is None:
                raise AssertionError("planned GRIT search lacks its pair manifest")
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

    candidates_by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[CmnistCompletedStageRun, ...]:
        if any(not isinstance(run, CmnistCompletedStageRun) for run in runs):
            raise ValueError("CMNIST lifecycle received another dataset result")
        return cast(tuple[CmnistCompletedStageRun, ...], runs)

    def confirmation_tasks(
        state: tuple[
            dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
            dict[MethodId, FinalistUnion],
        ],
    ) -> tuple[SearchRunTask, ...]:
        _, unions = state
        return tuple(
            make_search_task(
                plan,
                candidates_by_id[candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for candidate_id in unions[method].confirmation_candidate_ids
            for seed in config.seeds.stages.confirmation
        )

    def make_winners(
        state: tuple[
            dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
            dict[MethodId, FinalistUnion],
        ],
        runs: tuple[CmnistCompletedStageRun, ...],
        root: Path,
    ) -> dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection]:
        finalists, _ = state
        return _freeze_cmnist_winners(plan, finalists, runs, root)

    def final_tasks(
        winners: dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection],
    ) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                candidates_by_id[frozen.candidate_id],
                seed,
                frozen,
            )
            for method in plan.methods
            for selector in (
                CmnistSelector.PRIMARY_ROBUST,
                CmnistSelector.SECONDARY_SOURCE,
            )
            for frozen in (winners[(method, selector)],)
            for seed in config.seeds.stages.final
        )

    def final_executor(
        winners: dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection],
    ) -> StageExecutor:
        if isinstance(cache, CmnistTuningFeatureCache):
            raise AssertionError("tuning-only CMNIST execution reached final stage")
        full_cache = cache

        def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
            selector = CmnistSelector(task.selector)
            frozen = winners[(task.candidate.method_id, selector)]
            runtime = runtime_candidate(task.candidate)
            runtime = _CmnistRuntimeCandidate(
                runtime.planned,
                materialize_cmnist_candidate_config(plan, task.candidate, selector),
                runtime.projection,
            )
            trained = _train_cmnist_task(full_cache, runtime, task)
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
            handle = full_cache.issue_final_handle(
                run_id=trained.run_id,
                candidate_id=task.candidate.candidate_id,
                scientific_config_digest=task.candidate.scientific_config_digest,
            )
            final_view = open_final_test(
                handle, frozen, frozen_checkpoint, restoration
            )
            final_table = full_cache.open_final_table(final_view)
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

        return execute_final

    def make_summary(
        state: tuple[
            dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
            dict[MethodId, FinalistUnion],
        ],
        winners: dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection],
        runs: tuple[CmnistCompletedStageRun, ...],
    ) -> CmnistProductionSummary:
        finalists, _ = state
        return _cmnist_summary(plan, finalists, winners, runs)

    def persist_summary(summary: CmnistProductionSummary, root: Path) -> None:
        persist_canonical_artifact(
            root / "summaries" / "cmnist-summary.json", summary
        )
        for paired in summary.paired_selectors:
            persist_canonical_artifact(
                root
                / "summaries"
                / f"cmnist-{paired.selector.value}-paired-differences.json",
                paired,
            )

    hooks = ProductionLifecycleHooks(
        coerce_runs=coerce_runs,
        execute_pre_final=execute_pre_final,
        make_finalists=lambda runs, root: _cmnist_finalists(plan, runs, root),
        confirmation_tasks=confirmation_tasks,
        make_winners=make_winners,
        final_tasks=final_tasks,
        final_executor=final_executor,
        make_summary=make_summary,
        persist_summary=persist_summary,
        status=lambda: cmnist_status_from_plan(plan),
    )
    return run_production_lifecycle(plan, limits, hooks)

def _cmnist_finalists(
    plan: SearchPlan,
    runs: tuple[CmnistCompletedStageRun, ...],
    output_root: Path,
) -> tuple[
    dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    dict[MethodId, FinalistUnion],
]:
    artifacts, unions = compute_cmnist_finalists(plan, runs)
    for method in plan.methods:
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
    for method in plan.methods:
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
    for method in plan.methods:
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
    elif candidate.method_id == "grit":
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
    elif candidate.method_id == "groupdro":
        step_size = candidate.groupdro_step_size
        if step_size is None:
            raise AssertionError("planned GroupDRO candidate lacks a step size")
        pairs = DisabledPairsConfig(kind="disabled")
        projection = DisabledProjectionConfig(kind="disabled")
        algorithm = GroupDroAlgorithmConfig(
            kind="groupdro",
            group_definition="target_color",
            adversarial_step_size=step_size,
            sampling="inverse_group_frequency_with_replacement",
            generalization_adjustment=0.0,
            normalize_loss=False,
        )
        pair_digest = None
    elif candidate.method_id == "rex":
        penalty_weight = candidate.penalty_weight
        anneal_updates = config.search_space.rex_penalty_anneal_updates
        if penalty_weight is None or anneal_updates is None:
            raise AssertionError("planned REx candidate lacks its settings")
        pairs = DisabledPairsConfig(kind="disabled")
        projection = DisabledProjectionConfig(kind="disabled")
        algorithm = RexAlgorithmConfig(
            kind="rex",
            environment_names=("train_e01", "train_e02"),
            penalty_weight=penalty_weight,
            penalty_anneal_updates=anneal_updates,
            risk_variance="population",
            sampling="environment_balanced_without_replacement",
            loss_rescaling="divide_by_penalty_weight_above_one",
        )
        pair_digest = None
    elif candidate.method_id == "irm":
        penalty_weight = candidate.penalty_weight
        anneal_updates = config.search_space.irm_penalty_anneal_updates
        if penalty_weight is None or anneal_updates is None:
            raise AssertionError("planned IRM candidate lacks its settings")
        pairs = DisabledPairsConfig(kind="disabled")
        projection = DisabledProjectionConfig(kind="disabled")
        algorithm = IrmAlgorithmConfig(
            kind="irm",
            environment_names=("train_e01", "train_e02"),
            penalty_weight=penalty_weight,
            penalty_anneal_updates=anneal_updates,
            penalty="irmv1_dummy_classifier_scale",
            sampling="environment_balanced_without_replacement",
            loss_rescaling="divide_by_penalty_weight_above_one",
        )
        pair_digest = None
    else:
        raise AssertionError(
            f"CMNIST config materializer is missing {candidate.method_id}"
        )
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
    training_tables = cache.training_tables()
    algorithm = runtime.config.algorithm
    if isinstance(algorithm, (RexAlgorithmConfig, IrmAlgorithmConfig)):
        names = tuple(table.name for table in training_tables)
        if names != algorithm.environment_names:
            raise ValueError(
                "invariant training tables do not match configured environments"
            )
        environment_ids = torch.cat(
            [
                torch.full((len(table.source_ids),), index, dtype=torch.int64)
                for index, table in enumerate(training_tables)
            ]
        )
        method = (
            RexLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=len(training_tables),
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
            if isinstance(algorithm, RexAlgorithmConfig)
            else IrmLinearProbeMethod(
                environment_ids=environment_ids,
                environment_count=len(training_tables),
                penalty_weight=float(algorithm.penalty_weight),
                penalty_anneal_updates=int(algorithm.penalty_anneal_updates),
            )
        )
    elif isinstance(algorithm, GroupDroAlgorithmConfig):
        if algorithm.group_definition != "target_color":
            raise ValueError("CMNIST GroupDRO requires target-color groups")
        train_targets = torch.cat(
            [table.targets for table in training_tables], dim=0
        )
        train_colors = torch.cat(
            [table.colors for table in training_tables], dim=0
        )
        method = GroupDroLinearProbeMethod(
            group_ids=cmnist_group_ids(train_targets, train_colors),
            group_count=CMNIST_GROUP_COUNT,
            step_size=float(algorithm.adversarial_step_size),
        )
    else:
        method = OrdinaryLinearProbeMethod(
            method_id=task.candidate.method_id,
            projection=runtime.projection,
            projection_rank=task.candidate.requested_rank,
        )
    return train_linear_probe(
        training_tables,
        cache.validation_tables(),
        runtime.config.training,
        run_id=f"run:{task.task_id}",
        candidate_id=task.candidate.candidate_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        seed_stage=task.stage,
        seed=task.seed,
        method=method,
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
    for method in plan.methods:
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
    if "erm" in plan.methods and "grit" in plan.methods:
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
        methods=tuple(methods),
        paired_selectors=tuple(paired),
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
    by_id = {item.candidate_id: item for item in plan.candidates}

    def coerce_runs(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[CmnistCompletedStageRun, ...]:
        if any(not isinstance(run, CmnistCompletedStageRun) for run in runs):
            raise ValueError("CMNIST status received another dataset result")
        return cast(tuple[CmnistCompletedStageRun, ...], runs)

    def finalists_complete(
        state: tuple[
            dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
            dict[MethodId, FinalistUnion],
        ],
        root: Path,
    ) -> bool:
        expected_finalists, expected_unions = state
        count = 0
        for method in plan.methods:
            selection_root = root / "selection" / method
            artifacts = (
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
            for path, artifact_type, expected in artifacts:
                if not path.exists():
                    continue
                count += 1
                observed = artifact_type.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                if observed != expected:
                    raise ValueError(
                        "CMNIST selection artifact does not match canonical tuning "
                        f"results: {path}"
                    )
        return count == len(plan.methods) * (len(config.selectors) + 1)

    def confirmation_tasks(
        state: tuple[
            dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
            dict[MethodId, FinalistUnion],
        ],
    ) -> tuple[SearchRunTask, ...]:
        _, unions = state
        return tuple(
            make_search_task(
                plan,
                by_id[candidate_id],
                SeedStage.CONFIRMATION,
                seed,
            )
            for method in plan.methods
            for candidate_id in unions[method].confirmation_candidate_ids
            for seed in config.seeds.stages.confirmation
        )

    def winner_count(
        winners: dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection],
        root: Path,
    ) -> int:
        count = 0
        for identity, expected in winners.items():
            method, selector = identity
            path = root / "selection" / method / f"{selector.value}-winner.json"
            if not path.exists():
                continue
            count += 1
            observed = FrozenCandidateSelection.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError(
                    "CMNIST frozen winner does not match canonical confirmation "
                    f"results: {path}"
                )
        return count

    def final_tasks(
        winners: dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection],
    ) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                by_id[winner.candidate_id],
                seed,
                winner,
            )
            for winner in winners.values()
            for seed in config.seeds.stages.final
        )

    hooks = ProductionStatusHooks(
        coerce_runs=coerce_runs,
        compute_finalists=lambda runs: compute_cmnist_finalists(plan, runs),
        finalists_complete=finalists_complete,
        confirmation_tasks=confirmation_tasks,
        compute_winners=lambda state, runs: compute_cmnist_winners(
            plan, state[0], runs
        ),
        winner_count=winner_count,
        expected_winner_count=len(plan.methods) * len(config.selectors),
        final_tasks=final_tasks,
    )
    return production_lifecycle_status(plan, hooks)
