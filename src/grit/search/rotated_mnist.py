"""RotatedMNIST wiring for the shared production and linear-probe lifecycles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
    CpuRuntimeConfig,
    DisabledPairsConfig,
    DisabledProjectionConfig,
    ErmAlgorithmConfig,
    FrozenFeatureConfig,
    GritAlgorithmConfig,
    LinearProbeTrainingConfig,
    LinearProjectionConfig,
    OrdinarySelectionConfig,
    RotatedMnistArtifactLineageConfig,
    RotatedMnistDatasetConfig,
    RotatedMnistExperimentConfig,
    RotatedMnistOraclePairsConfig,
    RotatedMnistSourceCounts,
)
from grit.data.rotated_mnist import RotatedMnistOraclePairManifest
from grit.features.rotated_mnist import (
    RotatedMnistFeatureCache,
    RotatedMnistTuningFeatureCache,
    load_rotated_mnist_feature_cache,
    load_rotated_mnist_tuning_feature_cache,
)
from grit.lifecycle import open_final_test, record_final_accuracy
from grit.methods.checkpoints import restore_checkpoint
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import (
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    TrainedLinearProbeRun,
    evaluate_accuracy,
    persist_selected_linear_checkpoint,
    train_linear_probe,
)
from grit.methods.types import MethodId
from grit.results import ArtifactReference, OrdinaryRunResult, SucceededStatus
from grit.schemas import CmnistSelector, SeedStage
from grit.search.lifecycle import (
    ProductionLifecycleHooks,
    ProductionStatusHooks,
    production_lifecycle_status,
    run_production_lifecycle,
)
from grit.search.outputs import (
    RotatedMnistFinalSeedObservation,
    RotatedMnistMethodSelectorSummary,
    RotatedMnistPairedSeedDifference,
    RotatedMnistPairedSelectorSummary,
    RotatedMnistProductionSummary,
    make_rotated_mnist_accuracy_summary,
)
from grit.search.plan import (
    RotatedMnistProductionSearchConfig,
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
class _RuntimeCandidate:
    planned: SearchCandidate
    config: RotatedMnistExperimentConfig
    projection: FittedLinearProjection | None


SelectionState = tuple[
    dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    dict[MethodId, FinalistUnion],
]
WinnerState = dict[tuple[MethodId, CmnistSelector], FrozenCandidateSelection]


def run_rotated_mnist_search(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None = None,
) -> RotatedMnistProductionSummary | ProductionSearchStatus:
    config = _search_config(plan)
    output_root = Path(plan.resolved_config.output_root)
    cache = _load_cache(
        plan, tuning_only=limits is not None and limits.stop_after == "tuning"
    )
    pair_manifest = _pair_manifest(plan)
    projections: dict[int, FittedLinearProjection] = {}

    def runtime_candidate(candidate: SearchCandidate) -> _RuntimeCandidate:
        projection: FittedLinearProjection | None = None
        if candidate.method_id == "grit":
            rank = candidate.requested_rank
            if rank is None:
                raise AssertionError("planned GRIT candidate lacks a rank")
            projection = projections.get(rank)
            if projection is None:
                left, right = cache.pair_tables()
                projection = fit_linear_projection(
                    left.features[: config.pair_count],
                    right.features[: config.pair_count],
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
        resolved = materialize_rotated_mnist_candidate_config(
            plan, candidate, CmnistSelector.PRIMARY_ROBUST
        )
        return _RuntimeCandidate(candidate, resolved, projection)

    def execute_pre_final(task: SearchRunTask, _run_root: Path) -> CompletedStageRun:
        trained = _train_task(cache, runtime_candidate(task.candidate), task)
        decisions = tuple(
            select_checkpoint(trained.validation_metrics, selector)
            for selector in (
                CmnistSelector.PRIMARY_ROBUST,
                CmnistSelector.SECONDARY_SOURCE,
            )
        )
        return CmnistCompletedStageRun(
            schema_version="grit.rotated-mnist-search-stage-run/v1",
            dataset="rotated_mnist",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            code=current_code_provenance(),
            environment=current_environment_provenance(),
            validation_metrics=trained.validation_metrics,
            checkpoint_decisions=decisions,
        )

    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in plan.candidates
    }

    def confirmation_tasks(state: SelectionState) -> tuple[SearchRunTask, ...]:
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
        state: SelectionState,
        runs: tuple[CmnistCompletedStageRun, ...],
        root: Path,
    ) -> WinnerState:
        finalists, _ = state
        winners = compute_rotated_mnist_winners(plan, finalists, runs)
        for (method, selector), winner in winners.items():
            persist_canonical_artifact(
                root / "selection" / method / f"{selector.value}-winner.json",
                winner,
            )
        return winners

    def final_tasks(winners: WinnerState) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(
                plan,
                candidates_by_id[winner.candidate_id],
                seed,
                winner,
            )
            for winner in winners.values()
            for seed in config.seeds.stages.final
        )

    def final_executor(winners: WinnerState) -> StageExecutor:
        if not isinstance(cache, RotatedMnistFeatureCache):
            raise AssertionError("tuning-only execution reached final stage")
        full_cache = cache

        def execute_final(task: SearchRunTask, run_root: Path) -> CompletedStageRun:
            selector = CmnistSelector(task.selector)
            winner = winners[(task.candidate.method_id, selector)]
            initial = runtime_candidate(task.candidate)
            runtime = _RuntimeCandidate(
                initial.planned,
                materialize_rotated_mnist_candidate_config(
                    plan, task.candidate, selector
                ),
                initial.projection,
            )
            trained = _train_task(full_cache, runtime, task)
            decision = select_checkpoint(trained.validation_metrics, selector)
            frozen_checkpoint = freeze_final_checkpoint(decision, winner)
            selected = trained.store.load(frozen_checkpoint.checkpoint.checkpoint_id)
            checkpoint_root = run_root / "selected-checkpoint"
            checkpoint_manifest = persist_selected_linear_checkpoint(
                selected, checkpoint_root
            )
            restoration = restore_checkpoint(
                frozen_checkpoint,
                PersistedLinearCheckpointStore(checkpoint_root),
                trained.algorithm.restore_inference_state,
            )
            handle = full_cache.issue_final_handle(
                run_id=trained.run_id,
                candidate_id=task.candidate.candidate_id,
                scientific_config_digest=task.candidate.scientific_config_digest,
            )
            final_view = open_final_test(handle, winner, frozen_checkpoint, restoration)
            final_table = full_cache.open_final_table(final_view)
            final_metric = record_final_accuracy(
                final_view,
                record_id=f"metric:{trained.run_id}:test_r90",
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
                candidate_selection=winner,
                checkpoint_selection=frozen_checkpoint,
                restoration=restoration,
                final_test_metrics=(final_metric,),
                artifacts=_result_artifacts(
                    plan,
                    task,
                    checkpoint_manifest.canonical_digest(),
                    runtime,
                ),
            )
            persist_canonical_artifact(run_root / "final-result.json", result)
            completed = CmnistCompletedStageRun(
                schema_version="grit.rotated-mnist-search-stage-run/v1",
                dataset="rotated_mnist",
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

    def make_finalists(
        runs: tuple[CmnistCompletedStageRun, ...], root: Path
    ) -> SelectionState:
        state = compute_rotated_mnist_finalists(plan, runs)
        finalists, unions = state
        for method in plan.methods:
            method_root = root / "selection" / method
            persist_canonical_artifact(
                method_root / "primary-tuning-finalists.json",
                finalists[(method, CmnistSelector.PRIMARY_ROBUST)],
            )
            persist_canonical_artifact(
                method_root / "secondary-tuning-finalists.json",
                finalists[(method, CmnistSelector.SECONDARY_SOURCE)],
            )
            persist_canonical_artifact(
                method_root / "confirmation-union.json", unions[method]
            )
        return state

    def make_summary(
        state: SelectionState,
        winners: WinnerState,
        runs: tuple[CmnistCompletedStageRun, ...],
    ) -> RotatedMnistProductionSummary:
        return _summary(plan, state[0], winners, runs)

    def persist_summary(summary: RotatedMnistProductionSummary, root: Path) -> None:
        persist_canonical_artifact(
            root / "summaries" / "rotated-mnist-summary.json", summary
        )
        for paired in summary.paired_selectors:
            persist_canonical_artifact(
                root
                / "summaries"
                / f"rotated-mnist-{paired.selector.value}-paired-differences.json",
                paired,
            )

    hooks = ProductionLifecycleHooks(
        coerce_runs=_coerce_runs,
        execute_pre_final=execute_pre_final,
        make_finalists=make_finalists,
        confirmation_tasks=confirmation_tasks,
        make_winners=make_winners,
        final_tasks=final_tasks,
        final_executor=final_executor,
        make_summary=make_summary,
        persist_summary=persist_summary,
        status=lambda: rotated_mnist_status_from_plan(plan),
    )
    return run_production_lifecycle(plan, limits, hooks)


def compute_rotated_mnist_finalists(
    plan: SearchPlan, runs: tuple[CmnistCompletedStageRun, ...]
) -> SelectionState:
    metrics = tuple(metric for run in runs for metric in run.validation_metrics)
    finalists: dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact] = {}
    unions: dict[MethodId, FinalistUnion] = {}
    for method in plan.methods:
        method_metrics = tuple(item for item in metrics if item.method_id == method)
        primary = _rotated_finalists(
            make_tuning_finalists(
                method_metrics, CmnistSelector.PRIMARY_ROBUST, plan.seeds.stages
            )
        )
        secondary = _rotated_finalists(
            make_tuning_finalists(
                method_metrics, CmnistSelector.SECONDARY_SOURCE, plan.seeds.stages
            )
        )
        union = _rotated_union(make_finalist_union(primary, secondary))
        finalists[(method, CmnistSelector.PRIMARY_ROBUST)] = primary
        finalists[(method, CmnistSelector.SECONDARY_SOURCE)] = secondary
        unions[method] = union
    return finalists, unions


def compute_rotated_mnist_winners(
    plan: SearchPlan,
    finalists: dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    runs: tuple[CmnistCompletedStageRun, ...],
) -> WinnerState:
    metrics = tuple(metric for run in runs for metric in run.validation_metrics)
    winners: WinnerState = {}
    for method in plan.methods:
        for selector in (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            artifact = finalists[(method, selector)]
            finalist_ids = {
                candidate.candidate_id for candidate in artifact.ordered_candidates
            }
            records = tuple(
                metric
                for metric in metrics
                if metric.method_id == method and metric.candidate_id in finalist_ids
            )
            decision = select_confirmed_candidate(records, artifact, plan.seeds.stages)
            winners[(method, selector)] = _rotated_winner(
                freeze_candidate(decision, artifact, plan.seeds.stages)
            )
    return winners


def materialize_rotated_mnist_candidate_config(
    plan: SearchPlan,
    candidate: SearchCandidate,
    selector: CmnistSelector,
) -> RotatedMnistExperimentConfig:
    config = _search_config(plan)
    lineage = plan.resolved_config.lineage
    pairs: DisabledPairsConfig | RotatedMnistOraclePairsConfig
    projection: DisabledProjectionConfig | LinearProjectionConfig
    algorithm: ErmAlgorithmConfig | GritAlgorithmConfig
    pair_digest: str | None
    if candidate.method_id == "erm":
        pairs = DisabledPairsConfig(kind="disabled")
        projection = DisabledProjectionConfig(kind="disabled")
        algorithm = ErmAlgorithmConfig(kind="erm")
        pair_digest = None
    elif candidate.method_id == "grit":
        rank = candidate.requested_rank
        if rank is None:
            raise AssertionError("planned GRIT candidate lacks a rank")
        pairs = RotatedMnistOraclePairsConfig(
            kind="oracle",
            construction_id="rotated-mnist-exact-source-oracle-pairs-v1",
            source_partition_ids=("train_r0_sources", "train_r45_sources"),
            pair_count=config.pair_count,
            pair_seed=config.seeds.pairs,
            orientation="rotation_0_minus_45",
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
    else:
        raise ValueError("RotatedMNIST supports only ERM and GRIT")
    resolved = RotatedMnistExperimentConfig(
        schema_version="grit.experiment/v1",
        run_kind="rotated_mnist_ordinary",
        experiment_name=config.experiment_name,
        protocol_id="rotated_mnist/v1",
        reportable=True,
        dataset=RotatedMnistDatasetConfig(
            dataset_id="rotated_mnist",
            construction_method_id="rotated-mnist-stratified-hash-v1",
            construction_seed=config.seeds.construction,
            source_counts=RotatedMnistSourceCounts(
                train_r0=25_000,
                train_r45=25_000,
                validation=10_000,
                test=10_000,
            ),
            training_split_names=("train_r0", "train_r45"),
            validation_split_names=("val_r0", "val_r45", "val_r60"),
            final_test_split_name="test_r90",
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
        training=LinearProbeTrainingConfig(
            optimizer="adam",
            batch_size=config.batch_size,
            learning_rate=float(candidate.learning_rate),
            weight_decay=float(candidate.weight_decay),
            max_epochs=config.max_epochs,
        ),
        runtime=CpuRuntimeConfig(device="cpu", deterministic_algorithms=True),
        seed_sets=config.seeds.stages,
        artifact_lineage=RotatedMnistArtifactLineageConfig(
            dataset_manifest_digest=lineage.dataset_manifest_digest,
            feature_cache_manifest_digest=lineage.feature_cache_manifest_digest,
            pair_manifest_digest=pair_digest,
        ),
        selection=OrdinarySelectionConfig(selector=selector),
    )
    if resolved.scientific_config_digest() != candidate.scientific_config_digest:
        raise ValueError("planned RotatedMNIST candidate digest cannot be materialized")
    return resolved


def _train_task(
    cache: RotatedMnistFeatureCache | RotatedMnistTuningFeatureCache,
    runtime: _RuntimeCandidate,
    task: SearchRunTask,
) -> TrainedLinearProbeRun:
    method = OrdinaryLinearProbeMethod(
        method_id=task.candidate.method_id,
        projection=runtime.projection,
        projection_rank=task.candidate.requested_rank,
    )
    return train_linear_probe(
        cache.training_tables(),
        cache.validation_tables(),
        runtime.config.training,
        run_id=f"run:{task.task_id}",
        candidate_id=task.candidate.candidate_id,
        scientific_config_digest=task.candidate.scientific_config_digest,
        seed_stage=task.stage,
        seed=task.seed,
        method=method,
        num_classes=10,
    )


def _result_artifacts(
    plan: SearchPlan,
    task: SearchRunTask,
    checkpoint_manifest_digest: str,
    runtime: _RuntimeCandidate,
) -> tuple[ArtifactReference, ...]:
    root = Path(plan.resolved_config.output_root)
    inputs = {item.kind: item for item in plan.resolved_config.input_artifacts}
    artifacts = [
        ArtifactReference(
            artifact_id="rotated-mnist-dataset-manifest",
            kind="dataset_manifest",
            relative_uri=os.path.relpath(inputs["dataset_manifest"].path, root),
            digest=inputs["dataset_manifest"].digest,
        ),
        ArtifactReference(
            artifact_id="rotated-mnist-feature-manifest",
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
        artifacts.extend(
            (
                ArtifactReference(
                    artifact_id="rotated-mnist-oracle-pair-manifest",
                    kind="pair_manifest",
                    relative_uri=os.path.relpath(inputs["pair_manifest"].path, root),
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
    return tuple(artifacts)


def _summary(
    plan: SearchPlan,
    finalists: dict[tuple[MethodId, CmnistSelector], TuningFinalistsArtifact],
    winners: WinnerState,
    runs: tuple[CmnistCompletedStageRun, ...],
) -> RotatedMnistProductionSummary:
    by_identity = {
        (
            run.task.candidate.method_id,
            CmnistSelector(run.task.selector),
            run.task.seed,
        ): run
        for run in runs
    }
    methods: list[RotatedMnistMethodSelectorSummary] = []
    for method in plan.methods:
        for selector in (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            observations: list[RotatedMnistFinalSeedObservation] = []
            for seed in plan.seeds.stages.final:
                run = by_identity[(method, selector, seed)]
                result = run.final_result
                if (
                    not isinstance(result, OrdinaryRunResult)
                    or result.final_test_metrics is None
                ):
                    raise ValueError("final run is missing its test result")
                metric = result.final_test_metrics[0]
                observations.append(
                    RotatedMnistFinalSeedObservation(
                        seed=seed,
                        method_id=method,
                        selector=selector,
                        result_path=(
                            f"{run.task.relative_directory}/final-result.json"
                        ),
                        metric_record_id=metric.record_id,
                        test_r90_accuracy=float(metric.value),
                    )
                )
            values = tuple(float(item.test_r90_accuracy) for item in observations)
            artifact = finalists[(method, selector)]
            methods.append(
                RotatedMnistMethodSelectorSummary(
                    method_id=method,
                    selector=selector,
                    lineage=plan.resolved_config.lineage,
                    selected_candidate_id=winners[(method, selector)].candidate_id,
                    finalist_candidate_ids=cast(
                        tuple[str, str, str],
                        tuple(
                            candidate.candidate_id
                            for candidate in artifact.ordered_candidates
                        ),
                    ),
                    configured_final_seeds=plan.seeds.stages.final,
                    final_observations=tuple(observations),
                    accuracy_summary=make_rotated_mnist_accuracy_summary(
                        "test_r90_accuracy", values
                    ),
                )
            )
    method_map = {(item.method_id, item.selector): item for item in methods}
    paired: list[RotatedMnistPairedSelectorSummary] = []
    for selector in (
        CmnistSelector.PRIMARY_ROBUST,
        CmnistSelector.SECONDARY_SOURCE,
    ):
        erm = method_map[("erm", selector)]
        grit = method_map[("grit", selector)]
        erm_values = {
            item.seed: float(item.test_r90_accuracy) for item in erm.final_observations
        }
        grit_values = {
            item.seed: float(item.test_r90_accuracy) for item in grit.final_observations
        }
        differences = tuple(
            RotatedMnistPairedSeedDifference(
                seed=seed,
                selector=selector,
                grit_minus_erm_test_r90_accuracy=(grit_values[seed] - erm_values[seed]),
            )
            for seed in plan.seeds.stages.final
        )
        paired.append(
            RotatedMnistPairedSelectorSummary(
                selector=selector,
                configured_final_seeds=plan.seeds.stages.final,
                paired_differences=differences,
                difference_summary=make_rotated_mnist_accuracy_summary(
                    "grit_minus_erm_test_r90_accuracy",
                    tuple(
                        float(item.grit_minus_erm_test_r90_accuracy)
                        for item in differences
                    ),
                ),
            )
        )
    return RotatedMnistProductionSummary(
        schema_version="grit.rotated-mnist-production-summary/v1",
        reportable=True,
        plan_digest=plan.canonical_digest(),
        lineage=plan.resolved_config.lineage,
        methods=cast(
            tuple[
                RotatedMnistMethodSelectorSummary,
                RotatedMnistMethodSelectorSummary,
                RotatedMnistMethodSelectorSummary,
                RotatedMnistMethodSelectorSummary,
            ],
            tuple(methods),
        ),
        paired_selectors=cast(
            tuple[
                RotatedMnistPairedSelectorSummary,
                RotatedMnistPairedSelectorSummary,
            ],
            tuple(paired),
        ),
    )


def _rotated_finalists(
    artifact: TuningFinalistsArtifact,
) -> TuningFinalistsArtifact:
    payload = artifact.model_dump(mode="python")
    payload["schema_version"] = "grit.rotated-mnist-tuning-finalists/v1"
    return TuningFinalistsArtifact.model_validate(payload)


def _rotated_union(artifact: FinalistUnion) -> FinalistUnion:
    payload = artifact.model_dump(mode="python")
    payload["schema_version"] = "grit.rotated-mnist-finalist-union/v1"
    return FinalistUnion.model_validate(payload)


def _rotated_winner(
    artifact: FrozenCandidateSelection,
) -> FrozenCandidateSelection:
    payload = artifact.model_dump(mode="python")
    payload["schema_version"] = "grit.rotated-mnist-frozen-candidate/v1"
    return FrozenCandidateSelection.model_validate(payload)


def _coerce_runs(
    runs: tuple[CompletedStageRun, ...],
) -> tuple[CmnistCompletedStageRun, ...]:
    if any(
        not isinstance(run, CmnistCompletedStageRun) or run.dataset != "rotated_mnist"
        for run in runs
    ):
        raise ValueError("RotatedMNIST lifecycle received another dataset result")
    return cast(tuple[CmnistCompletedStageRun, ...], runs)


def _load_cache(
    plan: SearchPlan, *, tuning_only: bool = False
) -> RotatedMnistFeatureCache | RotatedMnistTuningFeatureCache:
    paths = {
        item.kind: Path(item.path) for item in plan.resolved_config.input_artifacts
    }
    loader = (
        load_rotated_mnist_tuning_feature_cache
        if tuning_only
        else load_rotated_mnist_feature_cache
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
        raise ValueError("RotatedMNIST feature manifest changed after planning")
    return cache


def _pair_manifest(plan: SearchPlan) -> RotatedMnistOraclePairManifest:
    path = next(
        Path(item.path)
        for item in plan.resolved_config.input_artifacts
        if item.kind == "pair_manifest"
    )
    manifest = RotatedMnistOraclePairManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if manifest.canonical_digest() != plan.resolved_config.lineage.pair_manifest_digest:
        raise ValueError("RotatedMNIST pair manifest changed after planning")
    return manifest


def _search_config(plan: SearchPlan) -> RotatedMnistProductionSearchConfig:
    config = plan.resolved_config.config
    if not isinstance(config, RotatedMnistProductionSearchConfig):
        raise TypeError("RotatedMNIST runner received another dataset")
    return config


def rotated_mnist_status_from_plan(plan: SearchPlan) -> ProductionSearchStatus:
    config = _search_config(plan)
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}

    def finalists_complete(state: SelectionState, root: Path) -> bool:
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
                    raise ValueError("RotatedMNIST selection artifact is inconsistent")
        return count == len(plan.methods) * (len(config.selectors) + 1)

    def confirmation_tasks(state: SelectionState) -> tuple[SearchRunTask, ...]:
        _, unions = state
        return tuple(
            make_search_task(plan, by_id[candidate_id], SeedStage.CONFIRMATION, seed)
            for method in plan.methods
            for candidate_id in unions[method].confirmation_candidate_ids
            for seed in config.seeds.stages.confirmation
        )

    def winner_count(winners: WinnerState, root: Path) -> int:
        count = 0
        for (method, selector), expected in winners.items():
            path = root / "selection" / method / f"{selector.value}-winner.json"
            if not path.exists():
                continue
            count += 1
            observed = FrozenCandidateSelection.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected:
                raise ValueError("RotatedMNIST frozen winner is inconsistent")
        return count

    def final_tasks(winners: WinnerState) -> tuple[SearchRunTask, ...]:
        return tuple(
            make_final_search_task(plan, by_id[winner.candidate_id], seed, winner)
            for winner in winners.values()
            for seed in config.seeds.stages.final
        )

    hooks = ProductionStatusHooks(
        coerce_runs=_coerce_runs,
        compute_finalists=lambda runs: compute_rotated_mnist_finalists(plan, runs),
        finalists_complete=finalists_complete,
        confirmation_tasks=confirmation_tasks,
        compute_winners=lambda state, runs: compute_rotated_mnist_winners(
            plan, state[0], runs
        ),
        winner_count=winner_count,
        expected_winner_count=len(plan.methods) * len(config.selectors),
        final_tasks=final_tasks,
    )
    return production_lifecycle_status(plan, hooks)
