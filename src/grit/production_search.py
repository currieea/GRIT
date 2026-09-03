"""Production-capable local CMNIST and Waterbirds search orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from pydantic import Field, StrictInt, StrictStr, model_validator

from grit.checkpoints import restore_checkpoint
from grit.cmnist import CmnistOraclePairManifest
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
from grit.features import (
    CmnistFeatureCache,
    CmnistTuningFeatureCache,
    load_cmnist_feature_cache,
    load_cmnist_tuning_feature_cache,
)
from grit.lifecycle import open_final_test, record_final_accuracy
from grit.projection import FittedLinearProjection, fit_linear_projection
from grit.results import ArtifactReference, OrdinaryRunResult, SucceededStatus
from grit.schemas import CmnistSelector, SeedStage, StrictBoundaryModel
from grit.search import (
    CmnistProductionSearchConfig,
    ResolvedProductionSearchConfig,
    SearchCandidate,
    SearchPlan,
    WaterbirdsProductionSearchConfig,
    current_code_provenance,
    current_environment_provenance,
    load_production_search_config,
    write_search_plan,
)
from grit.search_outputs import (
    CmnistFinalSeedObservation,
    CmnistMethodSelectorSummary,
    CmnistPairedSeedDifference,
    CmnistPairedSelectorSummary,
    CmnistProductionSummary,
    ExperimentIndex,
    IndexedArtifact,
    WaterbirdsPairedSummaryArtifact,
    WaterbirdsProductionSummary,
    make_cmnist_accuracy_summary,
    verify_experiment_index,
)
from grit.search_scheduler import (
    CmnistCompletedStageRun,
    CompletedStageRun,
    LocalRunScheduler,
    SearchRunTask,
    StageExecutor,
    WaterbirdsCompletedStageRun,
    make_final_search_task,
    make_search_task,
)
from grit.selection import (
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
from grit.training import (
    MethodId,
    PersistedLinearCheckpointStore,
    TrainedLinearProbeRun,
    evaluate_accuracy,
    persist_selected_linear_checkpoint,
    train_linear_probe,
)
from grit.waterbirds_selection import (
    FrozenWaterbirdsCandidate,
    WaterbirdsTuningFinalists,
)

NonEmptyStr: TypeAlias = str


class ProductionSearchStatus(StrictBoundaryModel):
    schema_version: Literal["grit.production-search-status/v1"]
    dataset: Literal["cmnist", "waterbirds_cf"]
    plan_digest: StrictStr = Field(min_length=1)
    phase: Literal[
        "tuning",
        "confirmation",
        "final",
        "complete",
    ]
    tuning_expected: StrictInt = Field(ge=0)
    tuning_complete: StrictInt = Field(ge=0)
    confirmation_expected: StrictInt = Field(ge=0)
    confirmation_complete: StrictInt = Field(ge=0)
    frozen_winner_count: StrictInt = Field(ge=0)
    final_expected: StrictInt = Field(ge=0)
    final_complete: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def _validate_counts(self) -> ProductionSearchStatus:
        if (
            self.tuning_complete > self.tuning_expected
            or self.confirmation_complete > self.confirmation_expected
            or self.final_complete > self.final_expected
        ):
            raise ValueError("production search status exceeds its planned work")
        return self


@dataclass(frozen=True, slots=True)
class ProductionExecutionLimits:
    """Operational task limits that never participate in scientific identity."""

    stop_after: Literal["tuning"] | None = None
    method: Literal["erm", "grit", "all"] = "all"
    candidate_ids: tuple[str, ...] = ()
    tuning_seed: int | None = None
    max_new_runs: int | None = None


class PilotCandidateSelection(StrictBoundaryModel):
    """Read-only deterministic presentation of the two recommended pilot tasks."""

    schema_version: Literal["grit.production-pilot-candidates/v1"]
    dataset: Literal["cmnist", "waterbirds_cf"]
    plan_digest: StrictStr = Field(min_length=1)
    tuning_seed: StrictInt
    erm: SearchCandidate
    grit_nonzero_rank: SearchCandidate


@dataclass(frozen=True, slots=True)
class _CmnistRuntimeCandidate:
    planned: SearchCandidate
    config: OrdinaryExperimentConfig
    projection: FittedLinearProjection | None


def plan_production_search(config_path: Path) -> SearchPlan:
    """Validate all production inputs and atomically write or reuse the plan."""

    config = load_production_search_config(config_path)
    return write_search_plan(config, config_path=config_path)


def run_production_search(
    config_path: Path,
    limits: ProductionExecutionLimits | None = None,
) -> CmnistProductionSummary | WaterbirdsProductionSummary | ProductionSearchStatus:
    """Run or continue the dataset-specific search selected by strict YAML."""

    plan = plan_production_search(config_path)
    checked_limits = validate_execution_limits(plan, limits)
    if isinstance(plan.resolved_config.config, CmnistProductionSearchConfig):
        return _run_cmnist_search(plan, checked_limits)
    from grit.production_waterbirds_search import run_waterbirds_production_search

    return run_waterbirds_production_search(plan, checked_limits)


def production_pilot_candidates(config_path: Path) -> PilotCandidateSelection:
    """Present canonical ERM/nonzero-GRIT pilot candidates from an existing plan."""

    plan = _load_existing_search_plan(config_path)
    erm = next(
        candidate for candidate in plan.candidates if candidate.method_id == "erm"
    )
    grit = next(
        candidate
        for candidate in plan.candidates
        if candidate.method_id == "grit"
        and candidate.requested_rank is not None
        and candidate.requested_rank > 0
    )
    return PilotCandidateSelection(
        schema_version="grit.production-pilot-candidates/v1",
        dataset=plan.dataset,
        plan_digest=plan.canonical_digest(),
        tuning_seed=plan.seeds.stages.tuning[0],
        erm=erm,
        grit_nonzero_rank=grit,
    )


def production_search_status(config_path: Path) -> ProductionSearchStatus:
    """Report canonical completed work without invoking a trainer."""

    plan = _load_existing_search_plan(config_path)
    if isinstance(plan.resolved_config.config, CmnistProductionSearchConfig):
        return _cmnist_status(plan)
    return waterbirds_status_from_plan(plan)


def _load_existing_search_plan(config_path: Path) -> SearchPlan:
    """Load one compatible planning triplet without resolving inputs or writing."""

    supplied_path = config_path.resolve()
    supplied = load_production_search_config(supplied_path)
    output_value = Path(supplied.output_root)
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (supplied_path.parent / output_value).resolve()
    )
    authored_path = output_root / "authored-config.yaml"
    resolved_path = output_root / "resolved-config.json"
    plan_path = output_root / "search-plan.json"
    planning_paths = (authored_path, resolved_path, plan_path)
    if not all(path.is_file() for path in planning_paths):
        raise ValueError(
            "no complete production search plan exists; run scripts/run_search.py first"
        )
    stored_authored = load_production_search_config(authored_path)
    stored_resolved = ResolvedProductionSearchConfig.model_validate_json(
        resolved_path.read_text(encoding="utf-8")
    )
    stored_plan = SearchPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    if stored_authored != supplied:
        raise ValueError("saved authored config does not match the supplied YAML")
    if (
        stored_resolved.config != supplied
        or Path(stored_resolved.output_root).resolve() != output_root
        or stored_plan.resolved_config != stored_resolved
    ):
        raise ValueError(
            "saved resolved config or search plan does not match the supplied YAML"
        )
    return stored_plan


def validate_execution_limits(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None,
) -> ProductionExecutionLimits | None:
    if limits is None or limits == ProductionExecutionLimits():
        return None
    if limits.stop_after not in {None, "tuning"}:
        raise ValueError("stop_after supports only the tuning boundary")
    if limits.method not in {"erm", "grit", "all"}:
        raise ValueError("method must be erm, grit, or all")
    max_new_runs = _validate_optional_exact_integer(
        limits.max_new_runs, "max_new_runs"
    )
    tuning_seed = _validate_optional_exact_integer(
        limits.tuning_seed, "tuning_seed"
    )
    candidate_ids = _validate_candidate_ids(limits.candidate_ids)
    if max_new_runs is not None and max_new_runs <= 0:
        raise ValueError("max_new_runs must be positive")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id filters must be unique")
    uses_tuning_filter = (
        limits.method != "all"
        or bool(candidate_ids)
        or tuning_seed is not None
    )
    if uses_tuning_filter and limits.stop_after != "tuning":
        raise ValueError(
            "method, candidate, and tuning-seed filters require stop_after=tuning"
        )
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    unknown = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in by_id
    )
    if unknown:
        raise ValueError(f"candidate IDs are not present in the plan: {unknown}")
    if limits.method != "all":
        wrong_method = tuple(
            candidate_id
            for candidate_id in candidate_ids
            if by_id[candidate_id].method_id != limits.method
        )
        if wrong_method:
            raise ValueError(
                "candidate IDs do not agree with the method filter: "
                f"{wrong_method}"
            )
    if tuning_seed is not None and tuning_seed not in plan.seeds.stages.tuning:
        raise ValueError("tuning_seed is not one of the configured tuning seeds")
    return limits


def _validate_optional_exact_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _validate_candidate_ids(values: tuple[object, ...]) -> tuple[str, ...]:
    validated: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("candidate IDs must be nonempty strings")
        validated.append(value)
    return tuple(validated)


def limited_tuning_candidates(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None,
) -> tuple[SearchCandidate, ...]:
    if limits is None or limits.stop_after is None:
        return plan.candidates
    selected_ids = set(limits.candidate_ids)
    candidates = tuple(
        candidate
        for candidate in plan.candidates
        if (limits.method == "all" or candidate.method_id == limits.method)
        and (not selected_ids or candidate.candidate_id in selected_ids)
    )
    if not candidates:
        raise ValueError("tuning filters select no canonical plan candidates")
    return candidates


def _run_cmnist_search(
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
        return _cmnist_status(plan)
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
        return _cmnist_status(plan)
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
        return _cmnist_status(plan)
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


def _cmnist_status(plan: SearchPlan) -> ProductionSearchStatus:
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
            _planned_candidate(by_id, candidate_id),
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
            _planned_candidate(by_id, winner.candidate_id),
            seed,
            winner,
        )
        for winner in expected_winners.values()
        for seed in config.seeds.stages.final
    )
    final_status = scheduler.status(final)
    final_complete = len(final_status.complete_task_ids)
    phase: Literal["final", "complete"] = "final"
    if final_complete == len(final) and _complete_outputs_valid(plan):
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


def waterbirds_status_from_plan(plan: SearchPlan) -> ProductionSearchStatus:
    config = plan.resolved_config.config
    if not isinstance(config, WaterbirdsProductionSearchConfig):
        raise TypeError("Waterbirds status received another dataset")
    scheduler = LocalRunScheduler(Path(plan.resolved_config.output_root), plan)
    tuning = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in plan.candidates
        for seed in config.seeds.stages.tuning
    )
    status = scheduler.status(tuning)
    tuning_complete = len(status.complete_task_ids)
    if tuning_complete != len(tuning):
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="waterbirds_cf",
            plan_digest=plan.canonical_digest(),
            phase="tuning",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
            confirmation_expected=0,
            confirmation_complete=0,
            frozen_winner_count=0,
            final_expected=0,
            final_complete=0,
        )
    tuning_runs_untyped = scheduler.completed_results(tuning)
    if any(
        not isinstance(run, WaterbirdsCompletedStageRun)
        for run in tuning_runs_untyped
    ):
        raise ValueError("Waterbirds tuning stage contains another dataset result")
    tuning_runs = cast(
        tuple[WaterbirdsCompletedStageRun, ...], tuning_runs_untyped
    )
    from grit.production_waterbirds_search import (
        compute_waterbirds_finalists,
        compute_waterbirds_winners,
    )

    expected_finalists = compute_waterbirds_finalists(plan, tuning_runs)
    root = Path(plan.resolved_config.output_root)
    finalist_count = 0
    for method in ("erm", "grit"):
        path = root / "selection" / method / "tuning-finalists.json"
        if not path.exists():
            continue
        finalist_count += 1
        observed = WaterbirdsTuningFinalists.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if observed != expected_finalists[method]:
            raise ValueError(
                "Waterbirds finalist artifact does not match canonical tuning "
                f"results: {path}"
            )
    if finalist_count != 2:
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="waterbirds_cf",
            plan_digest=plan.canonical_digest(),
            phase="confirmation",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
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
            _planned_candidate(by_id, item.candidate_id),
            SeedStage.CONFIRMATION,
            seed,
        )
        for method in ("erm", "grit")
        for item in expected_finalists[method].ordered_candidates
        for seed in config.seeds.stages.confirmation
    )
    confirmation_status = scheduler.status(confirmation)
    confirmation_complete = len(confirmation_status.complete_task_ids)
    if confirmation_complete != len(confirmation):
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="waterbirds_cf",
            plan_digest=plan.canonical_digest(),
            phase="confirmation",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
            confirmation_expected=len(confirmation),
            confirmation_complete=confirmation_complete,
            frozen_winner_count=0,
            final_expected=0,
            final_complete=0,
        )
    confirmation_runs_untyped = scheduler.completed_results(confirmation)
    if any(
        not isinstance(run, WaterbirdsCompletedStageRun)
        for run in confirmation_runs_untyped
    ):
        raise ValueError("Waterbirds confirmation contains another dataset result")
    confirmation_runs = cast(
        tuple[WaterbirdsCompletedStageRun, ...], confirmation_runs_untyped
    )
    expected_winners = compute_waterbirds_winners(
        plan, expected_finalists, confirmation_runs
    )
    winner_count = 0
    for method in ("erm", "grit"):
        path = root / "selection" / method / "winner.json"
        if path.exists():
            winner_count += 1
            observed = FrozenWaterbirdsCandidate.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if observed != expected_winners[method]:
                raise ValueError(
                    "Waterbirds frozen winner does not match canonical confirmation "
                    f"results: {path}"
                )
    if winner_count != 2:
        return ProductionSearchStatus(
            schema_version="grit.production-search-status/v1",
            dataset="waterbirds_cf",
            plan_digest=plan.canonical_digest(),
            phase="final",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
            confirmation_expected=len(confirmation),
            confirmation_complete=confirmation_complete,
            frozen_winner_count=winner_count,
            final_expected=0,
            final_complete=0,
        )
    final = tuple(
        make_final_search_task(
            plan,
            _planned_candidate(by_id, winner.candidate_id),
            seed,
            winner,
        )
        for winner in expected_winners.values()
        for seed in config.seeds.stages.final
    )
    final_status = scheduler.status(final)
    final_complete = len(final_status.complete_task_ids)
    phase = "final"
    if final_complete == len(final) and _complete_outputs_valid(plan):
        phase = "complete"
    return ProductionSearchStatus(
        schema_version="grit.production-search-status/v1",
        dataset="waterbirds_cf",
        plan_digest=plan.canonical_digest(),
        phase=phase,
        tuning_expected=len(tuning),
        tuning_complete=tuning_complete,
        confirmation_expected=len(confirmation),
        confirmation_complete=confirmation_complete,
        frozen_winner_count=2,
        final_expected=len(final),
        final_complete=final_complete,
    )


def _planned_candidate(
    candidates: dict[str, SearchCandidate], candidate_id: str
) -> SearchCandidate:
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise ValueError(
            f"selection artifact references an out-of-plan candidate: {candidate_id}"
        )
    return candidate


def _complete_outputs_valid(plan: SearchPlan) -> bool:
    root = Path(plan.resolved_config.output_root)
    index_path = root / "experiment-index.json"
    if plan.dataset == "cmnist":
        summary_path = root / "summaries" / "cmnist-summary.json"
        paired_paths = tuple(
            root
            / "summaries"
            / f"cmnist-{selector.value}-paired-differences.json"
            for selector in (
                CmnistSelector.PRIMARY_ROBUST,
                CmnistSelector.SECONDARY_SOURCE,
            )
        )
        required = (summary_path, *paired_paths, index_path)
        if not all(path.is_file() for path in required):
            return False
        summary = CmnistProductionSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        paired = tuple(
            CmnistPairedSelectorSummary.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for path in paired_paths
        )
        if (
            summary.plan_digest != plan.canonical_digest()
            or summary.lineage != plan.resolved_config.lineage
            or summary.paired_selectors != paired
        ):
            raise ValueError("CMNIST completion summary is inconsistent")
    else:
        summary_path = root / "summaries" / "waterbirds-summary.json"
        paired_path = root / "summaries" / "waterbirds-paired-differences.json"
        if not all(path.is_file() for path in (summary_path, paired_path, index_path)):
            return False
        summary = WaterbirdsProductionSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        paired = WaterbirdsPairedSummaryArtifact.model_validate_json(
            paired_path.read_text(encoding="utf-8")
        )
        if (
            summary.plan_digest != plan.canonical_digest()
            or summary.lineage != plan.resolved_config.lineage
            or paired.configured_final_seeds != plan.seeds.stages.final
            or paired.paired_worst_group_by_seed
            != summary.paired_worst_group_by_seed
            or paired.paired_worst_group_summary
            != summary.paired_worst_group_summary
        ):
            raise ValueError("Waterbirds completion summary is inconsistent")
    index = ExperimentIndex.model_validate_json(
        index_path.read_text(encoding="utf-8")
    )
    verify_experiment_index(root, index)
    return True


def persist_canonical_artifact(path: Path, model: StrictBoundaryModel) -> None:
    payload = model.canonical_json() + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"refusing to overwrite incompatible artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_experiment_index(plan: SearchPlan, output_root: Path) -> ExperimentIndex:
    index_path = output_root / "experiment-index.json"
    artifacts: list[IndexedArtifact] = []
    paths = tuple(output_root.rglob("*.json")) + tuple(
        output_root.rglob("*.yaml")
    )
    for path in sorted(paths):
        if path == index_path or any(
            ".incomplete" in part or ".interrupted-" in part
            for part in path.parts
        ):
            continue
        relative = path.relative_to(output_root).as_posix()
        schema_version: str | None = None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                mapping = cast(dict[str, object], loaded)
                value = mapping.get("schema_version")
                if isinstance(value, str):
                    schema_version = value
        except (OSError, ValueError):
            schema_version = None
        artifacts.append(
            IndexedArtifact(
                kind=_artifact_kind(relative),
                relative_path=relative,
                sha256=_file_sha256(path),
                schema_version=schema_version,
            )
        )
    index = ExperimentIndex(
        schema_version="grit.experiment-index/v1",
        dataset=plan.dataset,
        reportable=True,
        plan_digest=plan.canonical_digest(),
        resolved_config_digest=plan.resolved_config.canonical_digest(),
        artifacts=tuple(artifacts),
    )
    persist_canonical_artifact(index_path, index)
    return verify_experiment_index(output_root, index)


def _artifact_kind(relative: str) -> str:
    name = Path(relative).name
    if name == "result.json":
        return "stage_run"
    if name == "final-result.json":
        return "final_result"
    if "checkpoint" in name or "selected-checkpoint" in relative:
        return "selected_checkpoint_manifest"
    if "summary" in name:
        return "dataset_summary"
    if "winner" in name:
        return "frozen_winner"
    if "finalist" in name or "union" in name:
        return "selection_artifact"
    if "projection" in relative:
        return "projection_diagnostics"
    if name == "search-plan.json":
        return "search_plan"
    if name == "resolved-config.json":
        return "resolved_config"
    return "canonical_json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
