"""Validation-only selection contracts for the CMNIST contract spine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.config import SeedSets
from grit.schemas import CmnistSelector, SeedStage, StrictBoundaryModel

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]


class _MetricIdentity(StrictBoundaryModel):
    record_id: NonEmptyStr
    run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    method_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    checkpoint_id: NonEmptyStr
    epoch: NonNegativeInt
    seed: StrictInt
    value: FiniteFloat
    sample_count: PositiveInt


class ValidationMetricRecord(_MetricIdentity):
    """The only metric record accepted by ordinary selectors."""

    metric_kind: Literal["validation"]
    seed_stage: SeedStage
    split_name: Literal["val_e01", "val_e02", "val_e05"]
    metric_name: Literal["accuracy"]
    projection_rank: NonNegativeInt | None


class FinalTestMetricRecord(_MetricIdentity):
    metric_kind: Literal["final_test"]
    seed_stage: Literal[SeedStage.FINAL]
    split_name: Literal["test_ood"]
    metric_name: Literal["accuracy"]


class DiagnosticMetricRecord(_MetricIdentity):
    metric_kind: Literal["diagnostic_test_oracle"]
    seed_stage: SeedStage
    split_name: Literal["test_ood"]
    metric_name: Literal["accuracy"]


class CheckpointIdentity(StrictBoundaryModel):
    checkpoint_id: NonEmptyStr
    candidate_id: NonEmptyStr
    run_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    epoch: NonNegativeInt


class CheckpointSelectionDecision(StrictBoundaryModel):
    decision_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    seed_stage: SeedStage
    seed: StrictInt
    checkpoint: CheckpointIdentity
    objective_value: FiniteFloat
    mean_accuracy: FiniteFloat
    projection_rank: NonNegativeInt | None
    contributing_record_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_contributors(self) -> CheckpointSelectionDecision:
        expected_count = 3 if self.selector is CmnistSelector.PRIMARY_ROBUST else 2
        if len(self.contributing_record_ids) != expected_count:
            raise ValueError(
                "checkpoint selection must cite every selector validation split"
            )
        if len(set(self.contributing_record_ids)) != len(self.contributing_record_ids):
            raise ValueError("checkpoint selection record IDs must be unique")
        return self


class CandidateSelectionDecision(StrictBoundaryModel):
    decision_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    objective_value: FiniteFloat
    mean_accuracy: FiniteFloat
    projection_rank: NonNegativeInt | None
    contributing_checkpoint_decisions: tuple[CheckpointSelectionDecision, ...]

    @model_validator(mode="after")
    def _validate_contributors(self) -> CandidateSelectionDecision:
        if not self.contributing_checkpoint_decisions:
            raise ValueError("candidate selection requires checkpoint decisions")
        seed_keys = [
            (decision.seed_stage, decision.seed)
            for decision in self.contributing_checkpoint_decisions
        ]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("candidate selection cannot contain duplicate seed runs")
        for decision in self.contributing_checkpoint_decisions:
            if decision.selector is not self.selector:
                raise ValueError(
                    "checkpoint selector does not match candidate selector"
                )
            if decision.method_id != self.method_id:
                raise ValueError("checkpoint method does not match candidate method")
            if decision.checkpoint.candidate_id != self.candidate_id:
                raise ValueError(
                    "checkpoint candidate does not match candidate decision"
                )
            if (
                decision.checkpoint.scientific_config_digest
                != self.scientific_config_digest
            ):
                raise ValueError("checkpoint config does not match candidate decision")
            if decision.seed_stage is SeedStage.FINAL:
                raise ValueError("final-stage validation cannot select a candidate")
            if decision.projection_rank != self.projection_rank:
                raise ValueError("checkpoint rank does not match candidate decision")
        objectives = tuple(
            float(decision.objective_value)
            for decision in self.contributing_checkpoint_decisions
        )
        means = tuple(
            float(decision.mean_accuracy)
            for decision in self.contributing_checkpoint_decisions
        )
        if fmean(objectives) != float(self.objective_value):
            raise ValueError("candidate objective does not match checkpoint decisions")
        if fmean(means) != float(self.mean_accuracy):
            raise ValueError("candidate mean does not match checkpoint decisions")
        return self


class FrozenCandidateSelection(StrictBoundaryModel):
    frozen_selection_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    candidate_id: NonEmptyStr
    scientific_config_digest: NonEmptyStr
    seed_sets: SeedSets
    decision: CandidateSelectionDecision

    @model_validator(mode="after")
    def _validate_decision(self) -> FrozenCandidateSelection:
        expected = (
            self.selector,
            self.method_id,
            self.candidate_id,
            self.scientific_config_digest,
        )
        observed = (
            self.decision.selector,
            self.decision.method_id,
            self.decision.candidate_id,
            self.decision.scientific_config_digest,
        )
        if observed != expected:
            raise ValueError("frozen candidate fields must match its decision")
        observed_seed_keys = {
            (checkpoint.seed_stage, checkpoint.seed)
            for checkpoint in self.decision.contributing_checkpoint_decisions
        }
        expected_seed_keys = {
            *((SeedStage.TUNING, seed) for seed in self.seed_sets.tuning),
            *((SeedStage.CONFIRMATION, seed) for seed in self.seed_sets.confirmation),
        }
        if observed_seed_keys != expected_seed_keys:
            raise ValueError(
                "frozen candidate requires exactly its configured tuning and "
                "confirmation seeds"
            )
        return self


class FrozenCheckpointSelection(StrictBoundaryModel):
    frozen_checkpoint_id: NonEmptyStr
    candidate_selection_id: NonEmptyStr
    selector: CmnistSelector
    method_id: NonEmptyStr
    checkpoint: CheckpointIdentity
    decision: CheckpointSelectionDecision

    @model_validator(mode="after")
    def _validate_decision(self) -> FrozenCheckpointSelection:
        if self.decision.seed_stage is not SeedStage.FINAL:
            raise ValueError(
                "only final-stage validation may freeze a final checkpoint"
            )
        if self.decision.selector is not self.selector:
            raise ValueError("frozen checkpoint selector mismatch")
        if self.decision.method_id != self.method_id:
            raise ValueError("frozen checkpoint method mismatch")
        if self.decision.checkpoint != self.checkpoint:
            raise ValueError("frozen checkpoint identity must match its decision")
        return self


class FinalistUnion(StrictBoundaryModel):
    """Primary/secondary top sets confirmed once without merging their winners."""

    primary_candidate_ids: tuple[NonEmptyStr, ...]
    secondary_candidate_ids: tuple[NonEmptyStr, ...]
    confirmation_candidate_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def _validate_union(self) -> FinalistUnion:
        expected = tuple(
            dict.fromkeys(self.primary_candidate_ids + self.secondary_candidate_ids)
        )
        if self.confirmation_candidate_ids != expected:
            raise ValueError(
                "confirmation candidates must be the ordered finalist union"
            )
        return self


class DiagnosticSelectionDecision(StrictBoundaryModel):
    """A conspicuously separate test-oracle decision record."""

    decision_kind: Literal["cmnist_test_oracle"]
    decision_id: NonEmptyStr
    candidate_id: NonEmptyStr
    checkpoint_id: NonEmptyStr
    objective_value: FiniteFloat
    contributing_record_ids: tuple[NonEmptyStr, ...]


def _require_validation_records(
    records: Sequence[ValidationMetricRecord],
) -> tuple[ValidationMetricRecord, ...]:
    materialized = tuple(records)
    if not materialized:
        raise ValueError("selection requires validation metric records")
    if any(type(record) is not ValidationMetricRecord for record in materialized):
        raise TypeError("ordinary selectors accept ValidationMetricRecord values only")
    return materialized


def _selector_splits(selector: CmnistSelector) -> tuple[str, ...]:
    if selector is CmnistSelector.PRIMARY_ROBUST:
        return ("val_e01", "val_e02", "val_e05")
    return ("val_e01", "val_e02")


def _checkpoint_sort_key(
    item: tuple[
        CheckpointIdentity,
        float,
        float,
        int | None,
        tuple[str, ...],
    ],
) -> tuple[float, float, int, int, str]:
    checkpoint, objective, mean_accuracy, projection_rank, _ = item
    rank_key = projection_rank if projection_rank is not None else 0
    return (
        -objective,
        -mean_accuracy,
        rank_key,
        checkpoint.epoch,
        checkpoint.checkpoint_id,
    )


def select_checkpoint(
    records: Sequence[ValidationMetricRecord],
    selector: CmnistSelector,
) -> CheckpointSelectionDecision:
    """Select one persisted checkpoint within one candidate/run/seed."""

    validation_records = _require_validation_records(records)
    run_keys = {
        (
            record.run_id,
            record.candidate_id,
            record.method_id,
            record.scientific_config_digest,
            record.seed_stage,
            record.seed,
        )
        for record in validation_records
    }
    if len(run_keys) != 1:
        raise ValueError(
            "checkpoint selection must stay within one candidate run and seed"
        )
    run_projection_ranks = {record.projection_rank for record in validation_records}
    if len(run_projection_ranks) != 1:
        raise ValueError("checkpoint selection cannot change projection rank")

    grouped: dict[str, list[ValidationMetricRecord]] = defaultdict(list)
    for record in validation_records:
        grouped[record.checkpoint_id].append(record)

    required_splits = _selector_splits(selector)
    candidates: list[
        tuple[CheckpointIdentity, float, float, int | None, tuple[str, ...]]
    ] = []
    for checkpoint_id, checkpoint_records in grouped.items():
        by_split = {record.split_name: record for record in checkpoint_records}
        if len(by_split) != len(checkpoint_records):
            raise ValueError("a checkpoint has duplicate validation split records")
        missing = set(required_splits) - set(by_split)
        if missing:
            missing_names = sorted(missing)
            raise ValueError(
                f"checkpoint {checkpoint_id!r} is missing selector splits "
                f"{missing_names!r}"
            )
        identities = {
            (
                record.candidate_id,
                record.run_id,
                record.scientific_config_digest,
                record.epoch,
            )
            for record in checkpoint_records
        }
        if len(identities) != 1:
            raise ValueError(
                "checkpoint validation records have inconsistent identities"
            )
        projection_ranks = {record.projection_rank for record in checkpoint_records}
        if len(projection_ranks) != 1:
            raise ValueError("checkpoint validation records have inconsistent ranks")
        selected_records = tuple(by_split[name] for name in required_splits)
        values = tuple(float(record.value) for record in selected_records)
        first = selected_records[0]
        checkpoint = CheckpointIdentity(
            checkpoint_id=checkpoint_id,
            candidate_id=first.candidate_id,
            run_id=first.run_id,
            scientific_config_digest=first.scientific_config_digest,
            epoch=first.epoch,
        )
        candidates.append(
            (
                checkpoint,
                min(values),
                fmean(values),
                projection_ranks.pop(),
                tuple(record.record_id for record in selected_records),
            )
        )

    selected = min(candidates, key=_checkpoint_sort_key)
    checkpoint, objective, mean_accuracy, projection_rank, record_ids = selected
    first_record = validation_records[0]
    decision_id = f"checkpoint-selection:{selector.value}:{checkpoint.checkpoint_id}"
    return CheckpointSelectionDecision(
        decision_id=decision_id,
        selector=selector,
        method_id=first_record.method_id,
        seed_stage=first_record.seed_stage,
        seed=first_record.seed,
        checkpoint=checkpoint,
        objective_value=objective,
        mean_accuracy=mean_accuracy,
        projection_rank=projection_rank,
        contributing_record_ids=record_ids,
    )


def rank_candidates(
    records: Sequence[ValidationMetricRecord],
    selector: CmnistSelector,
) -> tuple[CandidateSelectionDecision, ...]:
    """Rank candidates from tuning/confirmation validation records only."""

    validation_records = _require_validation_records(records)
    if any(record.seed_stage is SeedStage.FINAL for record in validation_records):
        raise ValueError("final-stage validation cannot enter candidate selection")
    methods = {record.method_id for record in validation_records}
    if len(methods) != 1:
        raise ValueError("methods must be selected independently")

    by_run: dict[tuple[str, str], list[ValidationMetricRecord]] = defaultdict(list)
    for record in validation_records:
        by_run[(record.candidate_id, record.run_id)].append(record)

    by_candidate: dict[str, list[CheckpointSelectionDecision]] = defaultdict(list)
    for (candidate_id, _), run_records in by_run.items():
        decision = select_checkpoint(run_records, selector)
        by_candidate[candidate_id].append(decision)

    seed_keys_by_candidate = {
        candidate_id: {(decision.seed_stage, decision.seed) for decision in decisions}
        for candidate_id, decisions in by_candidate.items()
    }
    if len({frozenset(keys) for keys in seed_keys_by_candidate.values()}) != 1:
        raise ValueError(
            "candidate comparisons require identical validation seed stages"
        )
    for decisions in by_candidate.values():
        seed_keys = [(decision.seed_stage, decision.seed) for decision in decisions]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("a candidate has duplicate runs for one seed stage")

    ranked: list[CandidateSelectionDecision] = []
    for candidate_id, decisions in by_candidate.items():
        decisions.sort(key=lambda decision: (decision.seed_stage.value, decision.seed))
        config_digests = {
            decision.checkpoint.scientific_config_digest for decision in decisions
        }
        projection_ranks = {decision.projection_rank for decision in decisions}
        if len(config_digests) != 1 or len(projection_ranks) != 1:
            raise ValueError(
                "candidate runs must share configuration and projection rank"
            )
        objective = fmean(float(decision.objective_value) for decision in decisions)
        mean_accuracy = fmean(float(decision.mean_accuracy) for decision in decisions)
        method_id = decisions[0].method_id
        ranked.append(
            CandidateSelectionDecision(
                decision_id=f"candidate-selection:{selector.value}:{candidate_id}",
                selector=selector,
                method_id=method_id,
                candidate_id=candidate_id,
                scientific_config_digest=config_digests.pop(),
                objective_value=objective,
                mean_accuracy=mean_accuracy,
                projection_rank=projection_ranks.pop(),
                contributing_checkpoint_decisions=tuple(decisions),
            )
        )

    def candidate_key(
        decision: CandidateSelectionDecision,
    ) -> tuple[float, float, int, str]:
        rank_key = (
            decision.projection_rank if decision.projection_rank is not None else 0
        )
        return (
            -float(decision.objective_value),
            -float(decision.mean_accuracy),
            rank_key,
            decision.candidate_id,
        )

    return tuple(sorted(ranked, key=candidate_key))


def select_candidate(
    records: Sequence[ValidationMetricRecord],
    selector: CmnistSelector,
) -> CandidateSelectionDecision:
    """Select a candidate using aggregated validation outcomes only."""

    return rank_candidates(records, selector)[0]


def make_finalist_union(
    primary_ranked: Sequence[CandidateSelectionDecision],
    secondary_ranked: Sequence[CandidateSelectionDecision],
    *,
    finalists_per_selector: int = 3,
) -> FinalistUnion:
    """Return one confirmation set while preserving both selector rankings."""

    if finalists_per_selector < 1:
        raise ValueError("finalists_per_selector must be positive")
    primary_all = tuple(primary_ranked)
    secondary_all = tuple(secondary_ranked)
    if len(primary_all) < finalists_per_selector:
        raise ValueError("primary ranking has too few finalists")
    if len(secondary_all) < finalists_per_selector:
        raise ValueError("secondary ranking has too few finalists")
    if any(item.selector is not CmnistSelector.PRIMARY_ROBUST for item in primary_all):
        raise ValueError("primary finalists must use the primary selector")
    if any(
        item.selector is not CmnistSelector.SECONDARY_SOURCE for item in secondary_all
    ):
        raise ValueError("secondary finalists must use the secondary selector")
    methods = {item.method_id for item in primary_all + secondary_all}
    if len(methods) != 1:
        raise ValueError("primary and secondary finalists must belong to one method")
    primary_candidates = {
        item.candidate_id: (item.scientific_config_digest, item.projection_rank)
        for item in primary_all
    }
    secondary_candidates = {
        item.candidate_id: (item.scientific_config_digest, item.projection_rank)
        for item in secondary_all
    }
    if primary_candidates != secondary_candidates:
        raise ValueError(
            "primary and secondary rankings must describe the same candidate set"
        )
    primary = primary_all[:finalists_per_selector]
    secondary = secondary_all[:finalists_per_selector]
    primary_ids = tuple(item.candidate_id for item in primary)
    secondary_ids = tuple(item.candidate_id for item in secondary)
    union_ids = tuple(dict.fromkeys(primary_ids + secondary_ids))
    return FinalistUnion(
        primary_candidate_ids=primary_ids,
        secondary_candidate_ids=secondary_ids,
        confirmation_candidate_ids=union_ids,
    )


def freeze_candidate(
    decision: CandidateSelectionDecision,
    seed_sets: SeedSets,
) -> FrozenCandidateSelection:
    """Freeze one selector's validation-selected scientific candidate."""

    observed_seed_keys = {
        (checkpoint.seed_stage, checkpoint.seed)
        for checkpoint in decision.contributing_checkpoint_decisions
    }
    expected_seed_keys = {
        *((SeedStage.TUNING, seed) for seed in seed_sets.tuning),
        *((SeedStage.CONFIRMATION, seed) for seed in seed_sets.confirmation),
    }
    if observed_seed_keys != expected_seed_keys:
        raise ValueError(
            "candidate freeze requires exactly the configured tuning and "
            "confirmation seeds"
        )
    return FrozenCandidateSelection(
        frozen_selection_id=f"frozen:{decision.decision_id}",
        selector=decision.selector,
        method_id=decision.method_id,
        candidate_id=decision.candidate_id,
        scientific_config_digest=decision.scientific_config_digest,
        seed_sets=seed_sets,
        decision=decision,
    )


def freeze_final_checkpoint(
    decision: CheckpointSelectionDecision,
    candidate: FrozenCandidateSelection,
) -> FrozenCheckpointSelection:
    """Freeze a final run's checkpoint without allowing candidate mutation."""

    if decision.seed_stage is not SeedStage.FINAL:
        raise ValueError("a frozen final checkpoint requires final-stage validation")
    if decision.selector is not candidate.selector:
        raise ValueError("checkpoint selector does not match the frozen candidate")
    if decision.method_id != candidate.method_id:
        raise ValueError("checkpoint method does not match the frozen candidate")
    if decision.seed not in candidate.seed_sets.final:
        raise ValueError("final checkpoint seed is not configured for final evaluation")
    if decision.checkpoint.candidate_id != candidate.candidate_id:
        raise ValueError("final-stage validation cannot change the frozen candidate")
    if (
        decision.checkpoint.scientific_config_digest
        != candidate.scientific_config_digest
    ):
        raise ValueError("final-stage validation cannot change frozen hyperparameters")
    return FrozenCheckpointSelection(
        frozen_checkpoint_id=f"frozen:{decision.decision_id}",
        candidate_selection_id=candidate.frozen_selection_id,
        selector=candidate.selector,
        method_id=candidate.method_id,
        checkpoint=decision.checkpoint,
        decision=decision,
    )
