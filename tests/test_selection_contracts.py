"""Validation-only CMNIST selection and deterministic tie tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from pydantic import ValidationError

from grit.schemas import CmnistSelector, SeedStage
from grit.selection import (
    DiagnosticMetricRecord,
    FinalTestMetricRecord,
    ValidationMetricRecord,
    freeze_candidate,
    freeze_final_checkpoint,
    make_finalist_union,
    rank_candidates,
    select_candidate,
    select_checkpoint,
)
from tests.contract_fixtures import ordinary_grit_config, validation_records


def _candidate_records(
    candidate_id: str,
    scores: tuple[float, float, float],
    *,
    rank: int,
    scientific_config_digest: str,
    seeds: tuple[int, ...] = (101,),
    seed_stage: SeedStage = SeedStage.TUNING,
) -> tuple[ValidationMetricRecord, ...]:
    records: list[ValidationMetricRecord] = []
    for seed in seeds:
        run_id = f"run:{candidate_id}:{seed}"
        records.extend(
            validation_records(
                candidate_id=candidate_id,
                scientific_config_digest=scientific_config_digest,
                run_id=run_id,
                seed_stage=seed_stage,
                seed=seed,
                checkpoint_id=f"checkpoint:{candidate_id}:{seed}",
                epoch=3,
                scores=scores,
                projection_rank=rank,
                method_id="grit",
            )
        )
    return tuple(records)


def test_validation_selectors_reject_final_and_diagnostic_metric_types() -> None:
    final_metric = FinalTestMetricRecord(
        record_id="metric:final",
        run_id="run:final",
        candidate_id="candidate:a",
        method_id="erm",
        scientific_config_digest="sha256:config",
        checkpoint_id="checkpoint:a",
        epoch=1,
        seed=301,
        value=0.8,
        sample_count=10,
        metric_kind="final_test",
        seed_stage=SeedStage.FINAL,
        split_name="test_ood",
        metric_name="accuracy",
    )
    diagnostic_metric = DiagnosticMetricRecord(
        record_id="metric:diagnostic",
        run_id="run:diagnostic",
        candidate_id="candidate:a",
        method_id="erm",
        scientific_config_digest="sha256:config",
        checkpoint_id="checkpoint:a",
        epoch=1,
        seed=101,
        value=0.9,
        sample_count=10,
        metric_kind="diagnostic_test_oracle",
        seed_stage=SeedStage.TUNING,
        split_name="test_ood",
        metric_name="accuracy",
    )
    invalid_final = cast(
        Sequence[ValidationMetricRecord], cast(object, (final_metric,))
    )
    invalid_diagnostic = cast(
        Sequence[ValidationMetricRecord], cast(object, (diagnostic_metric,))
    )
    with pytest.raises(TypeError, match="ValidationMetricRecord"):
        select_checkpoint(invalid_final, CmnistSelector.PRIMARY_ROBUST)
    with pytest.raises(TypeError, match="ValidationMetricRecord"):
        select_candidate(invalid_diagnostic, CmnistSelector.PRIMARY_ROBUST)


def test_primary_and_secondary_selectors_apply_distinct_prespecified_splits() -> None:
    records = (
        *_candidate_records(
            "candidate:source-strong",
            (0.9, 0.9, 0.4),
            rank=2,
            scientific_config_digest="sha256:source-strong",
        ),
        *_candidate_records(
            "candidate:robust",
            (0.7, 0.7, 0.7),
            rank=2,
            scientific_config_digest="sha256:robust",
        ),
    )
    primary = select_candidate(records, CmnistSelector.PRIMARY_ROBUST)
    secondary = select_candidate(records, CmnistSelector.SECONDARY_SOURCE)
    assert primary.candidate_id == "candidate:robust"
    assert secondary.candidate_id == "candidate:source-strong"


def test_candidate_ties_use_mean_rank_then_stable_identity_deterministically() -> None:
    mean_tie_records = (
        *_candidate_records(
            "candidate:high-mean",
            (0.5, 0.9, 0.7),
            rank=3,
            scientific_config_digest="sha256:high-mean",
        ),
        *_candidate_records(
            "candidate:low-mean",
            (0.5, 0.5, 0.5),
            rank=1,
            scientific_config_digest="sha256:low-mean",
        ),
    )
    assert (
        select_candidate(mean_tie_records, CmnistSelector.PRIMARY_ROBUST).candidate_id
        == "candidate:high-mean"
    )

    rank_tie_records = (
        *_candidate_records(
            "candidate:rank-two",
            (0.6, 0.6, 0.6),
            rank=2,
            scientific_config_digest="sha256:rank-two",
        ),
        *_candidate_records(
            "candidate:rank-one",
            (0.6, 0.6, 0.6),
            rank=1,
            scientific_config_digest="sha256:rank-one",
        ),
    )
    assert (
        select_candidate(rank_tie_records, CmnistSelector.PRIMARY_ROBUST).candidate_id
        == "candidate:rank-one"
    )

    identity_tie_records = (
        *_candidate_records(
            "candidate:b",
            (0.6, 0.6, 0.6),
            rank=1,
            scientific_config_digest="sha256:b",
        ),
        *_candidate_records(
            "candidate:a",
            (0.6, 0.6, 0.6),
            rank=1,
            scientific_config_digest="sha256:a",
        ),
    )
    forward = select_candidate(identity_tie_records, CmnistSelector.PRIMARY_ROBUST)
    reverse = select_candidate(
        tuple(reversed(identity_tie_records)), CmnistSelector.PRIMARY_ROBUST
    )
    assert forward.candidate_id == reverse.candidate_id == "candidate:a"


def test_checkpoint_ties_prefer_earlier_epoch_then_stable_identity() -> None:
    scientific_config_digest = ordinary_grit_config().scientific_config_digest()
    records = (
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:later",
            epoch=2,
            scores=(0.7, 0.7, 0.7),
            projection_rank=2,
            method_id="grit",
        ),
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:earlier",
            epoch=1,
            scores=(0.7, 0.7, 0.7),
            projection_rank=2,
            method_id="grit",
        ),
    )
    decision = select_checkpoint(records, CmnistSelector.PRIMARY_ROBUST)
    assert decision.checkpoint.checkpoint_id == "checkpoint:earlier"

    same_epoch = tuple(
        record.model_copy(
            update={
                "checkpoint_id": "checkpoint:a"
                if record.checkpoint_id == "checkpoint:earlier"
                else "checkpoint:b",
                "epoch": 1,
                "record_id": record.record_id.replace(
                    record.checkpoint_id,
                    "checkpoint:a"
                    if record.checkpoint_id == "checkpoint:earlier"
                    else "checkpoint:b",
                ),
            }
        )
        for record in records
    )
    stable = select_checkpoint(same_epoch, CmnistSelector.PRIMARY_ROBUST)
    assert stable.checkpoint.checkpoint_id == "checkpoint:a"


def test_checkpoint_selection_cannot_change_candidate_rank() -> None:
    scientific_config_digest = ordinary_grit_config().scientific_config_digest()
    records = (
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:rank-one",
            epoch=1,
            scores=(0.7, 0.7, 0.7),
            projection_rank=1,
            method_id="grit",
        ),
        *validation_records(
            candidate_id="candidate:a",
            scientific_config_digest=scientific_config_digest,
            run_id="run:final",
            seed_stage=SeedStage.FINAL,
            seed=301,
            checkpoint_id="checkpoint:rank-two",
            epoch=2,
            scores=(0.8, 0.8, 0.8),
            projection_rank=2,
            method_id="grit",
        ),
    )
    with pytest.raises(ValueError, match="cannot change projection rank"):
        select_checkpoint(records, CmnistSelector.PRIMARY_ROBUST)


def test_candidate_selection_rejects_duplicate_seed_runs() -> None:
    records = (
        *_candidate_records(
            "candidate:a",
            (0.7, 0.7, 0.7),
            rank=1,
            scientific_config_digest="sha256:a",
        ),
        *tuple(
            record.model_copy(
                update={
                    "run_id": "run:candidate:a:duplicate",
                    "record_id": f"{record.record_id}:duplicate",
                }
            )
            for record in _candidate_records(
                "candidate:a",
                (0.9, 0.9, 0.9),
                rank=1,
                scientific_config_digest="sha256:a",
            )
        ),
    )
    with pytest.raises(ValueError, match="duplicate runs"):
        rank_candidates(records, CmnistSelector.PRIMARY_ROBUST)


def test_candidate_and_checkpoint_freezes_are_distinct_and_final_is_bounded() -> None:
    scientific_config_digest = ordinary_grit_config().scientific_config_digest()
    tuning = _candidate_records(
        "candidate:a",
        (0.7, 0.7, 0.7),
        rank=2,
        scientific_config_digest=scientific_config_digest,
        seeds=(101, 102, 103),
    )
    candidate = freeze_candidate(
        select_candidate(
            (
                *tuning,
                *_candidate_records(
                    "candidate:a",
                    (0.7, 0.7, 0.7),
                    rank=2,
                    scientific_config_digest=scientific_config_digest,
                    seeds=(201, 202),
                    seed_stage=SeedStage.CONFIRMATION,
                ),
            ),
            CmnistSelector.PRIMARY_ROBUST,
        ),
        ordinary_grit_config().seed_sets,
    )
    final_a = validation_records(
        candidate_id="candidate:a",
        scientific_config_digest=scientific_config_digest,
        run_id="run:final-a",
        seed_stage=SeedStage.FINAL,
        seed=301,
        checkpoint_id="checkpoint:final-a",
        epoch=2,
        scores=(0.8, 0.8, 0.8),
        projection_rank=2,
        method_id="grit",
    )
    checkpoint = freeze_final_checkpoint(
        select_checkpoint(final_a, CmnistSelector.PRIMARY_ROBUST), candidate
    )
    assert checkpoint.candidate_selection_id == candidate.frozen_selection_id
    assert checkpoint.frozen_checkpoint_id != candidate.frozen_selection_id

    final_b = validation_records(
        candidate_id="candidate:b",
        scientific_config_digest="sha256:different-config",
        run_id="run:final-b",
        seed_stage=SeedStage.FINAL,
        seed=301,
        checkpoint_id="checkpoint:final-b",
        epoch=2,
        scores=(0.99, 0.99, 0.99),
        projection_rank=1,
        method_id="grit",
    )
    with pytest.raises(ValueError, match="cannot change the frozen candidate"):
        freeze_final_checkpoint(
            select_checkpoint(final_b, CmnistSelector.PRIMARY_ROBUST), candidate
        )
    with pytest.raises(ValueError, match="final-stage"):
        select_candidate(final_a, CmnistSelector.PRIMARY_ROBUST)


def test_primary_secondary_finalists_form_one_union_and_keep_separate_winners() -> None:
    candidates = (
        ("candidate:a", (0.9, 0.9, 0.4)),
        ("candidate:b", (0.8, 0.8, 0.8)),
        ("candidate:c", (0.7, 0.7, 0.7)),
        ("candidate:d", (0.6, 0.6, 0.6)),
    )
    records = tuple(
        record
        for candidate_id, scores in candidates
        for record in _candidate_records(
            candidate_id,
            scores,
            rank=2,
            scientific_config_digest=f"sha256:{candidate_id}",
        )
    )
    primary = rank_candidates(records, CmnistSelector.PRIMARY_ROBUST)
    secondary = rank_candidates(records, CmnistSelector.SECONDARY_SOURCE)
    finalists = make_finalist_union(primary, secondary, finalists_per_selector=3)
    assert primary[0].candidate_id == "candidate:b"
    assert secondary[0].candidate_id == "candidate:a"
    assert finalists.confirmation_candidate_ids == (
        "candidate:b",
        "candidate:c",
        "candidate:d",
        "candidate:a",
    )

    mixed_method = secondary[0].model_copy(update={"method_id": "erm"})
    with pytest.raises(ValueError, match="one method"):
        make_finalist_union(
            primary,
            (mixed_method, *secondary[1:]),
            finalists_per_selector=3,
        )

    conflicting_candidate = secondary[0].model_copy(
        update={"scientific_config_digest": "sha256:conflicting"}
    )
    with pytest.raises(ValueError, match="same candidate set"):
        make_finalist_union(
            primary,
            (conflicting_candidate, *secondary[1:]),
            finalists_per_selector=3,
        )


def test_metric_records_reject_nan_and_infinity() -> None:
    with pytest.raises(ValidationError, match="finite"):
        validation_records(
            candidate_id="candidate:a",
            scientific_config_digest="sha256:a",
            run_id="run:a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            checkpoint_id="checkpoint:a",
            epoch=1,
            scores=(float("nan"), 0.5, 0.5),
            projection_rank=1,
            method_id="grit",
        )
    with pytest.raises(ValidationError, match="finite"):
        validation_records(
            candidate_id="candidate:a",
            scientific_config_digest="sha256:a",
            run_id="run:a",
            seed_stage=SeedStage.TUNING,
            seed=101,
            checkpoint_id="checkpoint:a",
            epoch=1,
            scores=(float("inf"), 0.5, 0.5),
            projection_rank=1,
            method_id="grit",
        )
