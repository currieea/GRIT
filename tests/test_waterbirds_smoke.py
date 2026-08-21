"""End-to-end offline Waterbirds ERM/oracle-GRIT smoke lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from grit.config import SeedSets
from grit.projection import ProjectionDiagnostics
from grit.waterbirds_run_contracts import WaterbirdsRunResult, WaterbirdsSmokeSummary
from grit.waterbirds_runner import WaterbirdsSmokeRunConfig, run_waterbirds_smoke
from grit.waterbirds_selection import (
    WaterbirdsFinalTestMetricRecord,
    WaterbirdsValidationMetricRecord,
    select_waterbirds_checkpoint,
)


def _config(output_root: Path) -> WaterbirdsSmokeRunConfig:
    return WaterbirdsSmokeRunConfig(
        schema_version="grit.waterbirds-smoke/v1",
        non_reportable=True,
        output_root=output_root.as_posix(),
        construction_seed=17,
        fake_encoder_seed=31,
        normalization="none",
        projection_rank=1,
        relative_singular_value_tolerance=1e-12,
        learning_rates=(0.001, 0.003, 0.01),
        weight_decay=0.0,
        batch_size=8,
        max_epochs=1,
        seed_sets=SeedSets(
            tuning=(101, 102, 103),
            confirmation=(201, 202),
            final=(301, 302, 303, 304, 305, 306, 307, 308, 309, 310),
        ),
    )


def test_waterbirds_smoke_runs_both_methods_through_restored_final_gate(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "smoke"
    summary = run_waterbirds_smoke(_config(output_root))
    restored_summary = WaterbirdsSmokeSummary.model_validate_json(
        (output_root / "smoke-result.json").read_text(encoding="utf-8")
    )
    assert restored_summary == summary
    assert tuple(method.method_id for method in summary.methods) == ("erm", "grit")
    assert all(len(method.finalist_candidate_ids) == 3 for method in summary.methods)
    assert all(len(method.result_paths) == 10 for method in summary.methods)
    assert all(
        len(method.final_worst_group_accuracies) == 10 for method in summary.methods
    )
    assert all(
        len(method.final_adjusted_average_accuracies) == 10
        and len(method.final_raw_average_accuracies) == 10
        and method.worst_group_summary.seed_count == 10
        for method in summary.methods
    )
    assert len(summary.paired_worst_group_differences) == 10
    assert summary.paired_worst_group_summary.seed_count == 10

    results = tuple(
        WaterbirdsRunResult.model_validate_json(
            (output_root / relative).read_text(encoding="utf-8")
        )
        for method in summary.methods
        for relative in method.result_paths
    )
    assert len(results) == 20
    assert {result.resolved_config.method_id for result in results} == {
        "erm",
        "grit",
    }
    assert all(result.resolved_config.non_reportable for result in results)
    assert all(
        result.restoration.checkpoint == result.checkpoint_selection.checkpoint
        for result in results
    )
    assert all(
        result.final_test_metric.checkpoint_id
        == result.checkpoint_selection.checkpoint.checkpoint_id
        for result in results
    )
    assert all(len(result.final_test_metric.groups) == 4 for result in results)

    projection = ProjectionDiagnostics.model_validate_json(
        (output_root / "runs/grit/projection.json").read_text(encoding="utf-8")
    )
    assert projection.pair_manifest_digest == summary.pair_manifest_digest
    assert (
        projection.feature_cache_manifest_digest
        == summary.feature_cache_manifest_digest
    )

    final = results[0].final_test_metric
    with pytest.raises(TypeError, match="validation metrics only"):
        select_waterbirds_checkpoint((cast(WaterbirdsValidationMetricRecord, final),))


def test_waterbirds_smoke_is_reproducible_for_identical_seeds(
    tmp_path: Path,
) -> None:
    first = run_waterbirds_smoke(_config(tmp_path / "first"))
    second = run_waterbirds_smoke(_config(tmp_path / "second"))
    assert first.dataset_manifest_digest == second.dataset_manifest_digest
    assert first.pair_manifest_digest == second.pair_manifest_digest
    assert first.feature_cache_manifest_digest == second.feature_cache_manifest_digest
    assert tuple(item.selected_candidate_id for item in first.methods) == tuple(
        item.selected_candidate_id for item in second.methods
    )
    assert tuple(item.final_worst_group_accuracies for item in first.methods) == tuple(
        item.final_worst_group_accuracies for item in second.methods
    )


def test_waterbirds_result_round_trip_rejects_final_identity_tampering(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "smoke"
    summary = run_waterbirds_smoke(_config(output_root))
    result_path = output_root / summary.methods[0].result_paths[0]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["final_test_metric"]["candidate_id"] = "candidate:other"
    with pytest.raises(ValidationError, match="final metric identity"):
        WaterbirdsRunResult.model_validate_json(json.dumps(payload))


def test_final_metric_cannot_be_parsed_as_validation_metric(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    summary = run_waterbirds_smoke(_config(output_root))
    result = WaterbirdsRunResult.model_validate_json(
        (output_root / summary.methods[0].result_paths[0]).read_text(encoding="utf-8")
    )
    final_json = result.final_test_metric.canonical_json()
    restored = WaterbirdsFinalTestMetricRecord.model_validate_json(final_json)
    assert restored == result.final_test_metric
    with pytest.raises(ValidationError):
        WaterbirdsValidationMetricRecord.model_validate_json(final_json)
