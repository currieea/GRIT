"""The labeled test-oracle track: selector isolation and a full paired lifecycle."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

from grit.config import CmnistTestOracleExperimentConfig, OrdinarySelectionConfig
from grit.features.cmnist import CmnistFeatureCache, FeatureTable
from grit.results import CmnistTestOracleDiagnosticResult, OrdinaryRunResult
from grit.schemas import ORDINARY_CMNIST_SELECTORS, CmnistSelector, SeedStage
from grit.search.cmnist import materialize_cmnist_candidate_config
from grit.search.outputs import CmnistProductionSummary
from grit.search.plan import CmnistProductionSearchConfig, SearchPlan
from grit.search.run import (
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.search.scheduler import CmnistCompletedStageRun
from grit.selection.cmnist import (
    DiagnosticMetricRecord,
    ValidationMetricRecord,
    select_checkpoint,
)
from tests.contract_fixtures import seed_sets, validation_records
from tests.test_search_plan import write_cmnist_production_config


def _diagnostic(epoch: int, value: float) -> DiagnosticMetricRecord:
    return DiagnosticMetricRecord(
        record_id=f"metric:run:checkpoint:{epoch}:test_ood",
        run_id="run:a",
        candidate_id="candidate:a",
        method_id="erm",
        scientific_config_digest="sha256:config",
        checkpoint_id=f"checkpoint:{epoch}",
        epoch=epoch,
        seed=101,
        value=value,
        sample_count=10,
        metric_kind="diagnostic_test_oracle",
        seed_stage=SeedStage.TUNING,
        split_name="test_ood",
        metric_name="accuracy",
        projection_rank=None,
    )


def test_test_oracle_selector_scores_test_ood_only_and_stays_separate() -> None:
    diagnostics = (_diagnostic(1, 0.5), _diagnostic(2, 0.9), _diagnostic(3, 0.9))
    decision = select_checkpoint(diagnostics, CmnistSelector.TEST_ORACLE)
    assert decision.selector is CmnistSelector.TEST_ORACLE
    assert decision.checkpoint.epoch == 2, "ties prefer the earlier epoch"
    assert decision.objective_value == 0.9
    assert decision.contributing_record_ids == (diagnostics[1].record_id,)

    validation = validation_records(
        candidate_id="candidate:a",
        scientific_config_digest="sha256:config",
        run_id="run:a",
        seed_stage=SeedStage.TUNING,
        seed=101,
        checkpoint_id="checkpoint:1",
        epoch=1,
        scores=(0.9, 0.8, 0.7),
        projection_rank=None,
    )
    with pytest.raises(TypeError, match="DiagnosticMetricRecord"):
        select_checkpoint(validation, CmnistSelector.TEST_ORACLE)
    for selector in ORDINARY_CMNIST_SELECTORS:
        with pytest.raises(TypeError, match="ValidationMetricRecord"):
            select_checkpoint(
                cast(Sequence[ValidationMetricRecord], diagnostics), selector
            )
    with pytest.raises(ValidationError, match="test_oracle"):
        OrdinarySelectionConfig(selector=CmnistSelector.TEST_ORACLE)
    with pytest.raises(ValidationError, match="selectors"):
        CmnistProductionSearchConfig.model_validate_json(
            json.dumps(
                {
                    "schema_version": "grit.production-search/v1",
                    "dataset": "cmnist",
                    "protocol_id": "cmnist/v1",
                    "experiment_name": "mixed",
                    "experiment_variant": "primary_unnormalized",
                    "normalization": "none",
                    "artifacts": {
                        "dataset_manifest": "d.json",
                        "feature_cache_manifest": "f.json",
                        "oracle_pair_manifest": "p.json",
                    },
                    "seeds": {
                        "construction": 1,
                        "pairs": 2,
                        "stages": seed_sets().model_dump(mode="json"),
                    },
                    "search_space": {
                        "methods": ["erm"],
                        "learning_rates": [0.001],
                        "weight_decays": [0.0, 0.0001, 0.001],
                    },
                    "selectors": ["primary_robust", "test_oracle"],
                    "pair_count": 256,
                    "batch_size": 256,
                    "max_epochs": 1,
                    "output_root": "out",
                    "runtime": {
                        "device": "cpu",
                        "deterministic_algorithms": True,
                        "workers": 1,
                    },
                    "relative_singular_value_tolerance": 1e-12,
                }
            )
        )


class _ManifestIdentity:
    def __init__(self, digest: str) -> None:
        self._digest = digest

    def canonical_digest(self) -> str:
        return self._digest


def _table(name: str, role: str, seed: int, rows: int = 8) -> FeatureTable:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn((rows, 512), generator=generator)
    targets = (features[:, 0] > 0).to(torch.int64)
    return FeatureTable(
        name=name,
        role=cast(Literal["training"], role),
        source_ids=tuple(f"source:{name}:{index}" for index in range(rows)),
        features=features,
        digits=targets * 5,
        clean_labels=targets,
        targets=targets,
        colors=torch.arange(rows) % 2,
    )


def fake_cache(plan: SearchPlan) -> CmnistFeatureCache:
    manifest = _ManifestIdentity(
        plan.resolved_config.lineage.feature_cache_manifest_digest
    )
    pair_red = _table("oracle_pair_red", "pair_projection", 7)
    pair_green = FeatureTable(
        name="oracle_pair_green",
        role="pair_projection",
        source_ids=pair_red.source_ids,
        features=pair_red.features + 1.0,
        digits=pair_red.digits,
        clean_labels=pair_red.clean_labels,
        targets=pair_red.targets,
        colors=1 - pair_red.colors,
    )
    return CmnistFeatureCache(
        train_e01=_table("train_e01", "training", 1),
        train_e02=_table("train_e02", "training", 2),
        val_e01=_table("val_e01", "validation", 3),
        val_e02=_table("val_e02", "validation", 4),
        val_e05=_table("val_e05", "validation", 5),
        _test_ood=_table("test_ood", "final_test", 6),
        oracle_pair_red=pair_red,
        oracle_pair_green=pair_green,
        manifest=cast(object, manifest),  # pyright: ignore[reportArgumentType]
        root=Path("."),
    )


@pytest.mark.parametrize(
    "selectors",
    [("primary_robust", "secondary_source"), ("test_oracle",)],
    ids=["ordinary", "test_oracle"],
)
def test_both_tracks_run_the_full_paired_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selectors: tuple[str, ...],
) -> None:
    output_root = tmp_path / "output"
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=output_root,
        overrides={
            "experiment_name": "cmnist-track-check",
            "search_space": {
                "methods": ["erm", "grit"],
                "learning_rates": [0.01],
                "weight_decays": [0.0, 0.0001, 0.001],
                "projection_ranks": [1, 2, 3],
            },
            "selectors": list(selectors),
            "max_epochs": 2,
        },
    )
    plan = plan_production_search(config_path)
    oracle = selectors == ("test_oracle",)
    assert plan.selectors == selectors
    assert ("finalist_union" in {s.artifact_kind for s in plan.output_schemas}) == (
        not oracle
    )

    cache = fake_cache(plan)
    loads: list[bool] = []

    def fake_cache_loader(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> CmnistFeatureCache:
        loads.append(tuning_only)
        return cache

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", fake_cache_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary)
    assert summary.selectors == tuple(CmnistSelector(s) for s in selectors)
    assert summary.test_oracle is oracle
    assert [(m.method_id, m.selector.value) for m in summary.methods] == [
        (method, selector) for method in ("erm", "grit") for selector in selectors
    ]
    assert [p.selector.value for p in summary.paired_selectors] == list(selectors)
    assert production_search_status(config_path).phase == "complete"

    stage_runs = [
        CmnistCompletedStageRun.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.rglob("result.json"))
    ]
    assert (
        len(stage_runs)
        == plan.expected_run_counts.tuning
        + len(tuple(output_root.glob("runs/confirmation/*/*/*")))
        + plan.expected_run_counts.final
    )
    assert all(run.test_oracle is oracle for run in stage_runs)
    final_results = [
        path.read_text(encoding="utf-8")
        for path in sorted(output_root.rglob("final-result.json"))
    ]
    assert len(final_results) == plan.expected_run_counts.final
    for text in final_results:
        payload = json.loads(text)
        if oracle:
            result = CmnistTestOracleDiagnosticResult.model_validate_json(text)
            assert result.resolved_config.run_kind == "cmnist_test_oracle_diagnostic"
            assert "candidate_selection" not in payload
            assert "final_test_metrics" not in payload
            candidate = result.test_oracle_candidate_selection
            checkpoint = result.test_oracle_checkpoint_selection
            selection = result.diagnostic_selection
            assert candidate is not None and checkpoint is not None
            assert selection is not None
            assert candidate.selector is CmnistSelector.TEST_ORACLE
            assert checkpoint.selector is CmnistSelector.TEST_ORACLE
            assert checkpoint.checkpoint.checkpoint_id == selection.checkpoint_id
            assert result.validation_metrics, "in-domain accuracies are kept"
        else:
            result = OrdinaryRunResult.model_validate_json(text)
            assert "diagnostic_metrics" not in payload
            assert result.candidate_selection is not None
            assert result.candidate_selection.selector.is_ordinary
    selection_files = sorted(
        path.name for path in (output_root / "selection" / "erm").iterdir()
    )
    if oracle:
        assert selection_files == [
            "test-oracle-tuning-finalists.json",
            "test_oracle-winner.json",
        ]
        assert loads == [False], "the oracle track always loads test features"
    else:
        assert selection_files == [
            "confirmation-union.json",
            "primary-tuning-finalists.json",
            "primary_robust-winner.json",
            "secondary-tuning-finalists.json",
            "secondary_source-winner.json",
        ]


def test_test_oracle_candidates_share_scientific_identity_with_ordinary_ones(
    tmp_path: Path,
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={"selectors": ["test_oracle"]},
    )
    plan = plan_production_search(config_path)
    candidate = plan.candidates[0]
    oracle = materialize_cmnist_candidate_config(
        plan, candidate, CmnistSelector.TEST_ORACLE
    )
    ordinary = materialize_cmnist_candidate_config(
        plan, candidate, CmnistSelector.PRIMARY_ROBUST
    )
    assert isinstance(oracle, CmnistTestOracleExperimentConfig)
    assert oracle.diagnostic_selection.test_oracle is True
    assert oracle.scientific_config_digest() == ordinary.scientific_config_digest()
    assert oracle.canonical_digest() != ordinary.canonical_digest()
