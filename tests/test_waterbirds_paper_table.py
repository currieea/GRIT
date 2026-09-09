"""Waterbirds paper table: every CMNIST method under both selector tracks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from grit.config import (
    FishAlgorithmConfig,
    GroupDroAlgorithmConfig,
    IrmAlgorithmConfig,
    LisaAlgorithmConfig,
    MatchDgAlgorithmConfig,
    RexAlgorithmConfig,
    SwadAlgorithmConfig,
)
from grit.features.waterbirds import (
    WaterbirdsEvaluationFeatureTable,
    WaterbirdsFeatureCache,
    WaterbirdsFeatureCacheManifest,
    WaterbirdsFinalTestHandle,
)
from grit.methods.invariance import environment_balanced_epoch_batches
from grit.methods.types import IMPLEMENTED_METHODS
from grit.methods.waterbirds_training import WaterbirdsRestorationReceipt
from grit.schemas import SeedStage
from grit.search.outputs import WaterbirdsProductionSummary
from grit.search.plan import (
    SearchPlan,
    WaterbirdsProductionSearchConfig,
    load_production_search_config,
)
from grit.search.run import (
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.search.scheduler import WaterbirdsCompletedStageRun
from grit.search.waterbirds import materialize_waterbirds_candidate_config
from grit.search.waterbirds_contracts import (
    WaterbirdsRunResult,
    WaterbirdsTestOracleRunResult,
)
from grit.selection.waterbirds import (
    WaterbirdsDiagnosticMetricRecord,
    WaterbirdsValidationMetricRecord,
    freeze_waterbirds_candidate,
    freeze_waterbirds_final_checkpoint,
    make_waterbirds_tuning_finalists,
    select_confirmed_waterbirds_candidate,
    select_waterbirds_checkpoint,
)
from tests.production_artifact_fixtures import (
    write_manifest_only_waterbirds_production,
)
from tests.test_waterbirds_selection import (
    _record,  # pyright: ignore[reportPrivateUsage]
    _seed_sets,  # pyright: ignore[reportPrivateUsage]
)

FULL_GRID: dict[str, object] = {
    "methods": list(IMPLEMENTED_METHODS),
    "learning_rates": [0.01],
    "weight_decays": [0.0, 0.0001, 0.001],
    "projection_ranks": [2],
    "groupdro_step_sizes": [0.01],
    "rex_penalty_weights": [10.0],
    "rex_penalty_anneal_updates": 100,
    "irm_penalty_weights": [100.0],
    "irm_penalty_anneal_updates": 190,
    "fish_meta_step_sizes": [0.1],
    "lisa_selection_probs": [0.5],
    "swad_tolerance_ratios": [0.3],
    "matchdg_latent_dims": [8],
    "matchdg_penalty_weights": [1.0],
}


def write_waterbirds_production_config(
    root: Path,
    *,
    output_root: Path,
    selectors: tuple[str, ...] = ("waterbirds_validation_worst_group",),
    search_space: dict[str, object] | None = None,
    max_epochs: int = 1,
) -> tuple[Path, tuple[Path, Path, Path]]:
    artifact_paths = write_manifest_only_waterbirds_production(root / "prepared")
    payload: dict[str, object] = {
        "schema_version": "grit.production-search/v1",
        "dataset": "waterbirds_cf",
        "protocol_id": "waterbirds_cf/v1",
        "experiment_name": "waterbirds-paper-table-check",
        "experiment_variant": "primary_unnormalized",
        "normalization": "none",
        "artifacts": {
            "dataset_manifest": artifact_paths[0].as_posix(),
            "feature_cache_manifest": artifact_paths[1].as_posix(),
            "oracle_pair_manifest": artifact_paths[2].as_posix(),
        },
        "seeds": {
            "construction": 17,
            "pairs": 2718,
            "stages": {
                "tuning": [1101, 1102, 1103],
                "confirmation": [2101, 2102],
                "final": [3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110],
            },
        },
        "search_space": search_space or dict(FULL_GRID),
        "selectors": list(selectors),
        "pair_count": 240,
        "batch_size": 256,
        "max_epochs": max_epochs,
        "output_root": output_root.as_posix(),
        "runtime": {"device": "cpu", "deterministic_algorithms": True, "workers": 1},
        "relative_singular_value_tolerance": 1e-12,
    }
    config_path = root / "production.yaml"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, artifact_paths


def fake_cache(plan: SearchPlan, feature_manifest_path: Path) -> WaterbirdsFeatureCache:
    """The real production manifest over random features with a weak label signal."""

    manifest = WaterbirdsFeatureCacheManifest.model_validate_json(
        feature_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.canonical_digest() == (
        plan.resolved_config.lineage.feature_cache_manifest_digest
    )
    generator = torch.Generator().manual_seed(0)
    features = torch.randn((len(manifest.records), 512), generator=generator)
    labels = torch.tensor([record.bird_label for record in manifest.records])
    backgrounds = torch.tensor([record.background for record in manifest.records])
    features[:, 0] += 2.0 * labels.to(torch.float32)
    features[:, 1] += 3.0 * backgrounds.to(torch.float32)
    return WaterbirdsFeatureCache(
        manifest=manifest, features=features, root=feature_manifest_path.parent
    )


def test_unequal_environments_are_balanced_and_exhaust_the_largest_once() -> None:
    ids = torch.tensor([0] * 3554 + [1] * 1241)
    batches = environment_balanced_epoch_batches(
        ids,
        environment_count=2,
        batch_size=256,
        generator=torch.Generator().manual_seed(1),
    )
    assert len(batches) == 28
    assert [len(batch) for batch in batches][-2:] == [256, 196]
    for batch in batches:
        assert int((ids[batch] == 0).sum()) == int((ids[batch] == 1).sum())
    land_rows = torch.cat([batch[ids[batch] == 0] for batch in batches])
    assert torch.equal(land_rows.sort().values, torch.arange(3554))
    equal = torch.tensor([0] * 100 + [1] * 100)
    reference = torch.Generator().manual_seed(3)
    shuffled = tuple(
        torch.nonzero(equal == e).flatten()[torch.randperm(100, generator=reference)]
        for e in (0, 1)
    )
    expected = tuple(
        torch.cat([rows[i : i + 32] for rows in shuffled]) for i in range(0, 100, 32)
    )
    observed = environment_balanced_epoch_batches(
        equal,
        environment_count=2,
        batch_size=64,
        generator=torch.Generator().manual_seed(3),
    )
    assert all(torch.equal(a, b) for a, b in zip(observed, expected, strict=True))


def _diagnostic(
    record: WaterbirdsValidationMetricRecord,
) -> WaterbirdsDiagnosticMetricRecord:
    payload = json.loads(record.canonical_json())
    payload["metric_kind"] = "diagnostic_test_oracle"
    payload["split_name"] = "test"
    payload["record_id"] = payload["record_id"] + ":test"
    return WaterbirdsDiagnosticMetricRecord.model_validate_json(json.dumps(payload))


def _diagnostics(
    candidates: tuple[tuple[str, tuple[int, int, int, int]], ...],
    stage: SeedStage,
    seeds: tuple[int, ...],
) -> tuple[WaterbirdsDiagnosticMetricRecord, ...]:
    return tuple(
        _diagnostic(
            _record(candidate=name, seed_stage=stage, seed=seed, correct=correct)
        )
        for name, correct in candidates
        for seed in seeds
    )


def test_test_oracle_selector_reads_only_labeled_test_records() -> None:
    validation = _record(candidate="a", seed_stage=SeedStage.TUNING, seed=101)
    diagnostic = _diagnostic(validation)
    decision = select_waterbirds_checkpoint((diagnostic,), "test_oracle")
    assert decision.selector == "test_oracle"
    with pytest.raises(TypeError, match="validation metrics only"):
        select_waterbirds_checkpoint((diagnostic,))
    with pytest.raises(TypeError, match="diagnostic test metrics only"):
        select_waterbirds_checkpoint((validation,), "test_oracle")
    seeds = _seed_sets()
    tuning = _diagnostics(
        (("a", (9, 9, 9, 9)), ("b", (8, 8, 8, 8)), ("c", (7, 7, 7, 7))),
        SeedStage.TUNING,
        seeds.tuning,
    )
    finalists = make_waterbirds_tuning_finalists(tuning, seeds, "test_oracle")
    assert finalists.selector == "test_oracle"
    with pytest.raises(TypeError, match="validation metrics only"):
        make_waterbirds_tuning_finalists(tuning, seeds)


def test_final_gate_refuses_test_oracle_selections() -> None:
    seeds = _seed_sets()
    names = (("a", (9, 9, 9, 9)), ("b", (8, 8, 8, 8)), ("c", (7, 7, 7, 7)))
    finalists = make_waterbirds_tuning_finalists(
        _diagnostics(names, SeedStage.TUNING, seeds.tuning), seeds, "test_oracle"
    )
    decision = select_confirmed_waterbirds_candidate(
        _diagnostics(names, SeedStage.CONFIRMATION, seeds.confirmation),
        finalists,
        seeds,
    )
    frozen = freeze_waterbirds_candidate(decision, finalists, seeds)
    assert frozen.selector == "test_oracle"
    final_record = _diagnostic(
        _record(candidate="a", seed_stage=SeedStage.FINAL, seed=seeds.final[0])
    )
    checkpoint = freeze_waterbirds_final_checkpoint(
        select_waterbirds_checkpoint((final_record,), "test_oracle"), frozen
    )
    identity = checkpoint.checkpoint
    restoration = WaterbirdsRestorationReceipt(
        receipt_id="restored:test",
        candidate_selection_id=frozen.frozen_selection_id,
        store_id="store:test",
        checkpoint=identity,
    )
    table = WaterbirdsEvaluationFeatureTable(
        dataset_manifest_digest="dataset:fixture",
        feature_cache_manifest_digest="features:fixture",
        normalization="none",
        split_role="final_test",
        record_ids=("r",),
        features=torch.zeros((1, 512)),
        labels=torch.zeros(1, dtype=torch.int64),
        backgrounds=torch.zeros(1, dtype=torch.int64),
        group_ids=("landbird_land",),
    )
    handle = WaterbirdsFinalTestHandle(
        run_id=identity.run_id,
        candidate_id=identity.candidate_id,
        method_id="erm",
        scientific_config_digest=identity.scientific_config_digest,
        seed=seeds.final[0],
        projection_rank=None,
        feature_cache_manifest_digest="features:fixture",
        table=table,
    )
    with pytest.raises(ValueError, match="validation-selected candidates only"):
        handle.open(frozen, checkpoint, restoration)
    with pytest.raises(ValueError, match="selector changed"):
        freeze_waterbirds_final_checkpoint(
            select_waterbirds_checkpoint(
                (
                    _record(
                        candidate="a", seed_stage=SeedStage.FINAL, seed=seeds.final[0]
                    ),
                )
            ),
            frozen,
        )


def test_plan_binds_every_method_to_waterbirds_environments_and_groups(
    tmp_path: Path,
) -> None:
    config_path, _ = write_waterbirds_production_config(
        tmp_path, output_root=tmp_path / "output"
    )
    plan = plan_production_search(config_path)
    assert plan.methods == IMPLEMENTED_METHODS
    assert len(plan.candidates) == 3 * len(IMPLEMENTED_METHODS)
    for candidate in plan.candidates:
        if candidate.method_id == "grit":
            continue
        for selector in ("waterbirds_validation_worst_group", "test_oracle"):
            resolved = materialize_waterbirds_candidate_config(
                plan, candidate, None, selector
            )
            assert resolved.selector == selector
            assert resolved.algorithm.kind == candidate.method_id
            assert (resolved.pair_manifest_digest is not None) == (
                candidate.method_id == "matchdg"
            )
            algorithm = resolved.algorithm
            if isinstance(
                algorithm,
                RexAlgorithmConfig | IrmAlgorithmConfig | FishAlgorithmConfig,
            ):
                assert algorithm.environment_names == (
                    "background_land",
                    "background_water",
                )
            if isinstance(algorithm, GroupDroAlgorithmConfig | LisaAlgorithmConfig):
                assert algorithm.group_definition == "target_background"
            if isinstance(algorithm, SwadAlgorithmConfig):
                assert algorithm.loss_split_names == ("validation",)
            if isinstance(algorithm, MatchDgAlgorithmConfig):
                assert algorithm.latent_dim == 8
    oracle_path, _ = write_waterbirds_production_config(
        tmp_path / "oracle",
        output_root=tmp_path / "oracle-output",
        selectors=("test_oracle",),
    )
    oracle_plan = plan_production_search(oracle_path)
    assert oracle_plan.selectors == ("test_oracle",)
    assert {c.scientific_config_digest for c in oracle_plan.candidates} == {
        c.scientific_config_digest for c in plan.candidates
    }, "the tracks retrain identical candidates"


@pytest.mark.parametrize(
    ("selectors", "methods"),
    [
        (("waterbirds_validation_worst_group",), IMPLEMENTED_METHODS),
        (("test_oracle",), ("erm", "grit", "fish", "matchdg")),
    ],
    ids=["ordinary", "test_oracle"],
)
def test_both_tracks_run_every_method_through_the_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selectors: tuple[str, ...],
    methods: tuple[str, ...],
) -> None:
    output_root = tmp_path / "output"
    grid = {
        key: value
        for key, value in FULL_GRID.items()
        if key.split("_")[0] not in set(IMPLEMENTED_METHODS) - set(methods)
    }
    grid["methods"] = list(methods)
    config_path, artifact_paths = write_waterbirds_production_config(
        tmp_path, output_root=output_root, selectors=selectors, search_space=grid
    )
    plan = plan_production_search(config_path)
    oracle = selectors == ("test_oracle",)
    cache = fake_cache(plan, artifact_paths[1])

    def fake_cache_loader(
        _plan: SearchPlan, *, tuning_only: bool = False
    ) -> WaterbirdsFeatureCache:
        assert not (oracle and tuning_only), "the oracle track needs test features"
        return cache

    monkeypatch.setattr("grit.search.waterbirds._load_cache", fake_cache_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, WaterbirdsProductionSummary)
    assert summary.selector == selectors[0]
    assert summary.test_oracle is oracle
    assert tuple(item.method_id for item in summary.methods) == methods
    assert summary.paired, "ERM and GRIT are both present"
    assert production_search_status(config_path).phase == "complete"

    stage_runs = [
        WaterbirdsCompletedStageRun.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(output_root.rglob("result.json"))
    ]
    assert len(stage_runs) == (
        plan.expected_run_counts.tuning
        + plan.expected_run_counts.confirmation_minimum
        + plan.expected_run_counts.final
    )
    assert all(run.test_oracle is oracle for run in stage_runs)
    final_texts = [
        path.read_text(encoding="utf-8")
        for path in sorted(output_root.rglob("final-result.json"))
    ]
    assert len(final_texts) == plan.expected_run_counts.final
    for text in final_texts:
        payload = json.loads(text)
        if oracle:
            result = WaterbirdsTestOracleRunResult.model_validate_json(text)
            assert "final_test_metric" not in payload
            assert "candidate_selection" not in payload
            assert result.test_oracle_candidate_selection.selector == "test_oracle"
            assert result.reported_test_metric.metric_kind == "diagnostic_test_oracle"
            assert result.validation_metrics, "validation metrics are still recorded"
        else:
            result = WaterbirdsRunResult.model_validate_json(text)
            assert "diagnostic_metrics" not in payload
            assert result.candidate_selection.selector == (
                "waterbirds_validation_worst_group"
            )
    selection_files = sorted(
        path.name for path in (output_root / "selection" / "matchdg").iterdir()
    )
    expected_files = (
        ["test-oracle-tuning-finalists.json", "test-oracle-winner.json"]
        if oracle
        else ["tuning-finalists.json", "winner.json"]
    )
    assert selection_files == expected_files
    assert (output_root / "summaries" / "waterbirds-paired-differences.json").is_file()


def test_checked_waterbirds_configs_cover_every_method_under_both_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJECT_SCRATCH", "/scratch")
    root = Path(__file__).resolve().parents[1] / "configs" / "waterbirds"
    seen: dict[tuple[str, str], str] = {}
    for path in sorted(root.glob("*-search*.yaml")):
        config = load_production_search_config(path)
        assert isinstance(config, WaterbirdsProductionSearchConfig)
        assert config.artifacts.dataset_manifest.startswith(
            "/scratch/artifacts/waterbirds-none/"
        )
        oracle = path.name.endswith("-test-oracle.yaml")
        assert config.test_oracle is oracle
        assert config.output_root.endswith("-test-oracle") is oracle
        for method in config.search_space.methods:
            seen[(method, "test_oracle" if oracle else "ordinary")] = path.name
    assert {key[0] for key in seen} == set(IMPLEMENTED_METHODS)
    assert {key[1] for key in seen} == {"ordinary", "test_oracle"}
    assert len(set(seen.values())) == 16
