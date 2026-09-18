"""Carry each objective's three variants through the actual production lifecycle."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

from grit.config import (
    ComposedAlgorithmConfig,
    algorithm_method_id,
)
from grit.features.cmnist import CmnistFeatureCache
from grit.features.waterbirds import WaterbirdsFeatureCache
from grit.methods.training import LinearProbeAlgorithm, PersistedLinearCheckpointStore
from grit.methods.types import MethodId, consumes_pairs, pair_intervention
from grit.results import (
    CmnistTestOracleDiagnosticResult,
    CodeProvenance,
    OrdinaryRunResult,
)
from grit.search.outputs import CmnistProductionSummary, WaterbirdsProductionSummary
from grit.search.plan import SearchPlan
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    plan_production_search,
    production_search_status,
    run_production_search,
)
from grit.search.waterbirds_contracts import (
    WaterbirdsRunResult,
    WaterbirdsTestOracleRunResult,
)
from tests.test_search_plan import write_cmnist_production_config
from tests.test_test_oracle_track import fake_cache
from tests.test_waterbirds_paper_table import fake_cache as waterbirds_fake_cache
from tests.test_waterbirds_paper_table import write_waterbirds_production_config

Objective = Literal["erm", "rex", "irm", "fishr"]
Dataset = Literal["cmnist", "waterbirds"]
ROWS: dict[Objective, tuple[MethodId, MethodId, MethodId]] = {
    "erm": ("erm", "grit", "erm_consistency"),
    "rex": ("rex", "rex_grit", "rex_consistency"),
    "irm": ("irm", "irm_grit", "irm_consistency"),
    "fishr": ("fishr", "fishr_grit", "fishr_consistency"),
}


@pytest.fixture(autouse=True)
def _fixed_code_provenance(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = CodeProvenance(git_revision="1" * 40, git_dirty=True)
    monkeypatch.setattr("grit.search.plan._code_provenance", lambda: code)


def _space(objective: Objective) -> dict[str, object]:
    space: dict[str, object] = {
        "methods": list(ROWS[objective]),
        "learning_rates": [0.01],
        "weight_decays": [0.0, 0.0001, 0.001],
        "projection_ranks": [1],
        "consistency_weights": [0.1],
    }
    if objective != "erm":
        space[f"{objective}_penalty_weights"] = [10.0]
        # Exercise activation and the Fishr optimizer reset in this tiny run.
        space[f"{objective}_penalty_anneal_updates"] = 2
    if objective == "fishr":
        space["fishr_ema"] = 0.95
    return space


def _forbid_access(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("this phase/objective must not access this information")


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds"])
@pytest.mark.parametrize("objective", ["erm", "rex", "irm", "fishr"])
@pytest.mark.parametrize("factorized", [False, True])
def test_matrix_row_runs_selection_restoration_and_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    objective: Objective,
    factorized: bool,
) -> None:
    output_root = tmp_path / "output"
    space = _space(objective)
    methods: tuple[MethodId, ...] = tuple(method for method in ROWS[objective])
    if factorized:
        methods = (
            *methods,
            cast(MethodId, f"{objective}_representation_consistency"),
            cast(MethodId, f"{objective}_two_layer"),
        )
        space.update(
            methods=list(methods),
            representation_consistency_weights=[0.1],
            representation_latent_dims=[4],
        )
    if dataset == "cmnist":
        config_path, _ = write_cmnist_production_config(
            tmp_path,
            output_root=output_root,
            overrides={
                "search_space": space,
                "batch_size": 8,
                "max_epochs": 2,
            },
        )
        plan = plan_production_search(config_path)
        cache = fake_cache(plan)

        def cmnist_loader(_plan: SearchPlan, *, tuning_only: bool = False):
            return cache

        monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", cmnist_loader)
    else:
        config_path, artifacts = write_waterbirds_production_config(
            tmp_path, output_root=output_root, search_space=space, max_epochs=1
        )
        plan = plan_production_search(config_path)
        cache = waterbirds_fake_cache(plan, artifacts[1])

        def waterbirds_loader(_plan: SearchPlan, *, tuning_only: bool = False):
            return cache

        monkeypatch.setattr("grit.search.waterbirds._load_cache", waterbirds_loader)
        # Interventions must not grant supervised group metadata to these bases.
        monkeypatch.setattr(
            WaterbirdsFeatureCache, "training_group_ids", _forbid_access
        )
        if objective == "erm":
            monkeypatch.setattr(
                WaterbirdsFeatureCache, "training_environment_ids", _forbid_access
            )

    assert Counter(candidate.method_id for candidate in plan.candidates) == {
        method: 3 for method in methods
    }
    with monkeypatch.context() as guarded:
        cache_type = type(cache)
        guarded.setattr(cache_type, "issue_final_handle", _forbid_access)
        guarded.setattr(cache_type, "open_test_oracle_table", _forbid_access)
        status = run_production_search(
            config_path, ProductionExecutionLimits(stop_after="tuning")
        )
    assert isinstance(status, ProductionSearchStatus)
    assert status.tuning_complete == status.tuning_expected == 9 * len(methods)
    assert status.final_complete == 0
    assert not tuple(output_root.rglob("final-result.json"))

    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary | WaterbirdsProductionSummary)
    assert summary.test_oracle is False
    selector_count = 2 if dataset == "cmnist" else 1
    assert Counter(item.method_id for item in summary.methods) == {
        method: selector_count for method in methods
    }
    expected_contrasts = {
        (methods[1], methods[0]),
        (methods[2], methods[0]),
        (methods[1], methods[2]),
    }
    if factorized:
        expected_contrasts.update(
            {
                (methods[3], methods[0]),
                (methods[1], methods[3]),
                (methods[3], methods[2]),
                (methods[3], methods[4]),
            }
        )
        diagnostic_paths = tuple(output_root.rglob("representation-diagnostics.json"))
        assert diagnostic_paths
        for diagnostic_path in diagnostic_paths:
            history = json.loads(diagnostic_path.read_text())
            assert history
            for values in history.values():
                assert values["representation/weight_norm"] > 0.0
                assert values["classifier/weight_norm"] > 0.0
                assert ("pairs/representation_discrepancy" in values) == (
                    "two_layer" not in str(diagnostic_path)
                )
    comparisons = summary.intervention_comparisons
    assert {(c.minuend, c.subtrahend) for c in comparisons} == expected_contrasts
    assert len(comparisons) == len(expected_contrasts) * (
        2 if dataset == "cmnist" else 3
    )
    for comparison in comparisons:
        assert comparison.base_objective == objective
        assert tuple(seed for seed, _ in comparison.differences_by_seed) == (
            plan.seeds.stages.final
        )
        assert comparison.ci95_lower <= comparison.mean <= comparison.ci95_upper
    status = production_search_status(config_path)
    assert status.phase == "complete"
    assert status.frozen_winner_count == len(methods) * selector_count
    assert status.final_complete == len(methods) * selector_count * 10
    for method in methods:
        assert len(
            tuple((output_root / "selection" / method).glob("*winner.json"))
        ) == (selector_count)

    # Reload one predictor per combination and compare its predictions to the
    # saved final metric, with no separately supplied projection object.
    checked: set[MethodId] = set()
    pair_digests: set[str] = set()
    for path in sorted(output_root.rglob("final-result.json")):
        result = (
            OrdinaryRunResult.model_validate_json(path.read_text())
            if dataset == "cmnist"
            else WaterbirdsRunResult.model_validate_json(path.read_text())
        )
        resolved = result.resolved_config
        method = algorithm_method_id(resolved.algorithm)
        if method in checked:
            continue
        checked.add(method)
        assert result.restoration is not None
        assert result.checkpoint_selection is not None
        assert result.restoration.checkpoint == result.checkpoint_selection.checkpoint
        if isinstance(result, OrdinaryRunResult):
            lineage = result.resolved_config.artifact_lineage
            assert lineage is not None
            pair_digest = lineage.pair_manifest_digest
            assert isinstance(cache, CmnistFeatureCache)
            features = cache._test_ood.features  # pyright: ignore[reportPrivateUsage]
            targets = cache._test_ood.targets  # pyright: ignore[reportPrivateUsage]
            assert result.final_test_metrics is not None
            saved_accuracy = float(result.final_test_metrics[0].value)
        else:
            pair_digest = result.resolved_config.pair_manifest_digest
            assert isinstance(cache, WaterbirdsFeatureCache)
            records = [
                r for r in cache.manifest.records if r.split_role == "final_test"
            ]
            features = cache.features[torch.tensor([r.row_index for r in records])]
            targets = torch.tensor([r.bird_label for r in records])
            saved_accuracy = float(result.final_test_metric.raw_average_accuracy)
        assert (pair_digest is not None) == consumes_pairs(method)
        if pair_digest is not None:
            pair_digests.add(pair_digest)
        store = PersistedLinearCheckpointStore(path.parent / "selected-checkpoint")
        stored = store.load(store.store_id.removeprefix("linear-checkpoint:"))
        assert (stored.state.factor_parameters is not None) == (
            pair_intervention(method) in ("representation_consistency", "two_layer")
        )
        basis = stored.state.projection_basis
        assert basis is not None
        assert basis.shape == (
            512,
            1 if pair_intervention(method) == "grit" else 0,
        )
        restored = LinearProbeAlgorithm(
            resolved.training, model_seed=0, projection=None
        )
        restored.restore_inference_state(stored.state)
        predictions = restored.predict(features)
        assert (
            float((predictions == targets).to(torch.float64).mean()) == saved_accuracy
        )
        if pair_intervention(method) == "grit":
            assert torch.equal(
                predictions, restored.predict(features + 10.0 * basis[:, 0])
            ), "restored predictor must remain invariant to the removed direction"
        if (
            isinstance(resolved.algorithm, ComposedAlgorithmConfig)
            and objective != "erm"
        ):
            payload = resolved.model_dump()
            payload["algorithm"]["base_objective"]["environment_names"] = (
                ("background_land", "background_water")
                if dataset == "cmnist"
                else ("train_e01", "train_e02")
            )
            with pytest.raises(ValidationError, match="invariant methods"):
                type(resolved).model_validate(payload)
    assert checked == set(methods)
    assert len(pair_digests) == 1, "both interventions use the same bank"


@pytest.mark.parametrize("objective", ["rex", "irm", "fishr"])
def test_matrix_candidates_jointly_search_objective_and_intervention(
    tmp_path: Path, objective: Objective
) -> None:
    space = _space(objective)
    space[f"{objective}_penalty_weights"] = [1.0, 10.0]
    space["projection_ranks"] = [1, 2]
    space["consistency_weights"] = [0.1, 1.0, 10.0]
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={"search_space": space},
    )
    plan = plan_production_search(config_path)
    vanilla, projected, consistency = ROWS[objective]
    assert Counter(candidate.method_id for candidate in plan.candidates) == {
        vanilla: 3 * 2,
        projected: 3 * 2 * 2,
        consistency: 3 * 2 * 3,
    }
    assert {
        (candidate.penalty_weight, candidate.requested_rank)
        for candidate in plan.candidates
        if candidate.method_id == projected
    } == {(weight, rank) for weight in (1.0, 10.0) for rank in (1, 2)}
    assert {
        (candidate.penalty_weight, candidate.consistency_weight)
        for candidate in plan.candidates
        if candidate.method_id == consistency
    } == {(weight, strength) for weight in (1.0, 10.0) for strength in (0.1, 1.0, 10.0)}


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds"])
def test_composed_irm_diagnostic_lifecycle_stays_explicitly_test_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> None:
    output_root = tmp_path / "diagnostic-output"
    methods = ROWS["irm"][1:]
    space = _space("irm")
    space["methods"] = list(methods)
    if dataset == "cmnist":
        config_path, _ = write_cmnist_production_config(
            tmp_path,
            output_root=output_root,
            overrides={
                "search_space": space,
                "selectors": ["test_oracle"],
                "batch_size": 8,
                "max_epochs": 2,
            },
        )
        plan = plan_production_search(config_path)
        cache = fake_cache(plan)
        loader_name = "grit.search.cmnist._load_cmnist_cache"
    else:
        config_path, artifacts = write_waterbirds_production_config(
            tmp_path,
            output_root=output_root,
            selectors=("test_oracle",),
            search_space=space,
        )
        plan = plan_production_search(config_path)
        cache = waterbirds_fake_cache(plan, artifacts[1])
        loader_name = "grit.search.waterbirds._load_cache"

    def diagnostic_loader(_plan: SearchPlan, *, tuning_only: bool = False):
        assert not tuning_only, "test access is explicit for the diagnostic track"
        return cache

    monkeypatch.setattr(loader_name, diagnostic_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary | WaterbirdsProductionSummary)
    assert summary.test_oracle
    assert tuple(item.method_id for item in summary.methods) == methods
    comparisons = summary.intervention_comparisons
    assert len(comparisons) == (1 if dataset == "cmnist" else 3)
    assert all(
        comparison.selector == "test_oracle"
        and comparison.minuend == "irm_grit"
        and comparison.subtrahend == "irm_consistency"
        for comparison in comparisons
    )
    status = production_search_status(config_path)
    assert status.phase == "complete"
    assert status.frozen_winner_count == 2
    assert status.final_complete == 20
    paths = tuple(output_root.rglob("final-result.json"))
    assert len(paths) == 20
    for path in paths:
        text = path.read_text()
        payload = json.loads(text)
        for ordinary_field in (
            "candidate_selection",
            "checkpoint_selection",
            "final_test_metric",
            "final_test_metrics",
        ):
            assert ordinary_field not in payload
        result = (
            CmnistTestOracleDiagnosticResult.model_validate_json(text)
            if dataset == "cmnist"
            else WaterbirdsTestOracleRunResult.model_validate_json(text)
        )
        assert result.test_oracle_candidate_selection is not None
        assert result.test_oracle_checkpoint_selection is not None
        assert result.test_oracle_candidate_selection.selector == "test_oracle"
        assert (
            result.test_oracle_checkpoint_selection.decision.selector == "test_oracle"
        )
        assert result.validation_metrics, "the original validation records remain"
        assert result.diagnostic_metrics
        assert all(
            metric.metric_kind == "diagnostic_test_oracle"
            for metric in result.diagnostic_metrics
        )


@pytest.mark.parametrize("objective", ["erm", "rex", "irm", "fishr"])
def test_representation_grid_keeps_strengths_width_and_base_independent(
    tmp_path: Path,
    objective: Objective,
) -> None:
    space = _space(objective)
    representation = cast(MethodId, f"{objective}_representation_consistency")
    control = cast(MethodId, f"{objective}_two_layer")
    space.update(
        methods=[*ROWS[objective], representation, control],
        representation_consistency_weights=[0.0, 0.2],
        representation_latent_dims=[4, 8],
    )
    if objective != "erm":
        space[f"{objective}_penalty_weights"] = [1.0, 10.0]
    path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={"search_space": space},
    )
    plan = plan_production_search(path)
    reps = [c for c in plan.candidates if c.method_id == representation]
    controls = [c for c in plan.candidates if c.method_id == control]
    base_weights = (None,) if objective == "erm" else (1.0, 10.0)
    assert {
        (
            c.penalty_weight,
            c.representation_consistency_weight,
            c.representation_latent_dim,
        )
        for c in reps
    } == {
        (base, strength, width)
        for base in base_weights
        for strength in (0.0, 0.2)
        for width in (4, 8)
    }
    assert all(c.consistency_weight is None for c in (*reps, *controls))
    assert all(c.representation_consistency_weight == 0.0 for c in controls)
    assert {c.representation_latent_dim for c in controls} == {4, 8}
    assert {c.candidate_id for c in reps}.isdisjoint(c.candidate_id for c in controls)


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds"])
@pytest.mark.parametrize("objective", ["erm", "rex", "irm", "fishr"])
def test_two_layer_only_tuning_never_requests_pairs_or_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    objective: Objective,
) -> None:
    control = cast(MethodId, f"{objective}_two_layer")
    space = _space(objective)
    space.update(methods=[control], representation_latent_dims=[4])
    del space["consistency_weights"]
    del space["projection_ranks"]
    if dataset == "cmnist":
        path, _ = write_cmnist_production_config(
            tmp_path,
            output_root=tmp_path / "output",
            overrides={"search_space": space, "batch_size": 8, "max_epochs": 2},
        )
        plan = plan_production_search(path)
        cache = fake_cache(plan)
        monkeypatch.setattr(CmnistFeatureCache, "pair_tables", _forbid_access)
        monkeypatch.setattr("grit.search.cmnist._cmnist_pair_manifest", _forbid_access)
        loader = "grit.search.cmnist._load_cmnist_cache"
    else:
        path, artifacts = write_waterbirds_production_config(
            tmp_path,
            output_root=tmp_path / "output",
            search_space=space,
            max_epochs=1,
        )
        plan = plan_production_search(path)
        cache = waterbirds_fake_cache(plan, artifacts[1])
        monkeypatch.setattr("grit.search.waterbirds._pair_manifest", _forbid_access)
        monkeypatch.setattr(
            "grit.search.waterbirds.waterbirds_oracle_pair_features", _forbid_access
        )
        loader = "grit.search.waterbirds._load_cache"

    def load_control_cache(_plan: SearchPlan, *, tuning_only: bool = False):
        assert tuning_only
        return cache

    monkeypatch.setattr(loader, load_control_cache)
    monkeypatch.setattr(type(cache), "issue_final_handle", _forbid_access)
    monkeypatch.setattr(type(cache), "open_test_oracle_table", _forbid_access)
    status = run_production_search(
        path,
        ProductionExecutionLimits(stop_after="tuning", max_new_runs=1),
    )
    assert isinstance(status, ProductionSearchStatus)
    assert status.tuning_complete == 1
    assert status.final_complete == 0
