"""Fish, LISA, SWAD, and MatchDG: objectives, determinism, planning, and lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch
from pydantic import ValidationError

from grit.config import (
    DisabledPairsConfig,
    DisabledProjectionConfig,
    LinearProbeTrainingConfig,
    MatchDgAlgorithmConfig,
    SwadAlgorithmConfig,
)
from grit.features.cmnist import FeatureTable
from grit.methods.baselines import (
    FishLinearProbeMethod,
    LisaLinearProbeMethod,
    MatchDgLinearProbeMethod,
    SwadLinearProbeMethod,
)
from grit.methods.groupdro import cmnist_group_ids
from grit.methods.lisa import beta_2_2, plan_lisa_epoch, soft_cross_entropy
from grit.methods.matchdg import MatchDgAlgorithm
from grit.methods.swad import LossValley, SwadSegment
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeState,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    TrainedLinearProbeRun,
    train_linear_probe,
)
from grit.schemas import CmnistSelector, SeedStage
from grit.search.cmnist import materialize_cmnist_candidate_config
from grit.search.outputs import CmnistProductionSummary
from grit.search.plan import (
    APPROVED_FISH_META_STEP_SIZES,
    APPROVED_LISA_SELECTION_PROBS,
    APPROVED_MATCHDG_LATENT_DIMS,
    APPROVED_MATCHDG_PENALTY_WEIGHTS,
    APPROVED_SWAD_TOLERANCE_RATIOS,
    SWAD_SEGMENT_UPDATES,
    SearchPlan,
)
from grit.search.run import (
    pilot_candidates,
    plan_production_search,
    production_search_status,
    run_production_search,
)
from tests.contract_fixtures import (
    dataset_config,
    feature_config,
    runtime_config,
    seed_sets,
    training_config,
)
from tests.test_search_plan import write_cmnist_production_config
from tests.test_test_oracle_track import fake_cache


def _table(name: str, role: str, rows: int, seed: int) -> FeatureTable:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn((rows, 512), generator=generator)
    targets = (features[:, 0] > 0).to(torch.int64)
    colors = torch.arange(rows) % 2
    return FeatureTable(
        name=name,
        role=role,  # pyright: ignore[reportArgumentType]
        source_ids=tuple(f"source:{name}:{index}" for index in range(rows)),
        features=features,
        digits=targets * 5,
        clean_labels=targets,
        targets=targets,
        colors=colors,
    )


def _tables() -> tuple[
    tuple[FeatureTable, FeatureTable],
    tuple[FeatureTable, FeatureTable, FeatureTable],
]:
    return (
        (
            _table("train_e01", "training", 16, 1),
            _table("train_e02", "training", 16, 2),
        ),
        (
            _table("val_e01", "validation", 8, 3),
            _table("val_e02", "validation", 8, 4),
            _table("val_e05", "validation", 8, 5),
        ),
    )


def _config(batch_size: int = 8, epochs: int = 3) -> LinearProbeTrainingConfig:
    return LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=batch_size,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=epochs,
    )


def _environment_ids() -> torch.Tensor:
    return torch.cat(
        (torch.zeros(16, dtype=torch.int64), torch.ones(16, dtype=torch.int64))
    )


def _train(
    method: LinearProbeTrainingMethod,
    run_id: str,
    config: LinearProbeTrainingConfig | None = None,
) -> TrainedLinearProbeRun:
    training, validation = _tables()
    return train_linear_probe(
        training,
        validation,
        config or _config(),
        run_id=run_id,
        candidate_id=f"candidate:{method.method_id}",
        scientific_config_digest=f"sha256:{method.method_id}",
        seed_stage=SeedStage.TUNING,
        seed=101,
        method=method,
    )


def _assert_deterministic(make_method: Callable[[], LinearProbeTrainingMethod]) -> None:
    first = _train(make_method(), "run:first")
    second = _train(make_method(), "run:second")
    assert first.epoch_losses == second.epoch_losses
    assert torch.equal(
        first.algorithm.capture_inference_state().weight,
        second.algorithm.capture_inference_state().weight,
    )
    assert len(first.validation_metrics) == 3 * 3


def test_fish_interpolates_between_start_and_inner_loop_result() -> None:
    training, _ = _tables()
    features = torch.cat([table.features for table in training])
    targets = torch.cat([table.targets for table in training])
    rows = torch.arange(32)

    def inner_result(meta_step: float) -> tuple[LinearProbeState, LinearProbeState]:
        algorithm = LinearProbeAlgorithm(_config(), model_seed=7, projection=None)
        start = algorithm.capture_inference_state()
        method = FishLinearProbeMethod(
            environment_ids=_environment_ids(),
            environment_count=2,
            meta_step_size=meta_step,
        )
        method.update(algorithm, features, targets, rows)
        return start, algorithm.capture_inference_state()

    start, full_step = inner_result(1.0)
    reference = LinearProbeAlgorithm(_config(), model_seed=7, projection=None)
    reference.update(features[:16], targets[:16])
    reference.update(features[16:], targets[16:])
    assert torch.allclose(
        full_step.weight, reference.capture_inference_state().weight, atol=1e-7
    )
    _, half_step = inner_result(0.5)
    assert torch.allclose(
        half_step.weight - start.weight, 0.5 * (full_step.weight - start.weight)
    )
    _assert_deterministic(
        lambda: FishLinearProbeMethod(
            environment_ids=_environment_ids(), environment_count=2, meta_step_size=0.1
        )
    )


def test_lisa_batches_are_single_group_with_the_right_partners() -> None:
    training, _ = _tables()
    targets = torch.cat([table.targets for table in training])
    colors = torch.cat([table.colors for table in training])
    groups = cmnist_group_ids(targets, colors)
    generator = torch.Generator(device="cpu").manual_seed(3)
    plans = plan_lisa_epoch(
        groups, group_count=4, batch_size=4, selection_prob=0.5, generator=generator
    )
    assert len(plans) == 32 // 4
    seen_intra = {plan.intra_label for plan in plans}
    assert seen_intra == {True, False}
    for plan in plans:
        anchor_groups = {int(value) for value in groups[plan.rows]}
        assert len(anchor_groups) == 1
        anchor = anchor_groups.pop()
        partner_groups = {int(value) for value in groups[plan.partner_rows]}
        assert len(partner_groups) == 1
        target, color = divmod(anchor, 2)
        expected = (
            target * 2 + (1 - color) if plan.intra_label else (1 - target) * 2 + color
        )
        assert partner_groups.pop() == expected
        assert torch.all((plan.mixing_weights > 0) & (plan.mixing_weights < 1))
    repeated = plan_lisa_epoch(
        groups,
        group_count=4,
        batch_size=4,
        selection_prob=0.5,
        generator=torch.Generator(device="cpu").manual_seed(3),
    )
    assert all(
        torch.equal(a.rows, b.rows) for a, b in zip(plans, repeated, strict=True)
    )
    weights = beta_2_2(20_000, torch.Generator(device="cpu").manual_seed(0))
    assert abs(float(weights.mean()) - 0.5) < 0.01
    assert abs(float(weights.var()) - 0.05) < 0.005, "Beta(2,2) variance is 1/20"
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.25]])
    hard = torch.tensor([0, 1])
    assert torch.isclose(
        soft_cross_entropy(logits, torch.nn.functional.one_hot(hard, 2).float()),
        torch.nn.functional.cross_entropy(logits, hard),
    )
    _assert_deterministic(
        lambda: LisaLinearProbeMethod(
            group_ids=groups, group_count=4, selection_prob=0.5
        )
    )


def _segment(index: int, loss: float, value: float) -> SwadSegment:
    return SwadSegment(
        state=LinearProbeState(
            weight=torch.full((2, 512), value), bias=torch.full((2,), value)
        ),
        start_update=index * 10,
        end_update=(index + 1) * 10,
        end_loss=loss,
    )


def test_swad_loss_valley_starts_after_the_minimum_and_ends_on_tolerance() -> None:
    valley = LossValley(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    live = LinearProbeState(weight=torch.zeros((2, 512)), bias=torch.zeros(2))
    losses = [1.0, 0.9, 0.8, 0.85, 0.9]
    for index, loss in enumerate(losses):
        valley.observe(_segment(index, loss, float(index)))
    assert valley.converged and valley.converge_update == 3 * 10
    assert valley.threshold is not None
    assert abs(valley.threshold - (0.8 + 0.85 + 0.9) / 3 * 1.3) < 1e-12
    assert not valley.dead
    current = valley.current_state(live)
    assert float(current.bias[0]) != 0.0, "converged: checkpoints are averages"
    for index in range(5, 12):
        valley.observe(_segment(index, 5.0, float(index)))
    assert valley.dead
    final = valley.current_state(live)
    assert valley.final is not None and valley.final.count > 0
    assert torch.equal(final.weight, valley.final.state.weight)

    fresh = LossValley(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
    fresh.observe(_segment(0, 1.0, 1.0))
    fresh.observe(_segment(1, 0.5, 2.0))
    assert not fresh.converged
    assert torch.equal(fresh.current_state(live).weight, live.weight)


def test_swad_method_checkpoints_live_weights_until_the_valley_starts() -> None:
    _, validation = _tables()
    loss_features = torch.cat([validation[0].features, validation[1].features])
    loss_targets = torch.cat([validation[0].targets, validation[1].targets])

    def make(segment_updates: int) -> SwadLinearProbeMethod:
        return SwadLinearProbeMethod(
            loss_features=loss_features,
            loss_targets=loss_targets,
            tolerance_ratio=0.3,
            segment_updates=segment_updates,
        )

    run = _train(make(1000), "run:swad:long")
    assert not run.algorithm.capture_inference_state().weight.isnan().any()
    stored = run.store.load(run.validation_metrics[-1].checkpoint_id).state
    assert torch.equal(stored.weight, run.algorithm.capture_inference_state().weight)
    _assert_deterministic(lambda: make(2))
    short = _train(make(1), "run:swad:short", _config(batch_size=4, epochs=12))
    method = SwadLinearProbeMethod(
        loss_features=loss_features,
        loss_targets=loss_targets,
        tolerance_ratio=0.3,
        segment_updates=1,
    )
    assert short.method_id == "swad"
    assert not method.dead


def test_matchdg_composition_mirrors_factors_and_penalty_shrinks_pair_response() -> (
    None
):
    training, _ = _tables()
    features = torch.cat([table.features for table in training])
    targets = torch.cat([table.targets for table in training])
    differences = torch.randn((8, 512), generator=torch.Generator().manual_seed(11))

    def trained(penalty_weight: float) -> MatchDgAlgorithm:
        algorithm = MatchDgAlgorithm(
            _config(),
            model_seed=5,
            latent_dim=4,
            pair_differences=differences,
            penalty_weight=penalty_weight,
        )
        for _ in range(30):
            algorithm.update(features, targets)
        return algorithm

    weak = trained(1e-6)
    strong = trained(100.0)
    assert float(strong.pair_penalty().detach()) < float(weak.pair_penalty().detach())
    composed = weak.predict(features)
    factor_logits = weak._classifier(  # pyright: ignore[reportPrivateUsage]
        weak._featurizer(features)  # pyright: ignore[reportPrivateUsage]
    )
    assert torch.equal(composed, factor_logits.argmax(dim=1))
    state = weak.capture_inference_state()
    assert state.weight.shape == (2, 512)
    _assert_deterministic(
        lambda: MatchDgLinearProbeMethod(
            pair_differences=differences, latent_dim=4, penalty_weight=1.0
        )
    )
    with pytest.raises(ValueError, match="pairs.kind='oracle'"):
        _experiment_config(
            MatchDgAlgorithmConfig(
                kind="matchdg",
                latent_dim=4,
                penalty_weight=1.0,
                pair_penalty="mean_squared_featurizer_difference",
            )
        )


def _experiment_config(algorithm: MatchDgAlgorithmConfig) -> None:
    from grit.config import OrdinaryExperimentConfig, OrdinarySelectionConfig

    OrdinaryExperimentConfig(
        schema_version="grit.experiment/v1",
        run_kind="ordinary",
        experiment_name="matchdg-without-pairs",
        protocol_id="cmnist/v1",
        reportable=False,
        dataset=dataset_config(),
        representation=feature_config(),
        pairs=DisabledPairsConfig(kind="disabled"),
        projection=DisabledProjectionConfig(kind="disabled"),
        algorithm=algorithm,
        training=training_config(),
        runtime=runtime_config(),
        seed_sets=seed_sets(),
        selection=OrdinarySelectionConfig(selector=CmnistSelector.PRIMARY_ROBUST),
    )


@pytest.mark.parametrize(
    ("method", "space", "count"),
    [
        ("fish", {"fish_meta_step_sizes": list(APPROVED_FISH_META_STEP_SIZES)}, 64),
        ("lisa", {"lisa_selection_probs": list(APPROVED_LISA_SELECTION_PROBS)}, 112),
        ("swad", {"swad_tolerance_ratios": list(APPROVED_SWAD_TOLERANCE_RATIOS)}, 80),
        (
            "matchdg",
            {
                "matchdg_latent_dims": list(APPROVED_MATCHDG_LATENT_DIMS),
                "matchdg_penalty_weights": list(APPROVED_MATCHDG_PENALTY_WEIGHTS),
            },
            192,
        ),
    ],
)
def test_baseline_grids_plan_and_materialize(
    tmp_path: Path,
    method: Literal["fish", "lisa", "swad", "matchdg"],
    space: dict[str, object],
    count: int,
) -> None:
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={
            "experiment_name": f"cmnist-{method}",
            "search_space": {
                "methods": [method],
                "learning_rates": [0.0001, 0.0003, 0.001, 0.003],
                "weight_decays": [0.0, 0.00001, 0.0001, 0.001],
                **space,
            },
        },
    )
    plan = plan_production_search(config_path)
    assert plan.methods == (method,)
    assert len(plan.candidates) == count
    assert plan.expected_run_counts.tuning == count * 3
    assert pilot_candidates(plan) == (plan.candidates[0],)
    resolved = materialize_cmnist_candidate_config(
        plan, plan.candidates[0], CmnistSelector.PRIMARY_ROBUST
    )
    assert resolved.algorithm.kind == method
    assert resolved.artifact_lineage is not None
    assert (resolved.artifact_lineage.pair_manifest_digest is not None) == (
        method == "matchdg"
    )
    assert resolved.scientific_config_digest() == (
        plan.candidates[0].scientific_config_digest
    )
    if method == "swad":
        assert isinstance(resolved.algorithm, SwadAlgorithmConfig)
        assert resolved.algorithm.segment_updates == SWAD_SEGMENT_UPDATES
    with pytest.raises(ValidationError, match="exactly when"):
        write_and_plan = write_cmnist_production_config(
            tmp_path / "wrong",
            output_root=tmp_path / "wrong-output",
            overrides={
                "search_space": {
                    "methods": ["erm"],
                    "learning_rates": [0.001],
                    "weight_decays": [0.0, 0.0001, 0.001],
                    **space,
                }
            },
        )
        plan_production_search(write_and_plan[0])


def test_all_baselines_run_the_full_lifecycle_on_fake_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=output_root,
        overrides={
            "experiment_name": "cmnist-baselines-check",
            "search_space": {
                "methods": ["fish", "lisa", "swad", "matchdg"],
                "learning_rates": [0.01],
                "weight_decays": [0.0, 0.0001, 0.001],
                "fish_meta_step_sizes": [0.5],
                "lisa_selection_probs": [0.5],
                "swad_tolerance_ratios": [0.3],
                "matchdg_latent_dims": [4],
                "matchdg_penalty_weights": [1.0],
            },
            "batch_size": 8,
            "max_epochs": 2,
        },
    )
    plan = plan_production_search(config_path)
    cache = fake_cache(plan)

    def fake_cache_loader(_plan: SearchPlan, *, tuning_only: bool = False):
        return cache

    monkeypatch.setattr("grit.search.cmnist._load_cmnist_cache", fake_cache_loader)
    summary = run_production_search(config_path)
    assert isinstance(summary, CmnistProductionSummary)
    assert [m.method_id for m in summary.methods] == [
        method
        for method in ("fish", "lisa", "swad", "matchdg")
        for _ in ("primary_robust", "secondary_source")
    ]
    assert summary.paired_selectors == ()
    assert production_search_status(config_path).phase == "complete"
    assert len(tuple(output_root.rglob("final-result.json"))) == 4 * 2 * 10


def test_ordinary_probe_defaults_are_unchanged_by_the_protocol_hooks() -> None:
    method = OrdinaryLinearProbeMethod("erm", None, None)
    algorithm = method.build_algorithm(_config(), model_seed=1, num_classes=2)
    assert isinstance(algorithm, LinearProbeAlgorithm)
    assert torch.equal(
        method.checkpoint_state(algorithm).weight,
        algorithm.capture_inference_state().weight,
    )
