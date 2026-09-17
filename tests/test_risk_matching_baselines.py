"""SD, Fishr, and RDM: objectives, warm-up, planning, and the CMNIST lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.nn.functional as functional
from pydantic import ValidationError

from grit.config import (
    FishrAlgorithmConfig,
    LinearProbeTrainingConfig,
    RdmAlgorithmConfig,
    SdAlgorithmConfig,
)
from grit.methods.baselines import (
    FishrLinearProbeMethod,
    RdmLinearProbeMethod,
    SpectralDecouplingLinearProbeMethod,
)
from grit.methods.fishr import (
    FishrGradientVarianceEma,
    fishr_example_gradients,
    fishr_penalty,
)
from grit.methods.invariance import spectral_decoupling_objective
from grit.methods.rdm import RDM_KERNEL_GAMMAS, gaussian_mmd, rdm_objective
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    TrainedLinearProbeRun,
    train_linear_probe,
)
from grit.methods.types import CMNIST_ONLY_METHODS
from grit.schemas import CmnistSelector, SeedStage
from grit.search.cmnist import materialize_cmnist_candidate_config
from grit.search.outputs import CmnistProductionSummary
from grit.search.plan import (
    APPROVED_FISHR_PENALTY_WEIGHTS,
    APPROVED_RDM_PENALTY_WEIGHTS,
    APPROVED_SD_PENALTY_WEIGHTS,
    FISHR_EMA,
    FISHR_PENALTY_ANNEAL_UPDATES,
    RDM_PENALTY_ANNEAL_UPDATES,
    RDM_VARIANCE_WEIGHT,
    SearchPlan,
    load_production_search_config,
)
from grit.search.run import (
    plan_production_search,
    production_search_status,
    run_production_search,
)
from tests.test_baselines import (
    _config,  # pyright: ignore[reportPrivateUsage]
    _environment_ids,  # pyright: ignore[reportPrivateUsage]
    _tables,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_search_plan import write_cmnist_production_config
from tests.test_test_oracle_track import fake_cache
from tests.test_waterbirds_paper_table import write_waterbirds_production_config

FISHR_EMA_DECAY = 0.95


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


def _probe(features: torch.Tensor, seed: int = 5) -> torch.nn.Linear:
    model = torch.nn.Linear(int(features.shape[1]), 2, bias=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=generator) * 0.1)
        model.bias.copy_(torch.randn(model.bias.shape, generator=generator) * 0.1)
    return model


def _autograd_example_gradients(
    model: torch.nn.Linear, features: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Per-example classifier gradients from autograd, one backward call per row."""

    rows: list[torch.Tensor] = []
    for index in range(int(features.shape[0])):
        loss = functional.cross_entropy(
            model(features[index : index + 1]), targets[index : index + 1]
        )
        weight_grad, bias_grad = torch.autograd.grad(
            loss, [model.weight, model.bias], create_graph=True
        )
        rows.append(torch.cat([weight_grad.flatten(), bias_grad.flatten()]))
    return torch.stack(rows)


def test_fishr_analytic_gradients_match_autograd() -> None:
    generator = torch.Generator(device="cpu").manual_seed(11)
    features = torch.randn((8, 6), generator=generator)
    targets = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    model = _probe(features)
    analytic = fishr_example_gradients(model(features), targets, features)
    expected = _autograd_example_gradients(model, features, targets)
    assert analytic.shape == (8, 2 * 6 + 2)
    assert torch.allclose(analytic, expected, atol=1e-6)


def test_fishr_penalty_and_its_derivative_match_an_autograd_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(13)
    features = torch.randn((8, 6), generator=generator)
    targets = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    environment_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    def reference_penalty(model: torch.nn.Linear, state: list[torch.Tensor]) -> (
        torch.Tensor
    ):
        gradients = _autograd_example_gradients(model, features, targets)
        corrected: list[torch.Tensor] = []
        for environment in (0, 1):
            rows = gradients[environment_ids == environment]
            variance = (rows - rows.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)
            averaged = FISHR_EMA_DECAY * state[environment] + (
                1.0 - FISHR_EMA_DECAY
            ) * variance
            state[environment] = averaged.detach().clone()
            corrected.append(averaged / (1.0 - FISHR_EMA_DECAY))
        mean_variance = torch.stack(corrected, dim=0).mean(dim=0)
        return torch.stack(
            [(value - mean_variance).pow(2).mean() for value in corrected]
        ).mean()

    model = _probe(features)
    ema = FishrGradientVarianceEma(environment_count=2, decay=FISHR_EMA_DECAY)
    reference_model = _probe(features)
    reference_state = [torch.zeros(2 * 6 + 2), torch.zeros(2 * 6 + 2)]
    # Two updates, so the second one exercises a non-zero historical average.
    for _ in range(2):
        penalty = fishr_penalty(
            model(features),
            targets,
            features,
            environment_ids,
            environment_count=2,
            ema=ema,
        )
        expected = reference_penalty(reference_model, reference_state)
        assert torch.isclose(penalty, expected, atol=1e-9)
        gradient = torch.autograd.grad(penalty, model.weight)[0]
        expected_gradient = torch.autograd.grad(expected, reference_model.weight)[0]
        assert torch.allclose(gradient, expected_gradient, atol=1e-7)


def test_fishr_ema_applies_the_one_minus_ema_correction_and_detaches_history() -> None:
    ema = FishrGradientVarianceEma(environment_count=2, decay=0.95)
    first = torch.tensor([2.0, 4.0], requires_grad=True)
    corrected = ema.update(0, first)
    # 0.95 * 0 + 0.05 * data, divided by 0.05, is the data itself on the first update.
    assert torch.allclose(corrected, first.detach())
    history = ema.state(0)
    assert history is not None
    assert torch.allclose(history, 0.05 * first.detach())
    assert not history.requires_grad
    second = torch.tensor([0.0, 0.0], requires_grad=True)
    assert torch.allclose(ema.update(0, second), 0.95 * first.detach())
    assert ema.state(1) is None, "environments hold independent state"


def test_spectral_decoupling_penalizes_raw_logits_and_reduces_to_erm() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.25]], requires_grad=True)
    targets = torch.tensor([0, 1])
    cross_entropy = functional.cross_entropy(logits, targets)
    expected = cross_entropy + 0.1 * (4.0 + 1.0 + 0.25 + 0.0625) / 4
    assert torch.isclose(
        spectral_decoupling_objective(logits, targets, penalty_weight=0.1), expected
    )
    assert torch.isclose(
        spectral_decoupling_objective(logits, targets, penalty_weight=0.0),
        cross_entropy,
    )
    # A zero coefficient is a numerical control: SD must then be plain ERM.
    erm = _train(OrdinaryLinearProbeMethod("erm", None, None), "run:erm")
    sd = _train(SpectralDecouplingLinearProbeMethod(penalty_weight=0.0), "run:sd")
    assert torch.allclose(
        erm.algorithm.capture_inference_state().weight,
        sd.algorithm.capture_inference_state().weight,
    )
    # The reductions differ only in summation order, so this is exact to float noise.
    assert all(
        abs(left - right) < 1e-8
        for left, right in zip(erm.epoch_losses, sd.epoch_losses, strict=True)
    )


def test_rdm_mmd_matches_the_reference_and_ties_pick_the_lowest_environment() -> None:
    left = torch.tensor([[0.1], [0.4], [0.9]])
    right = torch.tensor([[0.2], [0.3]])

    def reference_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distances = (x - y.T).pow(2).clamp_min(1e-30)
        kernel = torch.zeros_like(distances)
        for gamma in RDM_KERNEL_GAMMAS:
            kernel = kernel + torch.exp(-gamma * distances)
        return kernel

    expected = (
        reference_kernel(left, left).mean()
        + reference_kernel(right, right).mean()
        - 2 * reference_kernel(left, right).mean()
    )
    assert torch.isclose(gaussian_mmd(left, right), expected, atol=1e-9)
    assert torch.isclose(gaussian_mmd(left, left), torch.zeros(()), atol=1e-9)

    # Tied environment risks with *different* spreads, so the two candidate worst
    # environments give measurably different objectives and the tie rule is tested
    # rather than assumed. Both environments have mean 0.5.
    losses = torch.tensor([0.5, 0.5, 0.1, 0.9])
    environment_ids = torch.tensor([0, 0, 1, 1])
    first, second = losses[:2], losses[2:]
    assert torch.isclose(first.mean(), second.mean()), "the risks must actually tie"

    def objective_for(worst: torch.Tensor) -> torch.Tensor:
        return (
            losses.mean()
            + 2.5 * gaussian_mmd(worst.unsqueeze(1), losses.unsqueeze(1))
            + 0.004 * (losses.var(correction=1) + worst.var(correction=1))
        )

    objective = rdm_objective(
        losses,
        environment_ids,
        environment_count=2,
        penalty_weight=2.5,
        variance_weight=0.004,
    )
    assert not torch.isclose(objective_for(first), objective_for(second), atol=1e-6), (
        "the two candidate worst environments must be distinguishable"
    )
    assert torch.isclose(objective, objective_for(first), atol=1e-9)
    # Reversing the IDs moves the low-variance environment to ID 1, so the tie now
    # resolves the other way and the objective follows it.
    assert torch.isclose(
        rdm_objective(
            losses,
            torch.tensor([1, 1, 0, 0]),
            environment_count=2,
            penalty_weight=2.5,
            variance_weight=0.004,
        ),
        objective_for(second),
        atol=1e-9,
    )
    # With both coefficients zero the objective is the mean environment risk.
    assert torch.isclose(
        rdm_objective(
            losses,
            environment_ids,
            environment_count=2,
            penalty_weight=0.0,
            variance_weight=0.0,
        ),
        losses.mean(),
    )
    with pytest.raises(ValueError, match="at least 2 examples per environment"):
        rdm_objective(
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0, 0, 1]),
            environment_count=2,
            penalty_weight=1.0,
            variance_weight=0.004,
        )


def test_a_warm_up_that_would_never_activate_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose warm-up outlasts training would report ERM under another name."""

    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        overrides={
            "experiment_name": "cmnist-fishr-warm-up-too-long",
            "search_space": {
                "methods": ["fishr"],
                "learning_rates": [0.01],
                "weight_decays": [0.0, 0.0001, 0.001],
                "fishr_penalty_weights": [100.0],
                "fishr_penalty_anneal_updates": FISHR_PENALTY_ANNEAL_UPDATES,
                "fishr_ema": FISHR_EMA,
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
    with pytest.raises(ValueError, match="leaves no penalized update"):
        run_production_search(config_path)


def test_rdm_rejects_an_epoch_whose_remainder_batch_is_too_small() -> None:
    method = RdmLinearProbeMethod(
        environment_ids=_environment_ids(),
        environment_count=2,
        penalty_weight=1.0,
        penalty_anneal_updates=0,
        variance_weight=RDM_VARIANCE_WEIGHT,
    )
    generator = torch.Generator(device="cpu").manual_seed(2)
    # 16 rows per environment in batches of 3 each leaves a one-row remainder.
    with pytest.raises(ValueError, match="at least 2 examples per environment"):
        method.epoch_batches(row_count=32, batch_size=6, generator=generator)
    assert method.epoch_batches(row_count=32, batch_size=8, generator=generator)


class _ResetCountingAlgorithm(LinearProbeAlgorithm):
    resets: int = 0

    def reset_optimizer(self) -> None:
        self.resets += 1
        super().reset_optimizer()


@pytest.mark.parametrize("warm_up", [0, 3])
def test_warm_up_runs_plain_cross_entropy_and_resets_adam_once(warm_up: int) -> None:
    training, _ = _tables()
    features = torch.cat([table.features for table in training])
    targets = torch.cat([table.targets for table in training])
    rows = torch.arange(32)

    def run(
        make_method: Callable[[float], LinearProbeTrainingMethod],
        penalty_weight: float,
    ) -> tuple[torch.Tensor, list[float], int]:
        algorithm = _ResetCountingAlgorithm(_config(), model_seed=7, projection=None)
        method = make_method(penalty_weight)
        losses = [
            method.update(algorithm, features, targets, rows)
            for _ in range(warm_up + 2)
        ]
        return algorithm.capture_inference_state().weight, losses, algorithm.resets

    def make_fishr(weight: float) -> LinearProbeTrainingMethod:
        return FishrLinearProbeMethod(
            environment_ids=_environment_ids(),
            environment_count=2,
            penalty_weight=weight,
            penalty_anneal_updates=warm_up,
            ema_decay=FISHR_EMA,
        )

    def make_rdm(weight: float) -> LinearProbeTrainingMethod:
        return RdmLinearProbeMethod(
            environment_ids=_environment_ids(),
            environment_count=2,
            penalty_weight=weight,
            penalty_anneal_updates=warm_up,
            variance_weight=RDM_VARIANCE_WEIGHT,
        )

    for make_method in (make_fishr, make_rdm):
        weak, weak_losses, resets = run(make_method, 1e-8)
        strong, strong_losses, _ = run(make_method, 1e6)
        assert resets == 1, "Adam restarts exactly once, when the penalty activates"
        assert weak_losses[:warm_up] == strong_losses[:warm_up], (
            "warm-up updates must not see the penalty weight"
        )
        assert weak_losses[warm_up] != strong_losses[warm_up], (
            "the first post-warm-up update must see an active penalty"
        )
        assert not torch.allclose(weak, strong)


def test_fishr_advances_its_ema_during_warm_up() -> None:
    training, _ = _tables()
    features = torch.cat([table.features for table in training])
    targets = torch.cat([table.targets for table in training])
    algorithm = LinearProbeAlgorithm(_config(), model_seed=7, projection=None)
    method = FishrLinearProbeMethod(
        environment_ids=_environment_ids(),
        environment_count=2,
        penalty_weight=100.0,
        penalty_anneal_updates=5,
        ema_decay=FISHR_EMA,
    )
    assert method.ema.state(0) is None
    method.update(algorithm, features, targets, torch.arange(32))
    warmed = method.ema.state(0)
    assert warmed is not None and bool((warmed != 0).any())
    # Evaluation is side-effect free: the EMA is training-only state and the saved
    # checkpoint is only the linear weight and bias.
    algorithm.predict(features)
    algorithm.mean_loss(features, targets)
    still = method.ema.state(0)
    assert still is not None and torch.equal(still, warmed)
    checkpoint = method.checkpoint_state(algorithm)
    live = algorithm.capture_inference_state()
    assert torch.equal(checkpoint.weight, live.weight)
    assert torch.equal(checkpoint.bias, live.bias)


@pytest.mark.parametrize(
    ("method", "space", "count"),
    [
        (
            "sd",
            {"sd_penalty_weights": list(APPROVED_SD_PENALTY_WEIGHTS)},
            16 * 5,
        ),
        (
            "fishr",
            {
                "fishr_penalty_weights": list(APPROVED_FISHR_PENALTY_WEIGHTS),
                "fishr_penalty_anneal_updates": FISHR_PENALTY_ANNEAL_UPDATES,
                "fishr_ema": FISHR_EMA,
            },
            16 * 4,
        ),
        (
            "rdm",
            {
                "rdm_penalty_weights": list(APPROVED_RDM_PENALTY_WEIGHTS),
                "rdm_penalty_anneal_updates": RDM_PENALTY_ANNEAL_UPDATES,
                "rdm_variance_weight": RDM_VARIANCE_WEIGHT,
            },
            16 * 4,
        ),
    ],
)
def test_new_baseline_grids_plan_and_materialize(
    tmp_path: Path,
    method: Literal["sd", "fishr", "rdm"],
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
    resolved = materialize_cmnist_candidate_config(
        plan, plan.candidates[0], CmnistSelector.PRIMARY_ROBUST
    )
    algorithm = resolved.algorithm
    assert algorithm.kind == method
    assert resolved.artifact_lineage is not None
    assert resolved.artifact_lineage.pair_manifest_digest is None, (
        "these methods receive no oracle pair identities"
    )
    assert resolved.pairs.kind == "disabled"
    assert resolved.projection.kind == "disabled"
    assert resolved.scientific_config_digest() == (
        plan.candidates[0].scientific_config_digest
    )
    if isinstance(algorithm, SdAlgorithmConfig):
        assert algorithm.penalty == "mean_squared_logits"
    if isinstance(algorithm, FishrAlgorithmConfig):
        assert algorithm.environment_names == ("train_e01", "train_e02")
        assert algorithm.ema == FISHR_EMA
        assert algorithm.penalty_anneal_updates == FISHR_PENALTY_ANNEAL_UPDATES
    if isinstance(algorithm, RdmAlgorithmConfig):
        assert algorithm.environment_names == ("train_e01", "train_e02")
        assert algorithm.variance_weight == RDM_VARIANCE_WEIGHT
        assert algorithm.penalty_anneal_updates == RDM_PENALTY_ANNEAL_UPDATES
    with pytest.raises(ValidationError, match="exactly when"):
        other = write_cmnist_production_config(
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
        plan_production_search(other[0])


def test_waterbirds_refuses_the_cmnist_only_methods(tmp_path: Path) -> None:
    config_path, _ = write_waterbirds_production_config(
        tmp_path,
        output_root=tmp_path / "output",
        search_space={
            "methods": ["erm", "sd"],
            "learning_rates": [0.01],
            "weight_decays": [0.0, 0.0001, 0.001],
            "sd_penalty_weights": [0.1],
        },
    )
    with pytest.raises(ValidationError, match="CMNIST only"):
        load_production_search_config(config_path)


def test_checked_configs_use_the_documented_grids_and_separate_output_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJECT_SCRATCH", "/scratch")
    root = Path(__file__).resolve().parents[1] / "configs/cmnist"
    expected = {
        "sd": (APPROVED_SD_PENALTY_WEIGHTS, "/scratch/outputs/cmnist-sd"),
        "fishr": (APPROVED_FISHR_PENALTY_WEIGHTS, "/scratch/outputs/cmnist-fishr"),
        "rdm": (APPROVED_RDM_PENALTY_WEIGHTS, "/scratch/outputs/cmnist-rdm"),
    }
    roots: set[str] = set()
    for method, (weights, output_root) in expected.items():
        config = load_production_search_config(root / f"{method}-search.yaml")
        space = config.search_space
        assert space.methods == (method,)
        assert getattr(space, f"{method}_penalty_weights") == weights
        assert config.output_root == output_root
        assert config.selectors == ("primary_robust", "secondary_source"), (
            "the initial configurations select on validation only"
        )
        roots.add(config.output_root)
    assert len(roots) == 3
    fishr = load_production_search_config(root / "fishr-search.yaml").search_space
    assert fishr.fishr_penalty_anneal_updates == FISHR_PENALTY_ANNEAL_UPDATES
    assert fishr.fishr_ema == FISHR_EMA
    rdm = load_production_search_config(root / "rdm-search.yaml").search_space
    assert rdm.rdm_penalty_anneal_updates == RDM_PENALTY_ANNEAL_UPDATES
    assert rdm.rdm_variance_weight == RDM_VARIANCE_WEIGHT
    assert not any(
        (root / f"{method}-search-test-oracle.yaml").exists()
        for method in CMNIST_ONLY_METHODS
    ), "no diagnostic search files are added with this change"


def test_new_baselines_run_the_cmnist_lifecycle_on_fake_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    config_path, _ = write_cmnist_production_config(
        tmp_path,
        output_root=output_root,
        overrides={
            "experiment_name": "cmnist-risk-matching-check",
            "search_space": {
                "methods": ["sd", "fishr", "rdm"],
                "learning_rates": [0.01],
                "weight_decays": [0.0, 0.0001, 0.001],
                "sd_penalty_weights": [0.1],
                "fishr_penalty_weights": [100.0],
                # A tiny run must cross its warm-up, so it is shortened explicitly.
                "fishr_penalty_anneal_updates": 2,
                "fishr_ema": FISHR_EMA,
                "rdm_penalty_weights": [1.0],
                "rdm_penalty_anneal_updates": 2,
                "rdm_variance_weight": RDM_VARIANCE_WEIGHT,
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
    assert [item.method_id for item in summary.methods] == [
        method
        for method in ("sd", "fishr", "rdm")
        for _ in ("primary_robust", "secondary_source")
    ]
    assert production_search_status(config_path).phase == "complete"
    assert len(tuple(output_root.rglob("final-result.json"))) == 3 * 2 * 10
    checkpoints = tuple(output_root.rglob("selected-checkpoint/manifest.json"))
    assert len(checkpoints) == 3 * 2 * 10
    store = PersistedLinearCheckpointStore(checkpoints[0].parent)
    restored = store.load(store.store_id.removeprefix("linear-checkpoint:"))
    assert restored.state.weight.shape == (2, 512)
