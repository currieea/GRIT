"""Objective composition, training controls, and complete predictor restoration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
import torch.nn.functional as functional

from grit.config import LinearProbeTrainingConfig
from grit.methods.baselines import FishrLinearProbeMethod
from grit.methods.checkpoints import StoredCheckpoint
from grit.methods.interventions import (
    ComposedLinearProbeMethod,
    prediction_consistency_penalty,
)
from grit.methods.invariance import irmv1_objective, vrex_objective
from grit.methods.projection import FittedLinearProjection, fit_linear_projection
from grit.methods.training import (
    IrmLinearProbeMethod,
    LinearProbeAlgorithm,
    LinearProbeState,
    LinearProbeTrainingMethod,
    OrdinaryLinearProbeMethod,
    PersistedLinearCheckpointStore,
    RexLinearProbeMethod,
    persist_selected_linear_checkpoint,
)
from grit.methods.types import MethodId
from grit.selection.cmnist import CheckpointIdentity

Base = Literal["erm", "rex", "irm", "fishr"]


def _config() -> LinearProbeTrainingConfig:
    return LinearProbeTrainingConfig(
        optimizer="adam",
        batch_size=8,
        learning_rate=0.001,
        weight_decay=0.0,
        max_epochs=2,
    )


def _data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn((8, 512), generator=generator),
        torch.tensor([0, 1, 1, 0, 1, 0, 1, 0]),
        torch.randn((3, 512), generator=generator),
    )


def _base(kind: Base) -> LinearProbeTrainingMethod:
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    if kind == "erm":
        return OrdinaryLinearProbeMethod("erm", None, None)
    if kind == "rex":
        return RexLinearProbeMethod(ids, 2, 10.0, 1)
    if kind == "irm":
        return IrmLinearProbeMethod(ids, 2, 10.0, 1)
    return FishrLinearProbeMethod(ids, 2, 10.0, 1, 0.95)


def _projection(rank: int) -> FittedLinearProjection:
    differences = torch.zeros((2, 512))
    differences[0, 0] = 1.0
    differences[1, 1] = 1.0
    return fit_linear_projection(
        differences,
        torch.zeros_like(differences),
        requested_rank=rank,
        pair_manifest_digest="sha256:pairs",
        feature_cache_manifest_digest="sha256:cache",
    )


def test_consistency_matches_paired_logits_and_gradient() -> None:
    generator = torch.Generator().manual_seed(4)
    left = torch.randn((5, 7), generator=generator, dtype=torch.float64)
    right = torch.randn((5, 7), generator=generator, dtype=torch.float64)
    weight = torch.randn(
        (3, 7), generator=generator, dtype=torch.float64
    ).requires_grad_()
    bias = torch.randn((3,), generator=generator, dtype=torch.float64).requires_grad_()
    differences = left - right
    penalty = prediction_consistency_penalty(weight, differences)
    explicit = (
        (functional.linear(left, weight, bias) - functional.linear(right, weight, bias))
        .square()
        .sum(dim=1)
        .mean()
    )
    torch.testing.assert_close(penalty, explicit)
    gradient = torch.autograd.grad(penalty, weight)[0]
    explicit_weight, explicit_bias = torch.autograd.grad(explicit, (weight, bias))
    torch.testing.assert_close(gradient, explicit_weight)
    torch.testing.assert_close(gradient, 2.0 * weight @ differences.T @ differences / 5)
    torch.testing.assert_close(explicit_bias, torch.zeros_like(bias))


@pytest.mark.parametrize("kind", ["erm", "rex", "irm", "fishr"])
@pytest.mark.parametrize("control", ["identity", "zero_consistency"])
def test_control_interventions_reproduce_base_updates(kind: Base, control: str) -> None:
    features, targets, differences = _data()
    plain = _base(kind)
    if control == "identity":
        method_id = "grit" if kind == "erm" else f"{kind}_grit"
        composed = ComposedLinearProbeMethod(
            _base(kind),
            cast(MethodId, method_id),
            projection=_projection(0),
            projection_rank=0,
        )
    else:
        composed = ComposedLinearProbeMethod(
            _base(kind),
            cast(MethodId, f"{kind}_consistency"),
            pair_differences=differences,
            consistency_weight=0.0,
        )
    vanilla = plain.build_algorithm(_config(), model_seed=3, num_classes=2)
    intervened = composed.build_algorithm(_config(), model_seed=3, num_classes=2)
    # Includes the annealing transition, Fishr's optimizer reset, and EMA history.
    for _ in range(3):
        rows = torch.arange(8)
        assert plain.update(vanilla, features, targets, rows) == composed.update(
            intervened, features, targets, rows
        )
    expected = vanilla.capture_inference_state()
    actual = intervened.capture_inference_state()
    assert torch.equal(expected.weight, actual.weight)
    assert torch.equal(expected.bias, actual.bias)


@pytest.mark.parametrize("kind", ["rex", "irm"])
def test_rescaled_objective_scales_complete_consistency_term(kind: Base) -> None:
    features, targets, differences = _data()
    method = ComposedLinearProbeMethod(
        _base(kind),
        cast(MethodId, f"{kind}_consistency"),
        pair_differences=differences,
        consistency_weight=2.0,
    )
    algorithm = method.build_algorithm(_config(), model_seed=5, num_classes=2)
    for expected_weight in (1.0, 10.0):
        state = algorithm.capture_inference_state()
        logits = functional.linear(features, state.weight, state.bias)
        ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        if kind == "rex":
            expected = vrex_objective(
                functional.cross_entropy(logits, targets, reduction="none"),
                ids,
                environment_count=2,
                penalty_weight=expected_weight,
            )
        else:
            expected = irmv1_objective(
                logits,
                targets,
                ids,
                environment_count=2,
                penalty_weight=expected_weight,
            )
        expected += (
            2.0
            / expected_weight
            * prediction_consistency_penalty(state.weight, differences)
        )
        actual = method.update(algorithm, features, targets, torch.arange(8))
        assert math.isclose(actual, float(expected.item()), rel_tol=1e-6)


def test_fishr_projection_uses_projected_supervised_gradient_features() -> None:
    features, targets, _ = _data()
    projection = _projection(2)
    composed = ComposedLinearProbeMethod(_base("fishr"), "fishr_grit", projection, 2)
    projected = composed.build_algorithm(_config(), model_seed=4, num_classes=2)
    plain_method = _base("fishr")
    plain = plain_method.build_algorithm(_config(), model_seed=4, num_classes=2)
    for _ in range(3):
        assert composed.update(projected, features, targets, torch.arange(8)) == (
            plain_method.update(
                plain, projection.transform(features), targets, torch.arange(8)
            )
        )
    assert torch.equal(
        projected.capture_inference_state().weight,
        plain.capture_inference_state().weight,
    )


def test_fishr_consistency_is_active_in_warmup_and_excluded_from_statistics() -> None:
    features, targets, differences = _data()
    base = cast(FishrLinearProbeMethod, _base("fishr"))
    plain_method = cast(FishrLinearProbeMethod, _base("fishr"))
    composed = ComposedLinearProbeMethod(
        base,
        "fishr_consistency",
        pair_differences=differences,
        consistency_weight=2.0,
    )
    algorithm = composed.build_algorithm(_config(), model_seed=4, num_classes=2)
    vanilla = plain_method.build_algorithm(_config(), model_seed=4, num_classes=2)
    penalty = prediction_consistency_penalty(
        algorithm.capture_inference_state().weight, differences
    )
    ordinary_loss = plain_method.update(vanilla, features, targets, torch.arange(8))
    combined = composed.update(algorithm, features, targets, torch.arange(8))
    assert math.isclose(
        combined, ordinary_loss + 2.0 * float(penalty.item()), rel_tol=1e-6
    )
    for environment in range(2):
        actual, expected = (
            base.ema.state(environment),
            plain_method.ema.state(environment),
        )
        assert actual is not None and expected is not None
        assert torch.equal(actual, expected)


def test_persisted_predictor_restores_projection_without_external_fit(
    tmp_path: Path,
) -> None:
    features, targets, _ = _data()
    method = ComposedLinearProbeMethod(_base("erm"), "grit", _projection(2), 2)
    algorithm = method.build_algorithm(_config(), model_seed=4, num_classes=2)
    method.update(algorithm, features, targets, torch.arange(8))
    identity = CheckpointIdentity(
        checkpoint_id="checkpoint:projected",
        candidate_id="candidate:projected",
        run_id="run:projected",
        scientific_config_digest="sha256:config",
        epoch=1,
    )
    root = tmp_path / "checkpoint"
    persist_selected_linear_checkpoint(
        StoredCheckpoint(identity, method.checkpoint_state(algorithm)), root
    )
    restored = LinearProbeAlgorithm(_config(), model_seed=999, projection=None)
    restored.restore_inference_state(
        PersistedLinearCheckpointStore(root).load(identity.checkpoint_id).state
    )
    assert torch.equal(algorithm.predict(features), restored.predict(features))
    assert algorithm.mean_loss(features, targets) == restored.mean_loss(
        features, targets
    )
    perturbed = features.clone()
    perturbed[:, :2] += 100.0
    assert torch.equal(restored.predict(features), restored.predict(perturbed))
    assert restored.mean_loss(features, targets) == restored.mean_loss(
        perturbed, targets
    )


def test_modern_vanilla_state_clears_projection_and_legacy_state_retains_it() -> None:
    features, targets, _ = _data()
    vanilla = LinearProbeAlgorithm(_config(), model_seed=4, projection=None)
    projected = LinearProbeAlgorithm(_config(), model_seed=9, projection=_projection(2))
    state = vanilla.capture_inference_state()
    projected.restore_inference_state(LinearProbeState(state.weight, state.bias))
    assert torch.equal(
        projected.prepare_features(features), _projection(2).transform(features)
    )
    projected.restore_inference_state(state)
    assert torch.equal(projected.prepare_features(features), features)
    assert projected.mean_loss(features, targets) == vanilla.mean_loss(
        features, targets
    )
