"""Factorized objectives: mathematical gradients, controls and training restoration."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from math import isclose
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn.functional as functional

from grit.methods.baselines import FishrLinearProbeMethod
from grit.methods.checkpoints import StoredCheckpoint
from grit.methods.fishr import (
    FishrGradientVarianceEma,
    fishr_example_gradients,
    fishr_penalty,
)
from grit.methods.interventions import ComposedLinearProbeMethod
from grit.methods.invariance import irmv1_objective, vrex_objective
from grit.methods.matchdg import FactorizedLinearProbeAlgorithm, MatchDgAlgorithm
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeState,
    PersistedLinearCheckpointStore,
    persist_selected_linear_checkpoint,
)
from grit.methods.types import MethodId, consumes_pairs, pair_intervention
from grit.selection.cmnist import CheckpointIdentity
from tests.test_pair_interventions import Base, _base, _config, _data


def _method(kind: Base, strength: float, *, control: bool = False):
    return ComposedLinearProbeMethod(
        _base(kind),
        cast(
            MethodId,
            f"{kind}_two_layer" if control else f"{kind}_representation_consistency",
        ),
        pair_differences=None if control else _data()[2],
        representation_consistency_weight=strength,
        latent_dim=4,
    )


def _algorithm(kind: Base = "erm", strength: float = 2.0):
    method = _method(kind, strength)
    algorithm = method.build_algorithm(_config(), model_seed=5, num_classes=2)
    assert isinstance(algorithm, FactorizedLinearProbeAlgorithm)
    return method, algorithm


def test_representation_penalty_matches_explicit_pairs_and_gradients() -> None:
    _, algorithm = _algorithm()
    differences = _data()[2]
    left = _data()[0][:3]
    right = left - differences
    layer = algorithm._featurizer
    actual = algorithm.pair_penalty()
    explicit = (layer(left) - layer(right)).square().sum(dim=1).mean()
    torch.testing.assert_close(actual, explicit)
    actual_weight, actual_bias = torch.autograd.grad(
        actual, (layer.weight, layer.bias), allow_unused=True
    )
    expected_weight, expected_bias = torch.autograd.grad(
        explicit, (layer.weight, layer.bias)
    )
    torch.testing.assert_close(actual_weight, expected_weight)
    torch.testing.assert_close(
        actual_weight, 2 * layer.weight @ differences.T @ differences / 3
    )
    assert actual_bias is None
    torch.testing.assert_close(expected_bias, torch.zeros_like(layer.bias))


@pytest.mark.parametrize("kind", ["erm", "rex", "irm", "fishr"])
def test_zero_strength_matches_pair_free_control_through_warmup(kind: Base) -> None:
    method, algorithm = _algorithm(kind, 0.0)
    control = _method(kind, 0.0, control=True)
    plain = control.build_algorithm(_config(), model_seed=5, num_classes=2)
    assert isinstance(plain, FactorizedLinearProbeAlgorithm)
    assert plain._pair_differences is None
    features, targets, _ = _data()
    for _ in range(4):
        assert method.update(algorithm, features, targets, torch.arange(8)) == (
            control.update(plain, features, targets, torch.arange(8))
        )
        expected = plain.capture_inference_state().factor_parameters
        actual = algorithm.capture_inference_state().factor_parameters
        assert expected is not None and actual is not None
        assert all(torch.equal(actual[name], value) for name, value in expected.items())
    assert not consumes_pairs(control.method_id)
    assert consumes_pairs(method.method_id)
    assert pair_intervention(method.method_id) == "representation_consistency"


@pytest.mark.parametrize("kind", ["rex", "irm"])
@pytest.mark.parametrize("strength", [0.0, 2.0])
def test_representation_complete_objective_rescaling(
    kind: Base, strength: float
) -> None:
    method, algorithm = _algorithm(kind, strength)
    features, targets, _ = _data()
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    for alpha in (1.0, 10.0, 10.0):
        logits = algorithm._classifier(algorithm.prepare_features(features))
        if kind == "rex":
            base = vrex_objective(
                functional.cross_entropy(logits, targets, reduction="none"),
                ids,
                environment_count=2,
                penalty_weight=alpha,
            )
        else:
            base = irmv1_objective(
                logits, targets, ids, environment_count=2, penalty_weight=alpha
            )
        expected = base + strength * algorithm.pair_penalty() / max(1.0, alpha)
        actual = method.update(algorithm, features, targets, torch.arange(8))
        assert isclose(actual, float(expected.detach()), rel_tol=1e-6)


def test_factorized_fishr_statistics_match_autograd_and_train_representation() -> None:
    _, algorithm = _algorithm("fishr")
    features, targets, _ = _data()
    representation = algorithm.prepare_features(features)
    logits = algorithm._classifier(representation)
    analytic = fishr_example_gradients(logits, targets, representation)
    gradients: list[torch.Tensor] = []
    for loss in functional.cross_entropy(logits, targets, reduction="none"):
        weight, bias = torch.autograd.grad(
            loss,
            (algorithm._classifier.weight, algorithm._classifier.bias),
            create_graph=True,
            retain_graph=True,
        )
        gradients.append(torch.cat((weight.flatten(), bias)))
    torch.testing.assert_close(analytic, torch.stack(gradients))
    assert analytic.shape == (8, 2 * (4 + 1))
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    penalty = fishr_penalty(
        logits,
        targets,
        representation,
        ids,
        environment_count=2,
        ema=FishrGradientVarianceEma(environment_count=2, decay=0.95),
    )
    expected_variances = torch.stack(
        [
            torch.stack(gradients)[ids == environment].var(dim=0, unbiased=False)
            for environment in range(2)
        ]
    )
    expected = (expected_variances - expected_variances.mean(dim=0)).square().mean()
    torch.testing.assert_close(penalty, expected)
    a_gradient = torch.autograd.grad(
        penalty, algorithm._featurizer.weight, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(expected, algorithm._featurizer.weight)[0]
    torch.testing.assert_close(a_gradient, expected_gradient)
    assert float(a_gradient.abs().sum()) > 0.0


def test_fishr_pair_penalty_separate_ema_and_reset_updates_both_layers() -> None:
    method, algorithm = _algorithm("fishr")
    control = _method("fishr", 0.0, control=True)
    plain = control.build_algorithm(_config(), model_seed=5, num_classes=2)
    features, targets, _ = _data()
    pair_penalty = float(algorithm.pair_penalty().detach())
    plain_loss = control.update(plain, features, targets, torch.arange(8))
    loss = method.update(algorithm, features, targets, torch.arange(8))
    assert isclose(loss, plain_loss + 2.0 * pair_penalty, rel_tol=1e-6)
    assert isinstance(method.base, FishrLinearProbeMethod)
    assert isinstance(control.base, FishrLinearProbeMethod)
    for environment in range(2):
        torch.testing.assert_close(
            method.base.ema.state(environment), control.base.ema.state(environment)
        )
    old_optimizer = algorithm._optimizer
    for update in range(3):
        before_a = algorithm._featurizer.weight.detach().clone()
        before_b = algorithm._classifier.weight.detach().clone()
        method.update(algorithm, features, targets, torch.arange(8))
        assert not torch.equal(before_a, algorithm._featurizer.weight)
        assert not torch.equal(before_b, algorithm._classifier.weight)
        if update == 0:
            assert algorithm._optimizer is not old_optimizer
            old_optimizer = algorithm._optimizer
        else:
            assert algorithm._optimizer is old_optimizer
        parameters = {
            id(p)
            for group in algorithm._optimizer.param_groups
            for p in group["params"]
        }
        assert parameters == {
            id(p)
            for layer in (algorithm._featurizer, algorithm._classifier)
            for p in layer.parameters()
        }


def test_factorized_checkpoint_inference_and_guard_against_collapsed_resume(
    tmp_path: Path,
) -> None:
    method, algorithm = _algorithm()
    features, targets, _ = _data()
    method.update(algorithm, features, targets, torch.arange(8))
    state = algorithm.capture_inference_state()
    identity = CheckpointIdentity(
        checkpoint_id="checkpoint:factorized",
        candidate_id="candidate:factorized",
        run_id="run:factorized",
        scientific_config_digest="sha256:factorized",
        epoch=1,
    )
    root = tmp_path / "checkpoint"
    persist_selected_linear_checkpoint(StoredCheckpoint(identity, state), root)
    loaded = PersistedLinearCheckpointStore(root).load(identity.checkpoint_id).state
    inference = LinearProbeAlgorithm(_config(), model_seed=0, projection=None)
    inference.restore_inference_state(loaded)
    assert torch.equal(inference.predict(features), algorithm.predict(features))
    assert inference.mean_loss(features, targets) == algorithm.mean_loss(
        features, targets
    )
    method.update(algorithm, features, targets, torch.arange(8))
    algorithm.restore_inference_state(loaded)
    assert algorithm.training_diagnostics() == state.diagnostics
    assert torch.equal(algorithm.predict(features), inference.predict(features))
    algorithm.restore_inference_state(LinearProbeState(state.weight, state.bias))
    with pytest.raises(ValueError, match="collapsed inference checkpoint"):
        method.update(algorithm, features, targets, torch.arange(8))
    algorithm.restore_inference_state(loaded)
    method.update(algorithm, features, targets, torch.arange(8))


def test_standalone_matchdg_retains_updates_and_scaling_diagnostics() -> None:
    method, algorithm = _algorithm()
    matchdg = MatchDgAlgorithm(
        _config(),
        model_seed=5,
        latent_dim=4,
        pair_differences=_data()[2],
        penalty_weight=2.0,
    )
    features, targets, _ = _data()
    for _ in range(3):
        assert method.update(
            algorithm, features, targets, torch.arange(8)
        ) == matchdg.update(features, targets)
        assert torch.equal(
            algorithm.capture_inference_state().weight,
            matchdg.capture_inference_state().weight,
        )
    before = algorithm.training_diagnostics()
    logits = algorithm._classifier(algorithm.prepare_features(features)).detach()
    with torch.no_grad():
        algorithm._featurizer.weight.mul_(0.5)
        algorithm._featurizer.bias.mul_(0.5)
        algorithm._classifier.weight.mul_(2.0)
    after = algorithm.training_diagnostics()
    torch.testing.assert_close(
        algorithm._classifier(algorithm.prepare_features(features)), logits
    )
    assert isclose(
        after["pairs/representation_discrepancy"],
        before["pairs/representation_discrepancy"] / 4,
        rel_tol=1e-6,
    )
    assert isclose(
        after["pairs/logit_discrepancy"],
        before["pairs/logit_discrepancy"],
        rel_tol=1e-6,
    )
    assert isclose(
        after["classifier/weight_norm"],
        before["classifier/weight_norm"] * 2,
        rel_tol=1e-6,
    )


def test_configuration_rejects_ambiguous_interventions_and_nonzero_control() -> None:
    from pydantic import ValidationError

    from grit.config import ComposedAlgorithmConfig, ErmAlgorithmConfig

    valid = ComposedAlgorithmConfig(
        kind="composed",
        base_objective=ErmAlgorithmConfig(kind="erm"),
        pair_intervention="representation_consistency",
        representation_consistency_weight=0.1,
        latent_dim=32,
        objective_version="factorized-pairs/v1",
    )
    for update in (
        {"objective_version": "prediction-pairs/v1"},
        {"consistency_weight": 0.1},
        {"latent_dim": None},
        {"pair_intervention": "two_layer"},
    ):
        with pytest.raises(ValidationError):
            ComposedAlgorithmConfig.model_validate({**valid.model_dump(), **update})
    with pytest.raises(ValueError, match="must not receive pairs"):
        ComposedLinearProbeMethod(
            _base("erm"),
            "erm_two_layer",
            latent_dim=4,
            pair_differences=_data()[2],
        )
