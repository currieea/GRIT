"""Factorized linear predictors and standalone MatchDG-style compatibility."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch

from grit.config import LinearProbeTrainingConfig
from grit.methods.training import CallableWithoutArguments, LinearProbeAlgorithm
from grit.methods.training_state import LinearProbeState


class FactorizedLinearProbeAlgorithm(LinearProbeAlgorithm):
    """Two affine layers sharing the objective callbacks and epoch lifecycle.

    The direct head mirrors BA for inference; factor parameters remain available for
    diagnostics and subsequent updates. A collapsed-only restore forbids updates.
    """

    def __init__(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        latent_dim: int,
        pair_differences: torch.Tensor | None,
        penalty_weight: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            config, model_seed=model_seed, projection=None, num_classes=num_classes
        )
        if latent_dim <= 0:
            raise ValueError("representation latent dimension must be positive")
        if penalty_weight < 0.0 or not math.isfinite(penalty_weight):
            raise ValueError("representation strength must be non-negative and finite")
        if pair_differences is None and penalty_weight != 0.0:
            raise ValueError("positive representation strength requires training pairs")
        differences = None
        if pair_differences is not None:
            differences = pair_differences.detach().cpu().to(torch.float32).clone()
            if (
                differences.ndim != 2
                or int(differences.shape[0]) == 0
                or int(differences.shape[1]) != 512
                or not bool(torch.isfinite(differences).all())
            ):
                raise ValueError(
                    "pair differences must be finite with shape [pairs, 512]"
                )
        self._pair_differences = differences
        self._factorized_updates_available = True
        self._penalty_weight = float(penalty_weight)
        generator = torch.Generator(device="cpu").manual_seed(model_seed)
        self._featurizer = torch.nn.Linear(512, latent_dim, bias=True, device="cpu")
        self._classifier = torch.nn.Linear(
            latent_dim, num_classes, bias=True, device="cpu"
        )
        for layer in (self._featurizer, self._classifier):
            bound = 1.0 / math.sqrt(int(layer.in_features))
            with torch.no_grad():
                layer.weight.copy_(
                    torch.rand(layer.weight.shape, generator=generator).mul(2 * bound)
                    - bound
                )
                layer.bias.copy_(
                    torch.rand(layer.bias.shape, generator=generator).mul(2 * bound)
                    - bound
                )
        self.reset_optimizer()
        self._recompose()

    def reset_optimizer(self) -> None:
        self._optimizer = torch.optim.Adam(
            [*self._featurizer.parameters(), *self._classifier.parameters()],
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """Actual final-classifier inputs, retaining their graph for Fishr."""
        return self._featurizer(self._prepare(features))

    def pair_penalty(self) -> torch.Tensor:
        """Mean squared representation difference; the affine bias cancels."""

        if self._pair_differences is None:
            return self._featurizer.weight.new_zeros(())
        response = torch.nn.functional.linear(
            self._pair_differences, self._featurizer.weight, bias=None
        )
        return response.pow(2).sum(dim=1).mean()

    def update_with_logits_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        consistency_scale: float = 1.0,
    ) -> float:
        if not self._factorized_updates_available:
            raise ValueError(
                "collapsed inference checkpoint cannot resume factorized updates"
            )
        prepared = self.prepare_features(features)
        prepared_targets = targets.detach().cpu().to(torch.int64)
        self._optimizer.zero_grad(set_to_none=True)
        logits = self._classifier(prepared)
        loss = objective(logits, prepared_targets) + (
            consistency_scale * self._penalty_weight * self.pair_penalty()
        )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("factorized objective must return one finite scalar")
        backward = cast(CallableWithoutArguments, loss.backward)
        step = cast(CallableWithoutArguments, self._optimizer.step)
        backward()
        step()
        self._recompose()
        return float(loss.detach().item())

    def training_diagnostics(self) -> dict[str, float]:
        if not self._factorized_updates_available:
            return {}
        with torch.no_grad():
            values = {
                "representation/weight_norm": float(
                    self._featurizer.weight.square().sum().sqrt()
                ),
                "classifier/weight_norm": float(
                    self._classifier.weight.square().sum().sqrt()
                ),
            }
            if self._pair_differences is not None:
                response = torch.nn.functional.linear(
                    self._pair_differences, self._featurizer.weight
                )
                logits = torch.nn.functional.linear(response, self._classifier.weight)
                values["pairs/representation_discrepancy"] = float(
                    response.square().sum(dim=1).mean()
                )
                values["pairs/logit_discrepancy"] = float(
                    logits.square().sum(dim=1).mean()
                )
            return values

    def capture_inference_state(self) -> LinearProbeState:
        collapsed = super().capture_inference_state()
        factors = None
        if self._factorized_updates_available:
            factors = {
                f"{name}_{parameter}": tensor.detach().cpu().clone()
                for name, layer in (
                    ("representation", self._featurizer),
                    ("classifier", self._classifier),
                )
                for parameter, tensor in (
                    ("weight", layer.weight),
                    ("bias", layer.bias),
                )
            }
        return LinearProbeState(
            weight=collapsed.weight,
            bias=collapsed.bias,
            projection_basis=collapsed.projection_basis,
            factor_parameters=factors,
            diagnostics=self.training_diagnostics(),
        )

    def restore_inference_state(self, state: LinearProbeState) -> None:
        factors = state.factor_parameters
        if factors is not None:
            expected = {
                f"{name}_{parameter}": tensor
                for name, layer in (
                    ("representation", self._featurizer),
                    ("classifier", self._classifier),
                )
                for parameter, tensor in (
                    ("weight", layer.weight),
                    ("bias", layer.bias),
                )
            }
            if factors.keys() != expected.keys() or any(
                factors[name].shape != tensor.shape
                or not bool(torch.isfinite(factors[name]).all())
                for name, tensor in expected.items()
            ):
                raise ValueError(
                    "factorized checkpoint parameters have incompatible shapes"
                )
            if not torch.allclose(
                state.weight,
                factors["classifier_weight"] @ factors["representation_weight"],
            ):
                raise ValueError(
                    "factorized checkpoint does not match collapsed weight"
                )
            if not torch.allclose(
                state.bias,
                factors["classifier_weight"] @ factors["representation_bias"]
                + factors["classifier_bias"],
            ):
                raise ValueError("factorized checkpoint does not match collapsed bias")
            with torch.no_grad():
                for name, tensor in expected.items():
                    tensor.copy_(factors[name])
        super().restore_inference_state(state)
        self._factorized_updates_available = factors is not None

    def _recompose(self) -> None:
        with torch.no_grad():
            weight = self._classifier.weight @ self._featurizer.weight
            bias = self._classifier.weight @ self._featurizer.bias + (
                self._classifier.bias
            )
            self._model.weight.copy_(weight)
            self._model.bias.copy_(bias)


class MatchDgAlgorithm(FactorizedLinearProbeAlgorithm):
    """Keep standalone MatchDG's positive-strength contract."""

    def __init__(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        latent_dim: int,
        pair_differences: torch.Tensor,
        penalty_weight: float,
        num_classes: int = 2,
    ) -> None:
        if penalty_weight <= 0.0:
            raise ValueError("MatchDG penalty weight must be positive and finite")
        super().__init__(
            config,
            model_seed=model_seed,
            latent_dim=latent_dim,
            pair_differences=pair_differences,
            penalty_weight=penalty_weight,
            num_classes=num_classes,
        )
