"""The inherited MatchDG-style pair-difference penalty on a linear featurizer."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch

from grit.config import LinearProbeTrainingConfig
from grit.methods.training import CallableWithoutArguments, LinearProbeAlgorithm


class MatchDgAlgorithm(LinearProbeAlgorithm):
    """Linear featurizer plus linear classifier trained with a pair penalty.

    Both layers are linear, so the composition is a linear probe. The inherited
    ``_model`` head always mirrors that composition: prediction, checkpoint capture, and
    restoration go through it unchanged. Restoring a state is only meaningful for
    evaluation; further updates would recompose from the factor parameters.
    """

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
        super().__init__(
            config, model_seed=model_seed, projection=None, num_classes=num_classes
        )
        if latent_dim <= 0:
            raise ValueError("MatchDG latent dimension must be positive")
        if penalty_weight <= 0.0 or not math.isfinite(penalty_weight):
            raise ValueError("MatchDG penalty weight must be positive and finite")
        differences = pair_differences.detach().cpu().to(torch.float32)
        if differences.ndim != 2 or int(differences.shape[0]) == 0 or int(
            differences.shape[1]
        ) != 512:
            raise ValueError("MatchDG pair differences must have shape [pairs, 512]")
        self._pair_differences = differences
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
        self._optimizer = torch.optim.Adam(
            [*self._featurizer.parameters(), *self._classifier.parameters()],
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self._recompose()

    def pair_penalty(self) -> torch.Tensor:
        """Mean squared featurizer output over the pair differences."""

        return self._featurizer(self._pair_differences).pow(2).sum(dim=1).mean()

    def update_with_logits_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        prepared = self._prepare(features)
        prepared_targets = targets.detach().cpu().to(torch.int64)
        self._optimizer.zero_grad(set_to_none=True)
        logits = self._classifier(self._featurizer(prepared))
        loss = objective(logits, prepared_targets) + (
            self._penalty_weight * self.pair_penalty()
        )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("MatchDG objective must return one finite scalar")
        backward = cast(CallableWithoutArguments, loss.backward)
        step = cast(CallableWithoutArguments, self._optimizer.step)
        backward()
        step()
        self._recompose()
        return float(loss.detach().item())

    def _recompose(self) -> None:
        with torch.no_grad():
            weight = self._classifier.weight @ self._featurizer.weight
            bias = self._classifier.weight @ self._featurizer.bias + (
                self._classifier.bias
            )
            self._model.weight.copy_(weight)
            self._model.bias.copy_(bias)
