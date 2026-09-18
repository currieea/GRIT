"""Pair interventions composed with an unchanged base objective and sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from grit.config import LinearProbeTrainingConfig
from grit.methods.matchdg import FactorizedLinearProbeAlgorithm
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeState,
    LinearProbeTrainingMethod,
    prediction_consistency_penalty,
)
from grit.methods.types import MethodId, pair_intervention

__all__ = ["ComposedLinearProbeMethod", "prediction_consistency_penalty"]


@dataclass(slots=True)
class ComposedLinearProbeMethod:
    """Reuse base sampling, objective schedules, optimizer resets and checkpoints.

    The predictor supplies its actual classifier inputs to explicit gradient
    statistics. It applies projection or adds the selected consistency term outside
    supervised statistics, including throughout Fishr warm-up.
    """

    base: LinearProbeTrainingMethod
    method_id: MethodId
    projection: FittedLinearProjection | None = None
    projection_rank: int | None = None
    pair_differences: torch.Tensor | None = None
    consistency_weight: float = 0.0
    representation_consistency_weight: float = 0.0
    latent_dim: int | None = None

    def __post_init__(self) -> None:
        if self.base.method_id not in ("erm", "rex", "irm", "fishr"):
            raise ValueError("pair interventions support ERM, V-REx, IRMv1 and Fishr")
        if self.base.projection is not None:
            raise ValueError("base objective must not already have a projection")
        if (self.projection is None) != (self.projection_rank is None):
            raise ValueError(
                "a fitted projection and its rank must be supplied together"
            )
        if self.projection is not None:
            expected = (
                "grit"
                if self.base.method_id == "erm"
                else f"{self.base.method_id}_grit"
            )
            if self.pair_differences is not None or self.consistency_weight != 0.0:
                raise ValueError(
                    "projection and consistency are exclusive interventions"
                )
        elif self.latent_dim is not None:
            intervention = pair_intervention(self.method_id)
            if intervention not in ("representation_consistency", "two_layer"):
                raise ValueError("latent_dim requires a factorized intervention")
            expected = f"{self.base.method_id}_{intervention}"
            if self.consistency_weight != 0.0:
                raise ValueError("factorized predictors do not use prediction strength")
            if intervention == "two_layer":
                if (
                    self.pair_differences is not None
                    or self.representation_consistency_weight != 0.0
                ):
                    raise ValueError(
                        "two-layer control must not receive pairs or a pair penalty"
                    )
            elif self.pair_differences is None:
                raise ValueError("representation consistency requires training pairs")
        else:
            expected = f"{self.base.method_id}_consistency"
            if self.pair_differences is None:
                raise ValueError(
                    "prediction consistency requires training-pair differences"
                )
        if self.latent_dim is None and self.representation_consistency_weight != 0.0:
            raise ValueError("representation strength requires latent_dim")
        if self.projection is not None and self.latent_dim is not None:
            raise ValueError("factorized predictors cannot use a projection")
        if self.method_id != expected:
            raise ValueError("composed method identity does not match its intervention")

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return self.base.epoch_batches(
            row_count=row_count, batch_size=batch_size, generator=generator
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        return self.base.update(algorithm, features, targets, rows)

    def build_algorithm(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm:
        if self.latent_dim is not None:
            return FactorizedLinearProbeAlgorithm(
                config,
                model_seed=model_seed,
                num_classes=num_classes,
                latent_dim=self.latent_dim,
                pair_differences=self.pair_differences,
                penalty_weight=self.representation_consistency_weight,
            )
        return LinearProbeAlgorithm(
            config,
            model_seed=model_seed,
            projection=self.projection,
            num_classes=num_classes,
            pair_differences=self.pair_differences,
            consistency_weight=self.consistency_weight,
        )

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        return self.base.checkpoint_state(algorithm)
