"""Fish, LISA, SWAD, MatchDG, SD, Fishr, and RDM for the shared linear-probe trainer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import torch

from grit.config import LinearProbeTrainingConfig
from grit.methods.fishr import FishrGradientVarianceEma, fishr_penalty
from grit.methods.invariance import (
    environment_balanced_epoch_batches,
    spectral_decoupling_objective,
)
from grit.methods.lisa import (
    LisaBatchPlan,
    mix_features_and_targets,
    plan_lisa_epoch,
    soft_cross_entropy,
)
from grit.methods.matchdg import MatchDgAlgorithm
from grit.methods.rdm import rdm_objective, require_rdm_environment_rows
from grit.methods.swad import LossValley, RunningAverage, SwadSegment
from grit.methods.training import (
    LinearProbeAlgorithm,
    LinearProbeMethodDefaults,
    LinearProbeState,
    random_epoch_batches,
    require_aligned_ids,
)


@dataclass(slots=True)
class FishLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced batches with the inherited Reptile-style Fish step."""

    environment_ids: torch.Tensor
    environment_count: int
    meta_step_size: float
    method_id: Literal["fish"] = field(init=False, default="fish")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.meta_step_size <= 0.0:
            raise ValueError("Fish meta step size must be positive")
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        return environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        start = algorithm.capture_inference_state()
        batch_environments = self.environment_ids[rows]
        losses: list[float] = []
        for environment in range(self.environment_count):
            environment_rows = rows[batch_environments == environment]
            if int(environment_rows.numel()) == 0:
                raise ValueError("every Fish minibatch must contain every environment")
            losses.append(
                algorithm.update(features[environment_rows], targets[environment_rows])
            )
        inner = algorithm.capture_inference_state()
        step = self.meta_step_size
        algorithm.restore_inference_state(
            LinearProbeState(
                weight=start.weight + step * (inner.weight - start.weight),
                bias=start.bias + step * (inner.bias - start.bias),
            )
        )
        return sum(losses) / len(losses)


@dataclass(slots=True)
class LisaLinearProbeMethod(LinearProbeMethodDefaults):
    """Single-group minibatches mixed with intra-label or intra-domain partners."""

    group_ids: torch.Tensor
    group_count: int
    selection_prob: float
    num_classes: int = 2
    method_id: Literal["lisa"] = field(init=False, default="lisa")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _planned: deque[LisaBatchPlan] = field(
        init=False, default_factory=deque, repr=False
    )

    def __post_init__(self) -> None:
        self.group_ids = self.group_ids.detach().cpu().to(torch.int64)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.group_ids, row_count, "group")
        plans = plan_lisa_epoch(
            self.group_ids,
            group_count=self.group_count,
            batch_size=batch_size,
            selection_prob=self.selection_prob,
            generator=generator,
        )
        self._planned = deque(plans)
        return tuple(plan.rows for plan in plans)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        if not self._planned:
            raise ValueError("LISA update requested outside a planned epoch")
        plan = self._planned.popleft()
        if not torch.equal(plan.rows, rows):
            raise ValueError("LISA update rows do not match the planned minibatch")
        mixed_features, mixed_targets = mix_features_and_targets(
            features, targets, plan, num_classes=self.num_classes
        )
        return algorithm.update_with_logits_objective(
            mixed_features,
            targets[rows],
            lambda logits, _targets: soft_cross_entropy(logits, mixed_targets),
        )


@dataclass(slots=True)
class SwadLinearProbeMethod(LinearProbeMethodDefaults):
    """ERM minibatches whose checkpoints are the SWAD loss-valley average."""

    loss_features: torch.Tensor
    loss_targets: torch.Tensor
    tolerance_ratio: float
    segment_updates: int
    n_converge: int = 3
    n_tolerance: int = 6
    method_id: Literal["swad"] = field(init=False, default="swad")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _valley: LossValley = field(init=False, repr=False)
    _segment: RunningAverage | None = field(init=False, default=None, repr=False)
    _segment_start: int = field(init=False, default=0, repr=False)
    _update_count: int = field(init=False, default=0, repr=False)
    _last_loss: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.segment_updates <= 0:
            raise ValueError("SWAD segment length must be positive")
        self.loss_features = self.loss_features.detach().cpu().to(torch.float32)
        self.loss_targets = self.loss_targets.detach().cpu().to(torch.int64)
        if self.loss_features.ndim != 2 or int(self.loss_features.shape[0]) != int(
            self.loss_targets.shape[0]
        ):
            raise ValueError("SWAD validation-loss features and targets must align")
        self._valley = LossValley(
            n_converge=self.n_converge,
            n_tolerance=self.n_tolerance,
            tolerance_ratio=self.tolerance_ratio,
        )

    @property
    def dead(self) -> bool:
        return self._valley.dead

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return random_epoch_batches(row_count, batch_size, generator)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        if self._valley.dead:
            # The reference implementation stops training once the valley ends.
            return self._last_loss
        self._last_loss = algorithm.update(features[rows], targets[rows])
        live = algorithm.capture_inference_state()
        if self._segment is None:
            self._segment = RunningAverage(live)
            self._segment_start = self._update_count
        self._segment.add(live)
        self._update_count += 1
        if self._update_count - self._segment_start == self.segment_updates:
            self._valley.observe(
                SwadSegment(
                    state=self._segment.state,
                    start_update=self._segment_start,
                    end_update=self._update_count,
                    end_loss=algorithm.mean_loss(
                        self.loss_features, self.loss_targets
                    ),
                )
            )
            self._segment = None
        return self._last_loss

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        return self._valley.current_state(algorithm.capture_inference_state())


@dataclass(frozen=True, slots=True)
class MatchDgLinearProbeMethod:
    """ERM minibatches on the factorized model with the oracle-pair penalty."""

    pair_differences: torch.Tensor
    latent_dim: int
    penalty_weight: float
    method_id: Literal["matchdg"] = field(init=False, default="matchdg")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return random_epoch_batches(row_count, batch_size, generator)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        return algorithm.update(features[rows], targets[rows])

    def build_algorithm(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm:
        return MatchDgAlgorithm(
            config,
            model_seed=model_seed,
            latent_dim=self.latent_dim,
            pair_differences=self.pair_differences,
            penalty_weight=self.penalty_weight,
            num_classes=num_classes,
        )

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        return algorithm.capture_inference_state()


@dataclass(frozen=True, slots=True)
class SpectralDecouplingLinearProbeMethod(LinearProbeMethodDefaults):
    """ERM minibatches with the raw-logit magnitude penalty, active from update 0."""

    penalty_weight: float
    method_id: Literal["sd"] = field(init=False, default="sd")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        return random_epoch_batches(row_count, batch_size, generator)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        return algorithm.update_with_logits_objective(
            features[rows],
            targets[rows],
            partial(spectral_decoupling_objective, penalty_weight=self.penalty_weight),
        )


@dataclass(slots=True)
class _WarmedUpPenalty:
    """Shared warm-up bookkeeping: the penalty starts, and Adam restarts, together."""

    penalty_weight: float
    warm_up_updates: int
    _update_count: int = field(init=False, default=0, repr=False)

    def begin_update(self, algorithm: LinearProbeAlgorithm) -> float:
        """Return this update's penalty weight, resetting Adam once on activation."""

        if self._update_count == self.warm_up_updates:
            algorithm.reset_optimizer()
        return 0.0 if self._update_count < self.warm_up_updates else self.penalty_weight

    def end_update(self) -> None:
        self._update_count += 1

    @property
    def update_count(self) -> int:
        return self._update_count


@dataclass(slots=True)
class FishrLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs matching per-environment gradient variances."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    ema_decay: float
    method_id: Literal["fishr"] = field(init=False, default="fishr")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _schedule: _WarmedUpPenalty = field(init=False, repr=False)
    _ema: FishrGradientVarianceEma = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.penalty_anneal_updates < 0:
            raise ValueError("Fishr warm-up length must be non-negative")
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)
        self._schedule = _WarmedUpPenalty(
            penalty_weight=self.penalty_weight,
            warm_up_updates=self.penalty_anneal_updates,
        )
        self._ema = FishrGradientVarianceEma(
            environment_count=self.environment_count,
            decay=self.ema_decay,
        )

    @property
    def ema(self) -> FishrGradientVarianceEma:
        """Training-only state; evaluation never touches it."""

        return self._ema

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        return environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        penalty_weight = self._schedule.begin_update(algorithm)
        batch_environments = self.environment_ids[rows]
        prepared = algorithm.prepare_features(features[rows])

        def objective(
            logits: torch.Tensor, prepared_targets: torch.Tensor
        ) -> torch.Tensor:
            # The EMA advances during warm-up too, so the penalty it later reports
            # already reflects the whole run rather than restarting at activation.
            penalty = fishr_penalty(
                logits,
                prepared_targets,
                prepared,
                batch_environments,
                environment_count=self.environment_count,
                ema=self._ema,
            )
            cross_entropy = torch.nn.functional.cross_entropy(logits, prepared_targets)
            if penalty_weight == 0.0:
                return cross_entropy
            return cross_entropy + penalty_weight * penalty

        loss = algorithm.update_with_logits_objective(
            features[rows], targets[rows], objective
        )
        self._schedule.end_update()
        return loss


@dataclass(slots=True)
class RdmLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs matching the worst environment's risk spread."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    variance_weight: float
    method_id: Literal["rdm"] = field(init=False, default="rdm")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _schedule: _WarmedUpPenalty = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.penalty_anneal_updates < 0:
            raise ValueError("RDM warm-up length must be non-negative")
        if self.variance_weight < 0.0:
            raise ValueError("RDM variance weight must be non-negative")
        self.environment_ids = self.environment_ids.detach().cpu().to(torch.int64)
        self._schedule = _WarmedUpPenalty(
            penalty_weight=self.penalty_weight,
            warm_up_updates=self.penalty_anneal_updates,
        )

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.environment_ids, row_count, "environment")
        batches = environment_balanced_epoch_batches(
            self.environment_ids,
            environment_count=self.environment_count,
            batch_size=batch_size,
            generator=generator,
        )
        # Reject the run before its first update rather than quietly evaluating the
        # sample variance on a one-example remainder.
        for rows in batches:
            environments = self.environment_ids[rows]
            require_rdm_environment_rows(
                tuple(
                    int((environments == environment).sum())
                    for environment in range(self.environment_count)
                )
            )
        return batches

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        penalty_weight = self._schedule.begin_update(algorithm)
        active = penalty_weight != 0.0
        batch_environments = self.environment_ids[rows]
        loss = algorithm.update_with_objective(
            features[rows],
            targets[rows],
            partial(
                rdm_objective,
                environment_ids=batch_environments,
                environment_count=self.environment_count,
                penalty_weight=penalty_weight,
                variance_weight=self.variance_weight if active else 0.0,
            ),
        )
        self._schedule.end_update()
        return loss
