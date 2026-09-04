"""Environment-aware minibatches and invariant-training objectives."""

from __future__ import annotations

import math

import torch

CMNIST_TRAINING_ENVIRONMENT_COUNT = 2


def environment_balanced_epoch_batches(
    environment_ids: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    """Shuffle each equally sized environment and exhaust it without replacement."""

    ids = environment_ids.detach().cpu().to(torch.int64)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("environment IDs must be a non-empty vector")
    if batch_size <= 0 or batch_size % CMNIST_TRAINING_ENVIRONMENT_COUNT != 0:
        raise ValueError("environment-balanced batch size must be positive and even")
    if not bool(((ids == 0) | (ids == 1)).all()) or any(
        not bool((ids == environment).any())
        for environment in range(CMNIST_TRAINING_ENVIRONMENT_COUNT)
    ):
        raise ValueError("CMNIST invariant training requires environment IDs 0 and 1")
    rows_by_environment = tuple(
        torch.nonzero(ids == environment, as_tuple=False).flatten()
        for environment in range(CMNIST_TRAINING_ENVIRONMENT_COUNT)
    )
    counts = {int(rows.numel()) for rows in rows_by_environment}
    if len(counts) != 1:
        raise ValueError("environment-balanced epochs require equal environment sizes")
    shuffled = tuple(
        rows[torch.randperm(int(rows.numel()), generator=generator)]
        for rows in rows_by_environment
    )
    per_environment = batch_size // CMNIST_TRAINING_ENVIRONMENT_COUNT
    environment_size = int(shuffled[0].numel())
    return tuple(
        torch.cat(
            [rows[start : start + per_environment] for rows in shuffled], dim=0
        )
        for start in range(0, environment_size, per_environment)
    )


def environment_mean_losses(
    per_example_losses: torch.Tensor,
    environment_ids: torch.Tensor,
) -> torch.Tensor:
    """Return one mean loss for each required CMNIST training environment."""

    if per_example_losses.ndim != 1:
        raise ValueError("per-example losses must be a vector")
    ids = environment_ids.to(device=per_example_losses.device, dtype=torch.int64)
    if ids.shape != per_example_losses.shape:
        raise ValueError("environment IDs must align with per-example losses")
    risks: list[torch.Tensor] = []
    for environment in range(CMNIST_TRAINING_ENVIRONMENT_COUNT):
        mask = ids == environment
        if not bool(mask.any()):
            raise ValueError(
                "every invariant-training batch must contain both environments"
            )
        risks.append(per_example_losses[mask].mean())
    if not bool(((ids == 0) | (ids == 1)).all()):
        raise ValueError("CMNIST invariant training requires environment IDs 0 and 1")
    return torch.stack(risks)


def annealed_penalty_weight(
    configured_weight: float,
    *,
    anneal_updates: int,
    update_count: int,
) -> float:
    if not math.isfinite(configured_weight) or configured_weight <= 0.0:
        raise ValueError("penalty weight must be positive and finite")
    if anneal_updates < 0 or update_count < 0:
        raise ValueError("anneal updates and update count must be non-negative")
    return configured_weight if update_count >= anneal_updates else 1.0


def vrex_objective(
    per_example_losses: torch.Tensor,
    environment_ids: torch.Tensor,
    *,
    penalty_weight: float,
) -> torch.Tensor:
    """Compute mean environment risk plus population risk variance."""

    risks = environment_mean_losses(per_example_losses, environment_ids)
    mean_risk = risks.mean()
    penalty = ((risks - mean_risk) ** 2).mean()
    objective = mean_risk + penalty_weight * penalty
    if penalty_weight > 1.0:
        objective = objective / penalty_weight
    return objective
