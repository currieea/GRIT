"""Minimal frozen-feature trainer and restoration path for Waterbirds ERM/GRIT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from pydantic import Field, StrictStr

from grit.config import LinearProbeTrainingConfig
from grit.data.waterbirds import WaterbirdsAdjustedWeightSpec
from grit.features.waterbirds import (
    WaterbirdsEvaluationFeatureTable,
    WaterbirdsTrainingFeatureTable,
)
from grit.methods.checkpoints import CheckpointStore
from grit.methods.projection import FittedLinearProjection
from grit.methods.training import (
    InMemoryLinearCheckpointStore,
    LinearProbeAlgorithm,
    LinearProbeState,
)
from grit.methods.types import MethodId
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.selection.cmnist import CheckpointIdentity
from grit.selection.waterbirds import (
    FrozenWaterbirdsCheckpoint,
    WaterbirdsValidationMetricRecord,
    compute_waterbirds_validation_metric,
)

WaterbirdsMethod: TypeAlias = MethodId


class WaterbirdsRestorationReceipt(StrictBoundaryModel):
    receipt_id: StrictStr = Field(min_length=1)
    candidate_selection_id: StrictStr = Field(min_length=1)
    store_id: StrictStr = Field(min_length=1)
    checkpoint: CheckpointIdentity


@dataclass(frozen=True, slots=True)
class TrainedWaterbirdsRun:
    run_id: str
    candidate_id: str
    method_id: WaterbirdsMethod
    seed_stage: SeedStage
    seed: int
    projection_rank: int | None
    validation_metrics: tuple[WaterbirdsValidationMetricRecord, ...]
    store: InMemoryLinearCheckpointStore
    algorithm: LinearProbeAlgorithm
    epoch_losses: tuple[float, ...]


def train_waterbirds_linear_probe(
    training: WaterbirdsTrainingFeatureTable,
    validation: WaterbirdsEvaluationFeatureTable,
    adjusted_weights: WaterbirdsAdjustedWeightSpec,
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    method_id: WaterbirdsMethod,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    projection: FittedLinearProjection | None,
    projection_rank: int | None,
) -> TrainedWaterbirdsRun:
    """Run trainer-owned epochs and emit one four-group validation record each."""

    if training.dataset_manifest_digest != validation.dataset_manifest_digest:
        raise ValueError("Waterbirds train and validation datasets do not match")
    if (
        training.feature_cache_manifest_digest
        != validation.feature_cache_manifest_digest
    ):
        raise ValueError("Waterbirds train and validation feature caches do not match")
    if training.normalization != validation.normalization:
        raise ValueError("Waterbirds train and validation normalizations do not match")
    if method_id == "erm" and (projection is not None or projection_rank is not None):
        raise ValueError("Waterbirds ERM cannot use a projection")
    if method_id == "grit" and (projection is None or projection_rank is None):
        raise ValueError("Waterbirds GRIT requires a projection and rank")
    if int(training.features.shape[0]) == 0:
        raise ValueError("Waterbirds training data must not be empty")
    torch.use_deterministic_algorithms(True)
    algorithm = LinearProbeAlgorithm(config, model_seed=seed, projection=projection)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    store = InMemoryLinearCheckpointStore(store_id=f"memory:{run_id}")
    metrics: list[WaterbirdsValidationMetricRecord] = []
    losses: list[float] = []
    for epoch in range(1, config.max_epochs + 1):
        permutation = torch.randperm(len(training.record_ids), generator=generator)
        batch_losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            rows = permutation[start : start + config.batch_size]
            batch_losses.append(
                algorithm.update(training.features[rows], training.labels[rows])
            )
        losses.append(sum(batch_losses) / len(batch_losses))
        checkpoint_id = f"checkpoint:{run_id}:epoch:{epoch}"
        identity = CheckpointIdentity(
            checkpoint_id=checkpoint_id,
            candidate_id=candidate_id,
            run_id=run_id,
            scientific_config_digest=scientific_config_digest,
            epoch=epoch,
        )
        store.save(identity, algorithm.capture_inference_state())
        predictions = algorithm.predict(validation.features)
        metrics.append(
            compute_waterbirds_validation_metric(
                validation,
                predictions,
                adjusted_weights=adjusted_weights,
                record_id=f"metric:{checkpoint_id}:validation",
                run_id=run_id,
                candidate_id=candidate_id,
                method_id=method_id,
                scientific_config_digest=scientific_config_digest,
                checkpoint_id=checkpoint_id,
                epoch=epoch,
                seed_stage=seed_stage,
                seed=seed,
                projection_rank=projection_rank,
            )
        )
    return TrainedWaterbirdsRun(
        run_id=run_id,
        candidate_id=candidate_id,
        method_id=method_id,
        seed_stage=seed_stage,
        seed=seed,
        projection_rank=projection_rank,
        validation_metrics=tuple(metrics),
        store=store,
        algorithm=algorithm,
        epoch_losses=tuple(losses),
    )


def restore_waterbirds_checkpoint(
    selection: FrozenWaterbirdsCheckpoint,
    store: CheckpointStore[LinearProbeState],
    algorithm: LinearProbeAlgorithm,
) -> WaterbirdsRestorationReceipt:
    """Restore exactly the selected inference state and issue a typed receipt."""

    frozen = FrozenWaterbirdsCheckpoint.model_validate_json(selection.canonical_json())
    stored = store.load(frozen.checkpoint.checkpoint_id)
    if stored.identity != frozen.checkpoint:
        raise ValueError("Waterbirds checkpoint store returned the wrong identity")
    algorithm.restore_inference_state(stored.state)
    return WaterbirdsRestorationReceipt(
        receipt_id=f"restored:{store.store_id}:{stored.identity.checkpoint_id}",
        candidate_selection_id=frozen.candidate_selection_id,
        store_id=store.store_id,
        checkpoint=stored.identity,
    )
