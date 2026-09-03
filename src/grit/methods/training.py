"""Small real frozen-feature linear-probe path for CMNIST ERM and GRIT."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import StrictInt, StrictStr

from grit.config import LinearProbeTrainingConfig
from grit.features.cmnist import FeatureTable
from grit.methods.checkpoints import CheckpointStore, StoredCheckpoint
from grit.methods.projection import FittedLinearProjection
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.selection.cmnist import CheckpointIdentity, ValidationMetricRecord

MethodId = Literal["erm", "grit"]


@dataclass(frozen=True, slots=True)
class LinearProbeState:
    weight: torch.Tensor
    bias: torch.Tensor


class LinearProbeAlgorithm:
    """Own a two-class linear model, Adam, and one bounded update operation."""

    def __init__(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        projection: FittedLinearProjection | None,
    ) -> None:
        self._model = torch.nn.Linear(512, 2, bias=True, device="cpu")
        generator = torch.Generator(device="cpu").manual_seed(model_seed)
        bound = 1.0 / math.sqrt(512)
        weight = torch.rand(
            (2, 512), generator=generator, dtype=torch.float32
        ).mul(2 * bound).sub(bound)
        bias = torch.rand((2,), generator=generator, dtype=torch.float32).mul(
            2 * bound
        ).sub(bound)
        with torch.no_grad():
            self._model.weight.copy_(weight)
            self._model.bias.copy_(bias)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        self._projection = projection

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Perform the algorithm-owned mutation for one trainer-provided batch."""

        self._model.train()
        prepared = self._prepare(features)
        self._optimizer.zero_grad(set_to_none=True)
        logits = self._model(prepared)
        loss = torch.nn.functional.cross_entropy(logits, targets.to(torch.int64))
        backward = cast(CallableWithoutArguments, loss.backward)
        step = cast(CallableWithoutArguments, self._optimizer.step)
        backward()
        step()
        return float(loss.detach().item())

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Run side-effect-free evaluation through the same input transform."""

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.inference_mode():
                return self._model(self._prepare(features)).argmax(dim=1)
        finally:
            self._model.train(was_training)

    def capture_inference_state(self) -> LinearProbeState:
        return LinearProbeState(
            weight=self._model.weight.detach().cpu().clone(),
            bias=self._model.bias.detach().cpu().clone(),
        )

    def restore_inference_state(self, state: LinearProbeState) -> None:
        if state.weight.shape != (2, 512) or state.bias.shape != (2,):
            raise ValueError("linear checkpoint state has incompatible shapes")
        with torch.no_grad():
            self._model.weight.copy_(state.weight.to(torch.float32))
            self._model.bias.copy_(state.bias.to(torch.float32))

    def _prepare(self, features: torch.Tensor) -> torch.Tensor:
        prepared = features.detach().cpu().to(torch.float32)
        if prepared.ndim != 2 or int(prepared.shape[1]) != 512:
            raise ValueError("linear probe requires feature shape [N, 512]")
        if self._projection is not None:
            prepared = self._projection.transform(prepared)
        return prepared


class InMemoryLinearCheckpointStore(CheckpointStore[LinearProbeState]):
    """Epoch-candidate store; only the selected state is persisted by the runner."""

    def __init__(self, store_id: str) -> None:
        self._store_id = store_id
        self._states: dict[str, StoredCheckpoint[LinearProbeState]] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    def save(self, identity: CheckpointIdentity, state: LinearProbeState) -> None:
        if identity.checkpoint_id in self._states:
            raise ValueError("checkpoint identity was already stored")
        self._states[identity.checkpoint_id] = StoredCheckpoint(identity, state)

    def load(self, checkpoint_id: str) -> StoredCheckpoint[LinearProbeState]:
        try:
            return self._states[checkpoint_id]
        except KeyError as error:
            raise KeyError(f"checkpoint is unavailable: {checkpoint_id}") from error


@dataclass(frozen=True, slots=True)
class TrainedLinearProbeRun:
    run_id: str
    candidate_id: str
    method_id: MethodId
    seed_stage: SeedStage
    seed: int
    projection_rank: int | None
    validation_metrics: tuple[ValidationMetricRecord, ...]
    store: InMemoryLinearCheckpointStore
    algorithm: LinearProbeAlgorithm
    epoch_losses: tuple[float, ...]


def train_linear_probe(
    training_tables: tuple[FeatureTable, FeatureTable],
    validation_tables: tuple[FeatureTable, FeatureTable, FeatureTable],
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    method_id: MethodId,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    projection: FittedLinearProjection | None,
    projection_rank: int | None,
) -> TrainedLinearProbeRun:
    """Run trainer-owned epoch/batch iteration and emit validation every epoch."""

    if method_id == "erm" and projection is not None:
        raise ValueError("ERM must train on unprojected features")
    if method_id == "grit" and projection is None:
        raise ValueError("GRIT requires a fitted projection")
    if method_id == "erm" and projection_rank is not None:
        raise ValueError("ERM cannot declare a projection rank")
    if method_id == "grit" and projection_rank is None:
        raise ValueError("GRIT must declare its projection rank")
    torch.use_deterministic_algorithms(True)
    algorithm = LinearProbeAlgorithm(config, model_seed=seed, projection=projection)
    train_features = torch.cat(
        [table.features for table in training_tables], dim=0
    ).to(torch.float32)
    train_targets = torch.cat(
        [table.targets for table in training_tables], dim=0
    ).to(torch.int64)
    if int(train_features.shape[0]) == 0:
        raise ValueError("linear probe training data must be non-empty")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    store = InMemoryLinearCheckpointStore(store_id=f"memory:{run_id}")
    metrics: list[ValidationMetricRecord] = []
    epoch_losses: list[float] = []
    for epoch in range(1, config.max_epochs + 1):
        permutation = torch.randperm(int(train_features.shape[0]), generator=generator)
        batch_losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            rows = permutation[start : start + config.batch_size]
            batch_losses.append(
                algorithm.update(train_features[rows], train_targets[rows])
            )
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        checkpoint_id = f"checkpoint:{run_id}:epoch:{epoch}"
        identity = CheckpointIdentity(
            checkpoint_id=checkpoint_id,
            candidate_id=candidate_id,
            run_id=run_id,
            scientific_config_digest=scientific_config_digest,
            epoch=epoch,
        )
        store.save(identity, algorithm.capture_inference_state())
        metrics.extend(
            _validation_metrics(
                algorithm,
                validation_tables,
                run_id=run_id,
                candidate_id=candidate_id,
                method_id=method_id,
                scientific_config_digest=scientific_config_digest,
                seed_stage=seed_stage,
                seed=seed,
                checkpoint_id=checkpoint_id,
                epoch=epoch,
                projection_rank=projection_rank,
            )
        )
    return TrainedLinearProbeRun(
        run_id=run_id,
        candidate_id=candidate_id,
        method_id=method_id,
        seed_stage=seed_stage,
        seed=seed,
        projection_rank=projection_rank,
        validation_metrics=tuple(metrics),
        store=store,
        algorithm=algorithm,
        epoch_losses=tuple(epoch_losses),
    )


def evaluate_accuracy(algorithm: LinearProbeAlgorithm, table: FeatureTable) -> float:
    predictions = algorithm.predict(table.features)
    return float((predictions == table.targets).to(torch.float64).mean().item())


def _validation_metrics(
    algorithm: LinearProbeAlgorithm,
    tables: tuple[FeatureTable, FeatureTable, FeatureTable],
    *,
    run_id: str,
    candidate_id: str,
    method_id: MethodId,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    checkpoint_id: str,
    epoch: int,
    projection_rank: int | None,
) -> tuple[ValidationMetricRecord, ...]:
    return tuple(
        ValidationMetricRecord(
            record_id=f"metric:{run_id}:{checkpoint_id}:{table.name}",
            run_id=run_id,
            candidate_id=candidate_id,
            method_id=method_id,
            scientific_config_digest=scientific_config_digest,
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            seed=seed,
            value=evaluate_accuracy(algorithm, table),
            sample_count=len(table.source_ids),
            metric_kind="validation",
            seed_stage=seed_stage,
            split_name=cast(
                Literal["val_e01", "val_e02", "val_e05"], table.name
            ),
            metric_name="accuracy",
            projection_rank=projection_rank,
        )
        for table in tables
    )


class LinearCheckpointFile(StrictBoundaryModel):
    relative_path: StrictStr
    digest: StrictStr
    shape: tuple[StrictInt, ...]
    dtype: Literal["float32"]


class PersistedLinearCheckpointManifest(StrictBoundaryModel):
    schema_version: Literal["grit.linear-checkpoint/v1"]
    store_id: StrictStr
    checkpoint: CheckpointIdentity
    weight: LinearCheckpointFile
    bias: LinearCheckpointFile


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.float32]: ...


class _TorchFromNumpy(Protocol):
    def __call__(self, array: NDArray[np.float32]) -> torch.Tensor: ...


_torch_from_numpy = cast(_TorchFromNumpy, torch.from_numpy)


class CallableWithoutArguments(Protocol):
    def __call__(self) -> object: ...


class PersistedLinearCheckpointStore(CheckpointStore[LinearProbeState]):
    """Read one selected inference checkpoint from its narrow on-disk format."""

    def __init__(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        self._root = root
        self._manifest = PersistedLinearCheckpointManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    @property
    def store_id(self) -> str:
        return self._manifest.store_id

    def load(self, checkpoint_id: str) -> StoredCheckpoint[LinearProbeState]:
        if checkpoint_id != self._manifest.checkpoint.checkpoint_id:
            raise KeyError(f"checkpoint is unavailable: {checkpoint_id}")
        weight = _load_checkpoint_array(self._root, self._manifest.weight)
        bias = _load_checkpoint_array(self._root, self._manifest.bias)
        return StoredCheckpoint(
            self._manifest.checkpoint,
            LinearProbeState(weight=weight, bias=bias),
        )


def persist_selected_linear_checkpoint(
    stored: StoredCheckpoint[LinearProbeState],
    output_dir: Path,
) -> PersistedLinearCheckpointManifest:
    """Persist only the selected inference state; no resume guarantee is implied."""

    output_dir.mkdir(parents=True, exist_ok=False)
    weight = _write_checkpoint_array(output_dir, "weight.npy", stored.state.weight)
    bias = _write_checkpoint_array(output_dir, "bias.npy", stored.state.bias)
    manifest = PersistedLinearCheckpointManifest(
        schema_version="grit.linear-checkpoint/v1",
        store_id=f"linear-checkpoint:{stored.identity.checkpoint_id}",
        checkpoint=stored.identity,
        weight=weight,
        bias=bias,
    )
    (output_dir / "manifest.json").write_text(
        manifest.canonical_json() + "\n", encoding="utf-8"
    )
    return manifest


def _write_checkpoint_array(
    root: Path,
    name: str,
    tensor: torch.Tensor,
) -> LinearCheckpointFile:
    path = root / name
    contiguous = tensor.detach().cpu().to(torch.float32).contiguous()
    to_numpy = cast(_TensorToNumpy, contiguous.numpy)
    array = to_numpy()
    np.save(path, array, allow_pickle=False)
    return LinearCheckpointFile(
        relative_path=name,
        digest=_file_digest(path),
        shape=tuple(int(value) for value in contiguous.shape),
        dtype="float32",
    )


def _load_checkpoint_array(root: Path, file: LinearCheckpointFile) -> torch.Tensor:
    path = root / file.relative_path
    if _file_digest(path) != file.digest:
        raise ValueError("persisted linear checkpoint digest mismatch")
    loaded = np.load(path, allow_pickle=False)
    loaded_shape = cast(tuple[int, ...], loaded.shape)
    if loaded_shape != file.shape:
        raise ValueError("persisted linear checkpoint shape mismatch")
    if str(loaded.dtype) != file.dtype:
        raise ValueError("persisted linear checkpoint dtype mismatch")
    copied = cast(NDArray[np.float32], np.array(loaded, copy=True))
    return _torch_from_numpy(copied)


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
