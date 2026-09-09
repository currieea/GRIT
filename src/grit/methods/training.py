"""Shared frozen-feature linear-probe methods, training, and checkpoints."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import StrictInt, StrictStr

from grit.config import LinearProbeTrainingConfig
from grit.methods.checkpoints import CheckpointStore, StoredCheckpoint
from grit.methods.groupdro import (
    GroupDroObjective,
    group_balanced_epoch_indices,
)
from grit.methods.invariance import (
    annealed_penalty_weight,
    environment_balanced_epoch_batches,
    irmv1_objective,
    vrex_objective,
)
from grit.methods.projection import FittedLinearProjection
from grit.methods.training_state import LinearProbeState
from grit.methods.types import MethodId
from grit.schemas import SeedStage, StrictBoundaryModel
from grit.selection.cmnist import (
    CheckpointIdentity,
    ValidationMetricRecord,
    ValidationSplitName,
)

__all__ = ["LinearProbeState"]


class LinearProbeAlgorithm:
    """Own a fixed-class linear model, Adam, and one bounded update operation."""

    def __init__(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        projection: FittedLinearProjection | None,
        num_classes: int = 2,
    ) -> None:
        if num_classes <= 1:
            raise ValueError("linear probe requires at least two output classes")
        self._model = torch.nn.Linear(512, num_classes, bias=True, device="cpu")
        generator = torch.Generator(device="cpu").manual_seed(model_seed)
        bound = 1.0 / math.sqrt(512)
        weight = (
            torch.rand((num_classes, 512), generator=generator, dtype=torch.float32)
            .mul(2 * bound)
            .sub(bound)
        )
        bias = (
            torch.rand((num_classes,), generator=generator, dtype=torch.float32)
            .mul(2 * bound)
            .sub(bound)
        )
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

        return self.update_with_objective(
            features, targets, lambda losses: losses.mean()
        )

    def update_with_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        """Update from an algorithm-specific reduction of per-example losses."""

        def logits_objective(
            logits: torch.Tensor, prepared_targets: torch.Tensor
        ) -> torch.Tensor:
            per_example = torch.nn.functional.cross_entropy(
                logits, prepared_targets, reduction="none"
            )
            return objective(per_example)

        return self.update_with_logits_objective(
            features,
            targets,
            logits_objective,
        )

    def update_with_logits_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        """Update from an objective that needs logits and aligned targets."""

        self._model.train()
        prepared = self._prepare(features)
        prepared_targets = targets.detach().cpu().to(torch.int64)
        self._optimizer.zero_grad(set_to_none=True)
        logits = self._model(prepared)
        loss = objective(logits, prepared_targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("linear-probe objective must return one finite scalar")
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

    def mean_loss(self, features: torch.Tensor, targets: torch.Tensor) -> float:
        """Side-effect-free mean cross-entropy of the current inference state."""

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.inference_mode():
                logits = self._model(self._prepare(features))
                loss = torch.nn.functional.cross_entropy(
                    logits, targets.detach().cpu().to(torch.int64)
                )
                return float(loss.item())
        finally:
            self._model.train(was_training)

    def capture_inference_state(self) -> LinearProbeState:
        return LinearProbeState(
            weight=self._model.weight.detach().cpu().clone(),
            bias=self._model.bias.detach().cpu().clone(),
        )

    def restore_inference_state(self, state: LinearProbeState) -> None:
        if (
            state.weight.shape != self._model.weight.shape
            or state.bias.shape != self._model.bias.shape
        ):
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


class LinearProbeTrainingMethod(Protocol):
    """One method's batching, objective, and projection for a linear-probe run."""

    @property
    def method_id(self) -> MethodId: ...

    @property
    def projection(self) -> FittedLinearProjection | None: ...

    @property
    def projection_rank(self) -> int | None: ...

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]: ...

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float: ...

    def build_algorithm(
        self,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm: ...

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        """The state saved and validated at an epoch boundary."""
        ...


class _HasProjection(Protocol):
    @property
    def projection(self) -> FittedLinearProjection | None: ...


class LinearProbeMethodDefaults:
    """Plain linear probe over the method's projection; live weights are checkpoints."""

    def build_algorithm(
        self: _HasProjection,
        config: LinearProbeTrainingConfig,
        *,
        model_seed: int,
        num_classes: int,
    ) -> LinearProbeAlgorithm:
        return LinearProbeAlgorithm(
            config,
            model_seed=model_seed,
            projection=self.projection,
            num_classes=num_classes,
        )

    def checkpoint_state(self, algorithm: LinearProbeAlgorithm) -> LinearProbeState:
        return algorithm.capture_inference_state()


class FeatureTableLike(Protocol):
    """Minimal frozen-feature table consumed by shared training/evaluation."""

    @property
    def name(self) -> str: ...

    @property
    def source_ids(self) -> tuple[str, ...]: ...

    @property
    def features(self) -> torch.Tensor: ...

    @property
    def targets(self) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class OrdinaryLinearProbeMethod(LinearProbeMethodDefaults):
    """ERM-style minibatches, optionally preceded by a frozen GRIT projection."""

    method_id: MethodId
    projection: FittedLinearProjection | None
    projection_rank: int | None

    def __post_init__(self) -> None:
        if self.method_id not in ("erm", "grit"):
            raise ValueError("ordinary linear-probe updates support only ERM and GRIT")
        projected = self.projection is not None and self.projection_rank is not None
        if self.method_id == "erm" and (
            self.projection is not None or self.projection_rank is not None
        ):
            raise ValueError("ERM cannot use a projection")
        if self.method_id == "grit" and not projected:
            raise ValueError("GRIT requires a fitted projection and rank")

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


@dataclass(slots=True)
class GroupDroLinearProbeMethod(LinearProbeMethodDefaults):
    """Group-balanced epochs and the stateful GroupDRO robust objective."""

    group_ids: torch.Tensor
    group_count: int
    step_size: float
    method_id: Literal["groupdro"] = field(init=False, default="groupdro")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _objective: GroupDroObjective = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.group_ids = self.group_ids.detach().cpu().to(torch.int64)
        self._objective = GroupDroObjective(
            group_count=self.group_count,
            step_size=self.step_size,
        )

    def epoch_batches(
        self,
        *,
        row_count: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        require_aligned_ids(self.group_ids, row_count, "group")
        permutation = group_balanced_epoch_indices(
            self.group_ids,
            group_count=self.group_count,
            generator=generator,
        )
        return batch_indices(permutation, batch_size)

    def update(
        self,
        algorithm: LinearProbeAlgorithm,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor,
    ) -> float:
        batch_groups = self.group_ids[rows]
        return algorithm.update_with_objective(
            features[rows],
            targets[rows],
            lambda losses: self._objective(losses, batch_groups),
        )


@dataclass(slots=True)
class RexLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs and an annealed V-REx objective."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    method_id: Literal["rex"] = field(init=False, default="rex")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _update_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
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
        penalty_weight = annealed_penalty_weight(
            self.penalty_weight,
            anneal_updates=self.penalty_anneal_updates,
            update_count=self._update_count,
        )
        batch_environments = self.environment_ids[rows]
        loss = algorithm.update_with_objective(
            features[rows],
            targets[rows],
            partial(
                vrex_objective,
                environment_ids=batch_environments,
                environment_count=self.environment_count,
                penalty_weight=penalty_weight,
            ),
        )
        self._update_count += 1
        return loss


@dataclass(slots=True)
class IrmLinearProbeMethod(LinearProbeMethodDefaults):
    """Environment-balanced epochs and an annealed IRMv1 objective."""

    environment_ids: torch.Tensor
    environment_count: int
    penalty_weight: float
    penalty_anneal_updates: int
    method_id: Literal["irm"] = field(init=False, default="irm")
    projection: None = field(init=False, default=None)
    projection_rank: None = field(init=False, default=None)
    _update_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
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
        penalty_weight = annealed_penalty_weight(
            self.penalty_weight,
            anneal_updates=self.penalty_anneal_updates,
            update_count=self._update_count,
        )
        batch_environments = self.environment_ids[rows]
        loss = algorithm.update_with_logits_objective(
            features[rows],
            targets[rows],
            partial(
                irmv1_objective,
                environment_ids=batch_environments,
                environment_count=self.environment_count,
                penalty_weight=penalty_weight,
            ),
        )
        self._update_count += 1
        return loss


def random_epoch_batches(
    row_count: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    return batch_indices(torch.randperm(row_count, generator=generator), batch_size)


def batch_indices(
    permutation: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, ...]:
    if batch_size <= 0:
        raise ValueError("linear-probe batch size must be positive")
    return tuple(
        permutation[start : start + batch_size]
        for start in range(0, len(permutation), batch_size)
    )


def require_aligned_ids(ids: torch.Tensor, row_count: int, kind: str) -> None:
    if ids.ndim != 1 or int(ids.shape[0]) != row_count:
        raise ValueError(f"training-{kind} IDs must align with training rows")


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


@dataclass(frozen=True, slots=True)
class TrainedLinearProbeCore:
    """Dataset-neutral output of the shared linear-probe epoch lifecycle."""

    store: InMemoryLinearCheckpointStore
    algorithm: LinearProbeAlgorithm
    epoch_losses: tuple[float, ...]


def train_linear_probe_epochs(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    scientific_config_digest: str,
    seed: int,
    method: LinearProbeTrainingMethod,
    validate_epoch: Callable[[LinearProbeAlgorithm, CheckpointIdentity], None],
    num_classes: int = 2,
) -> TrainedLinearProbeCore:
    """Train one method and hand each saved epoch to dataset-specific validation."""

    features = train_features.detach().cpu().to(torch.float32)
    targets = train_targets.detach().cpu().to(torch.int64)
    if features.ndim != 2 or int(features.shape[0]) == 0:
        raise ValueError("linear probe training features must be a non-empty matrix")
    if targets.ndim != 1 or int(targets.shape[0]) != int(features.shape[0]):
        raise ValueError("linear probe training targets must align with features")
    torch.use_deterministic_algorithms(True)
    algorithm = method.build_algorithm(
        config, model_seed=seed, num_classes=num_classes
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    store = InMemoryLinearCheckpointStore(store_id=f"memory:{run_id}")
    epoch_losses: list[float] = []
    row_count = int(features.shape[0])
    for epoch in range(1, config.max_epochs + 1):
        row_batches = method.epoch_batches(
            row_count=row_count,
            batch_size=int(config.batch_size),
            generator=generator,
        )
        if not row_batches:
            raise ValueError("linear-probe method produced an empty epoch")
        batch_losses = tuple(
            method.update(algorithm, features, targets, rows) for rows in row_batches
        )
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        identity = CheckpointIdentity(
            checkpoint_id=f"checkpoint:{run_id}:epoch:{epoch}",
            candidate_id=candidate_id,
            run_id=run_id,
            scientific_config_digest=scientific_config_digest,
            epoch=epoch,
        )
        # Methods such as SWAD checkpoint a state other than the live parameters;
        # validate exactly what was stored, then hand the live state back.
        state = method.checkpoint_state(algorithm)
        store.save(identity, state)
        live = algorithm.capture_inference_state()
        algorithm.restore_inference_state(state)
        validate_epoch(algorithm, identity)
        algorithm.restore_inference_state(live)
    return TrainedLinearProbeCore(
        store=store,
        algorithm=algorithm,
        epoch_losses=tuple(epoch_losses),
    )


def train_linear_probe(
    training_tables: tuple[FeatureTableLike, FeatureTableLike],
    validation_tables: tuple[FeatureTableLike, FeatureTableLike, FeatureTableLike],
    config: LinearProbeTrainingConfig,
    *,
    run_id: str,
    candidate_id: str,
    scientific_config_digest: str,
    seed_stage: SeedStage,
    seed: int,
    method: LinearProbeTrainingMethod,
    num_classes: int = 2,
    epoch_hook: Callable[[LinearProbeAlgorithm, CheckpointIdentity], None]
    | None = None,
) -> TrainedLinearProbeRun:
    """Run trainer-owned epoch/batch iteration and emit validation every epoch.

    ``epoch_hook`` runs after each epoch's validation records are captured; the
    test-oracle track uses it to score ``test_ood`` at every saved checkpoint.
    """

    train_features = torch.cat([table.features for table in training_tables], dim=0).to(
        torch.float32
    )
    train_targets = torch.cat([table.targets for table in training_tables], dim=0).to(
        torch.int64
    )
    metrics: list[ValidationMetricRecord] = []

    def validate_epoch(
        algorithm: LinearProbeAlgorithm, identity: CheckpointIdentity
    ) -> None:
        metrics.extend(
            _validation_metrics(
                algorithm,
                validation_tables,
                run_id=run_id,
                candidate_id=candidate_id,
                method_id=method.method_id,
                scientific_config_digest=scientific_config_digest,
                seed_stage=seed_stage,
                seed=seed,
                checkpoint_id=identity.checkpoint_id,
                epoch=identity.epoch,
                projection_rank=method.projection_rank,
            )
        )
        if epoch_hook is not None:
            epoch_hook(algorithm, identity)

    core = train_linear_probe_epochs(
        train_features,
        train_targets,
        config,
        run_id=run_id,
        candidate_id=candidate_id,
        scientific_config_digest=scientific_config_digest,
        seed=seed,
        method=method,
        validate_epoch=validate_epoch,
        num_classes=num_classes,
    )
    return TrainedLinearProbeRun(
        run_id=run_id,
        candidate_id=candidate_id,
        method_id=method.method_id,
        seed_stage=seed_stage,
        seed=seed,
        projection_rank=method.projection_rank,
        validation_metrics=tuple(metrics),
        store=core.store,
        algorithm=core.algorithm,
        epoch_losses=core.epoch_losses,
    )


def evaluate_accuracy(
    algorithm: LinearProbeAlgorithm, table: FeatureTableLike
) -> float:
    predictions = algorithm.predict(table.features)
    return float((predictions == table.targets).to(torch.float64).mean().item())


def _validation_metrics(
    algorithm: LinearProbeAlgorithm,
    tables: tuple[FeatureTableLike, FeatureTableLike, FeatureTableLike],
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
            split_name=cast(ValidationSplitName, table.name),
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
