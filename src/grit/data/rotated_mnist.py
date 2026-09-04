"""Deterministic RotatedMNIST construction with exact training-source pairs."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, StrictInt, StrictStr, model_validator

from grit.data.cmnist import MnistPool
from grit.data.views import (
    ExampleIdentity,
    TrainingSplitDescriptor,
    TrainingView,
    ValidationSplitDescriptor,
    ValidationView,
    validate_rotated_mnist_repeated_validation_views,
)
from grit.schemas import StrictBoundaryModel, canonical_digest_value

PARTITION_METHOD_ID = "rotated-mnist-stratified-hash-v1"
RENDERING_METHOD_ID = "rotated-mnist-bilinear-rgb-v1"
ORACLE_PAIR_METHOD_ID = "rotated-mnist-exact-source-oracle-pairs-v1"

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
SourcePartitionName: TypeAlias = Literal[
    "train_r0_sources", "train_r45_sources", "validation_sources", "test_sources"
]
EnvironmentName: TypeAlias = Literal[
    "train_r0", "train_r45", "val_r0", "val_r45", "val_r60", "test_r90"
]


class RotatedMnistPartitionTargets(StrictBoundaryModel):
    train_r0: NonNegativeInt
    train_r45: NonNegativeInt
    validation: NonNegativeInt
    test: NonNegativeInt

    @model_validator(mode="after")
    def _require_nonempty(self) -> RotatedMnistPartitionTargets:
        if min(self.train_r0, self.train_r45, self.validation, self.test) <= 0:
            raise ValueError("every RotatedMNIST source partition must be non-empty")
        return self

    @property
    def official_train_count(self) -> int:
        return self.train_r0 + self.train_r45 + self.validation


PRODUCTION_PARTITION_TARGETS = RotatedMnistPartitionTargets(
    train_r0=25_000,
    train_r45=25_000,
    validation=10_000,
    test=10_000,
)


class DigitCount(StrictBoundaryModel):
    digit: Annotated[StrictInt, Field(ge=0, le=9)]
    count: NonNegativeInt


class RotatedMnistPartitionRecord(StrictBoundaryModel):
    name: SourcePartitionName
    official_split: Literal["train", "test"]
    source_indices: tuple[NonNegativeInt, ...]
    digit_counts: tuple[DigitCount, ...]
    membership_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_membership(self) -> RotatedMnistPartitionRecord:
        if len(self.source_indices) != len(set(self.source_indices)):
            raise ValueError("source partition indices must be unique")
        if tuple(item.digit for item in self.digit_counts) != tuple(range(10)):
            raise ValueError("digit counts must be ordered 0 through 9")
        if sum(item.count for item in self.digit_counts) != len(self.source_indices):
            raise ValueError("digit counts must sum to source partition size")
        expected = canonical_digest_value(
            {
                "name": self.name,
                "official_split": self.official_split,
                "source_indices": self.source_indices,
            }
        )
        if self.membership_digest != expected:
            raise ValueError("source partition membership digest is inconsistent")
        return self


class RotatedMnistPartitionManifest(StrictBoundaryModel):
    schema_version: Literal["grit.rotated-mnist-partitions/v1"]
    dataset_id: Literal["rotated_mnist"]
    construction_method_id: Literal["rotated-mnist-stratified-hash-v1"]
    construction_seed: StrictInt
    targets: RotatedMnistPartitionTargets
    partitions: tuple[RotatedMnistPartitionRecord, ...]

    @model_validator(mode="after")
    def _validate_partitions(self) -> RotatedMnistPartitionManifest:
        expected_names = (
            "train_r0_sources",
            "train_r45_sources",
            "validation_sources",
            "test_sources",
        )
        if tuple(item.name for item in self.partitions) != expected_names:
            raise ValueError("RotatedMNIST partitions must use canonical ordering")
        train_sets = [set(item.source_indices) for item in self.partitions[:3]]
        if any(
            train_sets[left] & train_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("official-train source partitions must be disjoint")
        expected_counts = (
            self.targets.train_r0,
            self.targets.train_r45,
            self.targets.validation,
            self.targets.test,
        )
        if (
            tuple(len(item.source_indices) for item in self.partitions)
            != expected_counts
        ):
            raise ValueError("source partition counts do not match targets")
        if any(item.official_split != "train" for item in self.partitions[:3]):
            raise ValueError("training and validation must use official MNIST train")
        if self.partitions[3].official_split != "test":
            raise ValueError("final sources must use official MNIST test")
        return self


@dataclass(frozen=True, slots=True)
class RotatedMnistPartitions:
    train_r0_source_indices: tuple[int, ...]
    train_r45_source_indices: tuple[int, ...]
    validation_source_indices: tuple[int, ...]
    test_source_indices: tuple[int, ...]
    manifest: RotatedMnistPartitionManifest


class RotatedMnistEnvironmentSpec(StrictBoundaryModel):
    name: EnvironmentName
    role: Literal["training", "validation", "final_test"]
    source_partition_id: SourcePartitionName
    angle_degrees: Literal[0, 45, 60, 90]


ROTATED_MNIST_ENVIRONMENT_SPECS: tuple[RotatedMnistEnvironmentSpec, ...] = (
    RotatedMnistEnvironmentSpec(
        name="train_r0",
        role="training",
        source_partition_id="train_r0_sources",
        angle_degrees=0,
    ),
    RotatedMnistEnvironmentSpec(
        name="train_r45",
        role="training",
        source_partition_id="train_r45_sources",
        angle_degrees=45,
    ),
    RotatedMnistEnvironmentSpec(
        name="val_r0",
        role="validation",
        source_partition_id="validation_sources",
        angle_degrees=0,
    ),
    RotatedMnistEnvironmentSpec(
        name="val_r45",
        role="validation",
        source_partition_id="validation_sources",
        angle_degrees=45,
    ),
    RotatedMnistEnvironmentSpec(
        name="val_r60",
        role="validation",
        source_partition_id="validation_sources",
        angle_degrees=60,
    ),
    RotatedMnistEnvironmentSpec(
        name="test_r90",
        role="final_test",
        source_partition_id="test_sources",
        angle_degrees=90,
    ),
)


class RotatedMnistEnvironmentManifest(StrictBoundaryModel):
    name: EnvironmentName
    role: Literal["training", "validation", "final_test"]
    source_partition_id: SourcePartitionName
    angle_degrees: Literal[0, 45, 60, 90]
    count: NonNegativeInt
    source_membership_digest: NonEmptyStr
    record_digest: NonEmptyStr


class RotatedMnistDatasetManifest(StrictBoundaryModel):
    schema_version: Literal["grit.rotated-mnist-dataset/v1"]
    dataset_id: Literal["rotated_mnist"]
    partition_manifest: RotatedMnistPartitionManifest
    partition_manifest_digest: NonEmptyStr
    official_train_pool_digest: NonEmptyStr
    official_test_pool_digest: NonEmptyStr
    rendering_method_id: Literal["rotated-mnist-bilinear-rgb-v1"]
    construction_seed: StrictInt
    environments: tuple[RotatedMnistEnvironmentManifest, ...]

    @model_validator(mode="after")
    def _validate_identity(self) -> RotatedMnistDatasetManifest:
        if self.partition_manifest_digest != self.partition_manifest.canonical_digest():
            raise ValueError("partition manifest digest is inconsistent")
        if self.construction_seed != self.partition_manifest.construction_seed:
            raise ValueError("dataset and partition construction seeds must match")
        expected = tuple(item.name for item in ROTATED_MNIST_ENVIRONMENT_SPECS)
        if tuple(item.name for item in self.environments) != expected:
            raise ValueError("dataset environments must use canonical ordering")
        return self


@dataclass(frozen=True, slots=True)
class RenderedRotatedMnistTable:
    name: EnvironmentName | str
    role: Literal["training", "validation", "final_test", "pair_projection"]
    source_partition_id: SourcePartitionName | Literal["training_sources"]
    source_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    images: torch.Tensor
    targets: torch.Tensor
    angles: torch.Tensor


@dataclass(frozen=True, slots=True)
class RotatedMnistConstruction:
    train_r0: RenderedRotatedMnistTable
    train_r45: RenderedRotatedMnistTable
    val_r0: RenderedRotatedMnistTable
    val_r45: RenderedRotatedMnistTable
    val_r60: RenderedRotatedMnistTable
    _test_r90: RenderedRotatedMnistTable
    partitions: RotatedMnistPartitions
    manifest: RotatedMnistDatasetManifest

    def training_views(self) -> tuple[TrainingView, TrainingView]:
        manifest_id = self.manifest.canonical_digest()
        tables = (self.train_r0, self.train_r45)
        views = tuple(
            TrainingView(
                descriptor=TrainingSplitDescriptor(
                    dataset_id="rotated_mnist",
                    manifest_id=manifest_id,
                    name=table.name,
                    role="training",
                    source_partition_id=table.source_partition_id,
                    view_id=table.name,
                ),
                examples=_example_identities(table),
            )
            for table in tables
        )
        return views[0], views[1]

    def validation_views(self) -> tuple[ValidationView, ValidationView, ValidationView]:
        manifest_id = self.manifest.canonical_digest()
        tables = (self.val_r0, self.val_r45, self.val_r60)
        views = tuple(
            ValidationView(
                descriptor=ValidationSplitDescriptor(
                    dataset_id="rotated_mnist",
                    manifest_id=manifest_id,
                    name=table.name,
                    role="validation",
                    source_partition_id="validation_sources",
                    view_id=table.name,
                ),
                examples=_example_identities(table),
            )
            for table in tables
        )
        typed = views[0], views[1], views[2]
        validate_rotated_mnist_repeated_validation_views(typed)
        return typed

    def preparation_tables(self) -> tuple[RenderedRotatedMnistTable, ...]:
        return (
            self.train_r0,
            self.train_r45,
            self.val_r0,
            self.val_r45,
            self.val_r60,
            self._test_r90,
        )


class RotatedMnistOraclePairRecord(StrictBoundaryModel):
    pair_id: NonEmptyStr
    source_id: NonEmptyStr
    source_partition_id: Literal["train_r0_sources", "train_r45_sources"]
    official_source_index: NonNegativeInt
    digit: Annotated[StrictInt, Field(ge=0, le=9)]
    left_angle_degrees: Literal[0]
    right_angle_degrees: Literal[45]


class RotatedMnistOraclePairManifest(StrictBoundaryModel):
    schema_version: Literal["grit.rotated-mnist-oracle-pairs/v1"]
    dataset_manifest_digest: NonEmptyStr
    construction_method_id: Literal["rotated-mnist-exact-source-oracle-pairs-v1"]
    pair_seed: StrictInt
    requested_count: NonNegativeInt
    realized_count: NonNegativeInt
    source_partition_ids: tuple[
        Literal["train_r0_sources"], Literal["train_r45_sources"]
    ]
    orientation: Literal["rotation_0_minus_45"]
    records: tuple[RotatedMnistOraclePairRecord, ...]
    membership_digest: NonEmptyStr

    @model_validator(mode="after")
    def _validate_pairs(self) -> RotatedMnistOraclePairManifest:
        if self.source_partition_ids != (
            "train_r0_sources",
            "train_r45_sources",
        ):
            raise ValueError("oracle pairs require both training source partitions")
        if self.requested_count != self.realized_count or self.realized_count != len(
            self.records
        ):
            raise ValueError("oracle pair count is inconsistent")
        source_ids = tuple(record.source_id for record in self.records)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("oracle pair sources must be unique")
        if self.membership_digest != canonical_digest_value(source_ids):
            raise ValueError("oracle pair membership digest is inconsistent")
        for record in self.records:
            expected_pair_id = canonical_digest_value(
                {
                    "dataset_manifest_digest": self.dataset_manifest_digest,
                    "method": self.construction_method_id,
                    "pair_seed": self.pair_seed,
                    "source_id": record.source_id,
                    "orientation": self.orientation,
                }
            )
            if record.pair_id != expected_pair_id:
                raise ValueError("oracle pair identity is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class RotatedMnistOraclePairSet:
    left_r0: RenderedRotatedMnistTable
    right_r45: RenderedRotatedMnistTable
    records: tuple[RotatedMnistOraclePairRecord, ...]
    manifest: RotatedMnistOraclePairManifest


def partition_rotated_mnist_sources(
    train_pool: MnistPool,
    test_pool: MnistPool,
    *,
    construction_seed: int,
    targets: RotatedMnistPartitionTargets = PRODUCTION_PARTITION_TARGETS,
) -> RotatedMnistPartitions:
    """Partition source identities before rendering any rotation."""

    train = _validated_pool(train_pool, "train", targets.official_train_count)
    test = _validated_pool(test_pool, "test", targets.test)
    if targets == PRODUCTION_PARTITION_TARGETS:
        if set(_tensor_ints(train.source_indices)) != set(range(60_000)):
            raise ValueError("official MNIST train indices must be 0 through 59999")
        if set(_tensor_ints(test.source_indices)) != set(range(10_000)):
            raise ValueError("official MNIST test indices must be 0 through 9999")

    train_indices = _tensor_ints(train.source_indices)
    train_digits = _tensor_ints(train.digits)
    by_digit: dict[int, list[int]] = {digit: [] for digit in range(10)}
    digit_by_index: dict[int, int] = {}
    for source_index, digit in zip(train_indices, train_digits, strict=True):
        digit_by_index[source_index] = digit
        by_digit[digit].append(source_index)
    for digit in range(10):
        by_digit[digit].sort(
            key=lambda source_index: (
                _membership_hash(construction_seed, "train", source_index),
                source_index,
            )
        )
    original = {digit: len(by_digit[digit]) for digit in range(10)}
    validation_counts = _largest_remainder(original, targets.validation)
    remaining = {
        digit: original[digit] - validation_counts[digit] for digit in range(10)
    }
    train_r0_counts = _largest_remainder(remaining, targets.train_r0)
    validation_indices: list[int] = []
    train_r0_indices: list[int] = []
    train_r45_indices: list[int] = []
    for digit in range(10):
        ordered = by_digit[digit]
        validation_stop = validation_counts[digit]
        train_r0_stop = validation_stop + train_r0_counts[digit]
        validation_indices.extend(ordered[:validation_stop])
        train_r0_indices.extend(ordered[validation_stop:train_r0_stop])
        train_r45_indices.extend(ordered[train_r0_stop:])
    memberships = (
        tuple(sorted(train_r0_indices)),
        tuple(sorted(train_r45_indices)),
        tuple(sorted(validation_indices)),
        tuple(sorted(_tensor_ints(test.source_indices))),
    )
    test_digit_by_index = dict(
        zip(_tensor_ints(test.source_indices), _tensor_ints(test.digits), strict=True)
    )
    names: tuple[SourcePartitionName, ...] = (
        "train_r0_sources",
        "train_r45_sources",
        "validation_sources",
        "test_sources",
    )
    splits: tuple[Literal["train", "test"], ...] = (
        "train",
        "train",
        "train",
        "test",
    )
    digit_maps = (digit_by_index, digit_by_index, digit_by_index, test_digit_by_index)
    records = tuple(
        _partition_record(name, split, membership, digit_map)
        for name, split, membership, digit_map in zip(
            names, splits, memberships, digit_maps, strict=True
        )
    )
    manifest = RotatedMnistPartitionManifest(
        schema_version="grit.rotated-mnist-partitions/v1",
        dataset_id="rotated_mnist",
        construction_method_id=PARTITION_METHOD_ID,
        construction_seed=construction_seed,
        targets=targets,
        partitions=records,
    )
    return RotatedMnistPartitions(
        train_r0_source_indices=memberships[0],
        train_r45_source_indices=memberships[1],
        validation_source_indices=memberships[2],
        test_source_indices=memberships[3],
        manifest=manifest,
    )


def construct_rotated_mnist(
    train_pool: MnistPool,
    test_pool: MnistPool,
    *,
    construction_seed: int,
    targets: RotatedMnistPartitionTargets = PRODUCTION_PARTITION_TARGETS,
) -> RotatedMnistConstruction:
    train = _validated_pool(train_pool, "train", targets.official_train_count)
    test = _validated_pool(test_pool, "test", targets.test)
    partitions = partition_rotated_mnist_sources(
        train, test, construction_seed=construction_seed, targets=targets
    )
    pools = {"train": train, "test": test}
    memberships: dict[SourcePartitionName, tuple[int, ...]] = {
        "train_r0_sources": partitions.train_r0_source_indices,
        "train_r45_sources": partitions.train_r45_source_indices,
        "validation_sources": partitions.validation_source_indices,
        "test_sources": partitions.test_source_indices,
    }
    split_by_partition: dict[SourcePartitionName, Literal["train", "test"]] = {
        "train_r0_sources": "train",
        "train_r45_sources": "train",
        "validation_sources": "train",
        "test_sources": "test",
    }
    tables: dict[str, RenderedRotatedMnistTable] = {}
    for spec in ROTATED_MNIST_ENVIRONMENT_SPECS:
        pool = pools[split_by_partition[spec.source_partition_id]]
        tables[spec.name] = _render_environment(
            pool, memberships[spec.source_partition_id], spec
        )
    environments = tuple(
        _environment_manifest(tables[spec.name], spec)
        for spec in ROTATED_MNIST_ENVIRONMENT_SPECS
    )
    manifest = RotatedMnistDatasetManifest(
        schema_version="grit.rotated-mnist-dataset/v1",
        dataset_id="rotated_mnist",
        partition_manifest=partitions.manifest,
        partition_manifest_digest=partitions.manifest.canonical_digest(),
        official_train_pool_digest=_pool_digest(train),
        official_test_pool_digest=_pool_digest(test),
        rendering_method_id=RENDERING_METHOD_ID,
        construction_seed=construction_seed,
        environments=environments,
    )
    return RotatedMnistConstruction(
        train_r0=tables["train_r0"],
        train_r45=tables["train_r45"],
        val_r0=tables["val_r0"],
        val_r45=tables["val_r45"],
        val_r60=tables["val_r60"],
        _test_r90=tables["test_r90"],
        partitions=partitions,
        manifest=manifest,
    )


def build_rotated_mnist_oracle_pairs(
    construction: RotatedMnistConstruction,
    train_pool: MnistPool,
    *,
    pair_seed: int,
    pair_count: int = 256,
) -> RotatedMnistOraclePairSet:
    """Render both endpoints from each selected exact training source."""

    manifest = RotatedMnistDatasetManifest.model_validate_json(
        construction.manifest.canonical_json()
    )
    train = _validated_pool(
        train_pool, "train", manifest.partition_manifest.targets.official_train_count
    )
    if _pool_digest(train) != manifest.official_train_pool_digest:
        raise ValueError("pair-source pool does not match dataset manifest")
    support = (
        construction.partitions.train_r0_source_indices
        + construction.partitions.train_r45_source_indices
    )
    if pair_count <= 0 or pair_count > len(support):
        raise ValueError("pair_count must fit the training-source support")
    selected = tuple(
        sorted(
            support,
            key=lambda index: (
                _pair_hash(pair_seed, _source_id("train", index)),
                index,
            ),
        )[:pair_count]
    )
    r0_set = set(construction.partitions.train_r0_source_indices)
    positions = _pool_positions(train)
    rows = torch.tensor([positions[index] for index in selected], dtype=torch.int64)
    gray = _normalized_images(train)[rows]
    targets = train.digits.detach().cpu().to(torch.int64)[rows]
    source_ids = tuple(_source_id("train", index) for index in selected)
    dataset_digest = manifest.canonical_digest()
    records = tuple(
        RotatedMnistOraclePairRecord(
            pair_id=canonical_digest_value(
                {
                    "dataset_manifest_digest": dataset_digest,
                    "method": ORACLE_PAIR_METHOD_ID,
                    "pair_seed": pair_seed,
                    "source_id": source_id,
                    "orientation": "rotation_0_minus_45",
                }
            ),
            source_id=source_id,
            source_partition_id=(
                "train_r0_sources" if source_index in r0_set else "train_r45_sources"
            ),
            official_source_index=source_index,
            digit=int(target),
            left_angle_degrees=0,
            right_angle_degrees=45,
        )
        for source_index, source_id, target in zip(
            selected, source_ids, _tensor_ints(targets), strict=True
        )
    )
    pair_manifest = RotatedMnistOraclePairManifest(
        schema_version="grit.rotated-mnist-oracle-pairs/v1",
        dataset_manifest_digest=dataset_digest,
        construction_method_id=ORACLE_PAIR_METHOD_ID,
        pair_seed=pair_seed,
        requested_count=pair_count,
        realized_count=pair_count,
        source_partition_ids=("train_r0_sources", "train_r45_sources"),
        orientation="rotation_0_minus_45",
        records=records,
        membership_digest=canonical_digest_value(source_ids),
    )
    left = _pair_table("oracle_pair_r0", source_ids, targets, gray, 0, records)
    right = _pair_table("oracle_pair_r45", source_ids, targets, gray, 45, records)
    return RotatedMnistOraclePairSet(
        left_r0=left,
        right_r45=right,
        records=records,
        manifest=pair_manifest,
    )


def _pair_table(
    name: str,
    source_ids: tuple[str, ...],
    targets: torch.Tensor,
    gray: torch.Tensor,
    angle: Literal[0, 45],
    records: tuple[RotatedMnistOraclePairRecord, ...],
) -> RenderedRotatedMnistTable:
    return RenderedRotatedMnistTable(
        name=name,
        role="pair_projection",
        source_partition_id="training_sources",
        source_ids=source_ids,
        example_ids=tuple(f"{source_id}:pair:r{angle}" for source_id in source_ids),
        images=_render_rgb(gray, angle),
        targets=targets,
        angles=torch.full((len(source_ids),), angle, dtype=torch.int64),
    )


def _render_environment(
    pool: MnistPool,
    source_indices: tuple[int, ...],
    spec: RotatedMnistEnvironmentSpec,
) -> RenderedRotatedMnistTable:
    positions = _pool_positions(pool)
    rows = torch.tensor(
        [positions[index] for index in source_indices], dtype=torch.int64
    )
    gray = _normalized_images(pool)[rows]
    targets = pool.digits.detach().cpu().to(torch.int64)[rows]
    source_ids = tuple(
        _source_id(pool.official_split, source_index) for source_index in source_indices
    )
    return RenderedRotatedMnistTable(
        name=spec.name,
        role=spec.role,
        source_partition_id=spec.source_partition_id,
        source_ids=source_ids,
        example_ids=tuple(f"{source_id}:view:{spec.name}" for source_id in source_ids),
        images=_render_rgb(gray, spec.angle_degrees),
        targets=targets,
        angles=torch.full(
            (len(source_indices),), spec.angle_degrees, dtype=torch.int64
        ),
    )


class _Rotate(Protocol):
    def __call__(
        self,
        image: torch.Tensor,
        angle: float,
        interpolation: object,
        fill: float,
    ) -> torch.Tensor: ...


_functional = importlib.import_module("torchvision.transforms.functional")
_interpolation = importlib.import_module("torchvision.transforms").InterpolationMode
_rotate = cast(_Rotate, _functional.rotate)


def _render_rgb(gray: torch.Tensor, angle: int) -> torch.Tensor:
    images = gray.detach().cpu().to(torch.float32).unsqueeze(1)
    if angle != 0:
        images = _rotate(
            images,
            float(angle),
            interpolation=_interpolation.BILINEAR,
            fill=0.0,
        )
    return images.repeat(1, 3, 1, 1).contiguous()


def _environment_manifest(
    table: RenderedRotatedMnistTable, spec: RotatedMnistEnvironmentSpec
) -> RotatedMnistEnvironmentManifest:
    return RotatedMnistEnvironmentManifest(
        name=spec.name,
        role=spec.role,
        source_partition_id=spec.source_partition_id,
        angle_degrees=spec.angle_degrees,
        count=len(table.source_ids),
        source_membership_digest=canonical_digest_value(table.source_ids),
        record_digest=canonical_digest_value(
            {
                "source_ids": table.source_ids,
                "targets": _tensor_ints(table.targets),
                "angles": _tensor_ints(table.angles),
            }
        ),
    )


def _example_identities(
    table: RenderedRotatedMnistTable,
) -> tuple[ExampleIdentity, ...]:
    return tuple(
        ExampleIdentity(example_id=example_id, source_id=source_id, view_id=table.name)
        for example_id, source_id in zip(
            table.example_ids, table.source_ids, strict=True
        )
    )


def _partition_record(
    name: SourcePartitionName,
    split: Literal["train", "test"],
    membership: tuple[int, ...],
    digit_by_index: dict[int, int],
) -> RotatedMnistPartitionRecord:
    return RotatedMnistPartitionRecord(
        name=name,
        official_split=split,
        source_indices=membership,
        digit_counts=tuple(
            DigitCount(
                digit=digit,
                count=sum(digit_by_index[index] == digit for index in membership),
            )
            for digit in range(10)
        ),
        membership_digest=canonical_digest_value(
            {
                "name": name,
                "official_split": split,
                "source_indices": membership,
            }
        ),
    )


def _validated_pool(
    pool: MnistPool, expected_split: Literal["train", "test"], expected_count: int
) -> MnistPool:
    if pool.official_split != expected_split:
        raise ValueError(f"expected official MNIST {expected_split!r} pool")
    if pool.source_indices.ndim != 1 or pool.digits.ndim != 1 or pool.images.ndim != 3:
        raise ValueError("MNIST pool arrays have invalid dimensions")
    if (
        int(pool.source_indices.shape[0]) != expected_count
        or int(pool.images.shape[0]) != expected_count
        or int(pool.digits.shape[0]) != expected_count
    ):
        raise ValueError(f"official MNIST {expected_split} pool has the wrong count")
    indices = _tensor_ints(pool.source_indices)
    if len(indices) != len(set(indices)):
        raise ValueError("official MNIST source indices must be unique")
    digits = _tensor_ints(pool.digits)
    if any(digit < 0 or digit > 9 for digit in digits):
        raise ValueError("MNIST digit labels must lie in 0 through 9")
    _ = _normalized_images(pool)
    return pool


def _normalized_images(pool: MnistPool) -> torch.Tensor:
    images = pool.images.detach().cpu()
    if images.dtype == torch.uint8:
        return images.to(torch.float32).div(255.0)
    result = images.to(torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("MNIST images must be finite")
    if float(result.min()) < 0.0 or float(result.max()) > 1.0:
        raise ValueError("floating-point MNIST images must lie in [0, 1]")
    return result


def _largest_remainder(counts: dict[int, int], requested: int) -> dict[int, int]:
    total = sum(counts.values())
    if requested < 0 or requested > total:
        raise ValueError("apportionment request exceeds available sources")
    allocated = {digit: (counts[digit] * requested) // total for digit in range(10)}
    remainders = {digit: (counts[digit] * requested) % total for digit in range(10)}
    remaining = requested - sum(allocated.values())
    for digit in sorted(range(10), key=lambda item: (-remainders[item], item))[
        :remaining
    ]:
        allocated[digit] += 1
    return allocated


def _pool_positions(pool: MnistPool) -> dict[int, int]:
    return {
        source_index: row
        for row, source_index in enumerate(_tensor_ints(pool.source_indices))
    }


def _membership_hash(seed: int, split: str, source_index: int) -> bytes:
    payload = f"{PARTITION_METHOD_ID}\0{seed}\0{split}\0{source_index}".encode()
    return hashlib.sha256(payload).digest()


def _pair_hash(seed: int, source_id: str) -> bytes:
    payload = f"{ORACLE_PAIR_METHOD_ID}\0{seed}\0{source_id}".encode()
    return hashlib.sha256(payload).digest()


def _source_id(split: str, source_index: int) -> str:
    return f"mnist:{split}:{source_index}"


def _tensor_ints(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values.detach().cpu().reshape(-1))


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


def _pool_digest(pool: MnistPool) -> str:
    positions = _pool_positions(pool)
    ordered = tuple(sorted(positions))
    rows = torch.tensor([positions[index] for index in ordered], dtype=torch.int64)
    digest = hashlib.sha256(pool.official_split.encode("utf-8"))
    for tensor in (
        pool.source_indices.detach().cpu()[rows],
        pool.digits.detach().cpu()[rows],
        pool.images.detach().cpu()[rows],
    ):
        array = cast(_TensorToNumpy, tensor.contiguous().numpy)()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"
