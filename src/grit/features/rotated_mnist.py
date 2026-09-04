"""Frozen-feature cache for the RotatedMNIST vertical slice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, StrictInt, StrictStr, model_validator

from grit.data.rotated_mnist import (
    RenderedRotatedMnistTable,
    RotatedMnistConstruction,
    RotatedMnistDatasetManifest,
    RotatedMnistOraclePairManifest,
    RotatedMnistOraclePairSet,
)
from grit.data.views import (
    ExampleIdentity,
    FinalTestHandle,
    FinalTestSplitDescriptor,
    FinalTestView,
    issue_final_test_handle,
)
from grit.features.cmnist import (
    FEATURE_DIMENSION,
    EncoderIdentity,
    FeatureExtractionRuntime,
    ImageEncoder,
)
from grit.schemas import StrictBoundaryModel, canonical_digest_value

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
Normalization: TypeAlias = Literal["none", "l2"]
TableRole: TypeAlias = Literal[
    "training", "validation", "final_test", "pair_projection"
]
ArrayField: TypeAlias = Literal["features", "targets", "angles"]


class RotatedMnistFeatureCacheValidationError(ValueError):
    """A cache is absent, damaged, or inconsistent with the protocol."""


class RotatedMnistArrayFileManifest(StrictBoundaryModel):
    field_name: ArrayField
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    shape: tuple[NonNegativeInt, ...]
    dtype: NonEmptyStr


class RotatedMnistFeatureTableManifest(StrictBoundaryModel):
    table_name: NonEmptyStr
    role: TableRole
    source_ids: tuple[NonEmptyStr, ...]
    source_ids_digest: NonEmptyStr
    row_count: NonNegativeInt
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    files: tuple[RotatedMnistArrayFileManifest, ...]

    @model_validator(mode="after")
    def _validate_table(self) -> RotatedMnistFeatureTableManifest:
        if len(self.source_ids) != self.row_count:
            raise ValueError("feature table source count is inconsistent")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("feature table source IDs must be unique")
        if self.source_ids_digest != canonical_digest_value(self.source_ids):
            raise ValueError("feature table source digest is inconsistent")
        if tuple(item.field_name for item in self.files) != (
            "features",
            "targets",
            "angles",
        ):
            raise ValueError("feature table files must use canonical ordering")
        if tuple(item.shape for item in self.files) != (
            (self.row_count, 512),
            (self.row_count,),
            (self.row_count,),
        ):
            raise ValueError("feature table array shapes are inconsistent")
        if tuple(item.dtype for item in self.files) != (
            "float32",
            "int64",
            "int64",
        ):
            raise ValueError("feature table array dtypes are inconsistent")
        return self


class RotatedMnistFeatureCacheManifest(StrictBoundaryModel):
    schema_version: Literal["grit.rotated-mnist-features/v1"]
    dataset_id: Literal["rotated_mnist"]
    source_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    encoder: EncoderIdentity
    extraction_runtime: FeatureExtractionRuntime
    normalization: Normalization
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    tables: tuple[RotatedMnistFeatureTableManifest, ...]

    @model_validator(mode="after")
    def _validate_tables(self) -> RotatedMnistFeatureCacheManifest:
        expected = (
            "train_r0",
            "train_r45",
            "val_r0",
            "val_r45",
            "val_r60",
            "test_r90",
            "oracle_pair_r0",
            "oracle_pair_r45",
        )
        if tuple(table.table_name for table in self.tables) != expected:
            raise ValueError("RotatedMNIST cache must contain the fixed table set")
        roles: tuple[TableRole, ...] = (
            "training",
            "training",
            "validation",
            "validation",
            "validation",
            "final_test",
            "pair_projection",
            "pair_projection",
        )
        if tuple(table.role for table in self.tables) != roles:
            raise ValueError("feature table roles are inconsistent")
        if len({table.source_ids for table in self.tables[2:5]}) != 1:
            raise ValueError("validation rotations must share source order")
        if self.tables[6].source_ids != self.tables[7].source_ids:
            raise ValueError("oracle endpoints must share exact source order")
        train_sources = set(self.tables[0].source_ids) | set(self.tables[1].source_ids)
        if not set(self.tables[6].source_ids) <= train_sources:
            raise ValueError("oracle endpoints must come from training sources")
        return self


@dataclass(frozen=True, slots=True)
class RotatedMnistFeatureTable:
    name: str
    role: TableRole
    source_ids: tuple[str, ...]
    features: torch.Tensor
    targets: torch.Tensor
    angles: torch.Tensor


@dataclass(frozen=True, slots=True)
class RotatedMnistTuningFeatureCache:
    train_r0: RotatedMnistFeatureTable
    train_r45: RotatedMnistFeatureTable
    val_r0: RotatedMnistFeatureTable
    val_r45: RotatedMnistFeatureTable
    val_r60: RotatedMnistFeatureTable
    oracle_pair_r0: RotatedMnistFeatureTable
    oracle_pair_r45: RotatedMnistFeatureTable
    manifest: RotatedMnistFeatureCacheManifest
    root: Path

    def training_tables(
        self,
    ) -> tuple[RotatedMnistFeatureTable, RotatedMnistFeatureTable]:
        return self.train_r0, self.train_r45

    def validation_tables(
        self,
    ) -> tuple[
        RotatedMnistFeatureTable,
        RotatedMnistFeatureTable,
        RotatedMnistFeatureTable,
    ]:
        return self.val_r0, self.val_r45, self.val_r60

    def pair_tables(
        self,
    ) -> tuple[RotatedMnistFeatureTable, RotatedMnistFeatureTable]:
        return self.oracle_pair_r0, self.oracle_pair_r45


@dataclass(frozen=True, slots=True)
class RotatedMnistFeatureCache(RotatedMnistTuningFeatureCache):
    _test_r90: RotatedMnistFeatureTable

    def issue_final_handle(
        self,
        *,
        run_id: str,
        candidate_id: str,
        scientific_config_digest: str,
    ) -> FinalTestHandle:
        descriptor = FinalTestSplitDescriptor(
            dataset_id="rotated_mnist",
            manifest_id=self.manifest.canonical_digest(),
            name="test_r90",
            role="final_test",
            source_partition_id="test_sources",
            view_id="test_r90",
        )
        examples = tuple(
            ExampleIdentity(
                example_id=f"{source_id}:view:test_r90",
                source_id=source_id,
                view_id="test_r90",
            )
            for source_id in self._test_r90.source_ids
        )
        return issue_final_test_handle(
            handle_id=f"final-handle:{run_id}",
            run_id=run_id,
            candidate_id=candidate_id,
            scientific_config_digest=scientific_config_digest,
            descriptor=descriptor,
            examples=examples,
        )

    def open_final_table(self, view: FinalTestView) -> RotatedMnistFeatureTable:
        if (
            view.descriptor.dataset_id != "rotated_mnist"
            or view.descriptor.name != "test_r90"
            or view.descriptor.manifest_id != self.manifest.canonical_digest()
        ):
            raise RotatedMnistFeatureCacheValidationError(
                "authorized final view does not match the RotatedMNIST cache"
            )
        if tuple(item.source_id for item in view.examples) != self._test_r90.source_ids:
            raise RotatedMnistFeatureCacheValidationError(
                "authorized final view has different final source identities"
            )
        return self._test_r90


def prepare_rotated_mnist_feature_cache(
    construction: RotatedMnistConstruction,
    pairs: RotatedMnistOraclePairSet,
    encoder: ImageEncoder,
    output_dir: Path,
    *,
    normalization: Normalization,
    overwrite: bool = False,
) -> RotatedMnistFeatureCacheManifest:
    _validate_preparation_inputs(construction, pairs)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"feature cache directory is not empty: {output_dir}; pass overwrite=True"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = construction.preparation_tables() + (pairs.left_r0, pairs.right_r45)
    table_manifests = tuple(
        _write_table(
            output_dir,
            table,
            _normalized_features(encoder.encode(table.images), normalization),
        )
        for table in inputs
    )
    manifest = RotatedMnistFeatureCacheManifest(
        schema_version="grit.rotated-mnist-features/v1",
        dataset_id="rotated_mnist",
        source_manifest_digest=construction.manifest.canonical_digest(),
        pair_manifest_digest=pairs.manifest.canonical_digest(),
        encoder=encoder.identity,
        extraction_runtime=encoder.extraction_runtime,
        normalization=normalization,
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        tables=table_manifests,
    )
    (output_dir / "manifest.json").write_text(
        manifest.canonical_json() + "\n", encoding="utf-8"
    )
    return manifest


def _validate_preparation_inputs(
    construction: RotatedMnistConstruction, pairs: RotatedMnistOraclePairSet
) -> None:
    dataset = RotatedMnistDatasetManifest.model_validate_json(
        construction.manifest.canonical_json()
    )
    pair_manifest = RotatedMnistOraclePairManifest.model_validate_json(
        pairs.manifest.canonical_json()
    )
    if pair_manifest.dataset_manifest_digest != dataset.canonical_digest():
        raise RotatedMnistFeatureCacheValidationError(
            "pair bank belongs to a different dataset manifest"
        )
    if pair_manifest.records != pairs.records:
        raise RotatedMnistFeatureCacheValidationError(
            "pair records do not match the pair manifest"
        )
    train_targets: dict[str, int] = {}
    for table in (construction.train_r0, construction.train_r45):
        train_targets.update(
            dict(zip(table.source_ids, _tensor_ints(table.targets), strict=True))
        )
    pair_sources = tuple(record.source_id for record in pairs.records)
    if (
        pairs.left_r0.source_ids != pair_sources
        or pairs.right_r45.source_ids != pair_sources
        or _tensor_ints(pairs.left_r0.targets)
        != tuple(record.digit for record in pairs.records)
        or not all(
            train_targets.get(record.source_id) == record.digit
            for record in pairs.records
        )
    ):
        raise RotatedMnistFeatureCacheValidationError(
            "oracle endpoints do not preserve exact training-source identity and target"
        )
    if not torch.equal(pairs.left_r0.targets, pairs.right_r45.targets):
        raise RotatedMnistFeatureCacheValidationError(
            "oracle endpoint targets must be identical"
        )


def load_rotated_mnist_feature_cache(
    root: Path,
    *,
    expected_source_manifest_digest: str | None = None,
    expected_pair_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
) -> RotatedMnistFeatureCache:
    manifest = _load_manifest(
        root,
        expected_source_manifest_digest,
        expected_pair_manifest_digest,
        expected_normalization,
    )
    tables = tuple(_load_table(root, item) for item in manifest.tables)
    return RotatedMnistFeatureCache(
        train_r0=tables[0],
        train_r45=tables[1],
        val_r0=tables[2],
        val_r45=tables[3],
        val_r60=tables[4],
        _test_r90=tables[5],
        oracle_pair_r0=tables[6],
        oracle_pair_r45=tables[7],
        manifest=manifest,
        root=root,
    )


def load_rotated_mnist_tuning_feature_cache(
    root: Path,
    *,
    expected_source_manifest_digest: str | None = None,
    expected_pair_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
) -> RotatedMnistTuningFeatureCache:
    manifest = _load_manifest(
        root,
        expected_source_manifest_digest,
        expected_pair_manifest_digest,
        expected_normalization,
    )
    selected = tuple(
        _load_table(root, item) for item in manifest.tables if item.role != "final_test"
    )
    return RotatedMnistTuningFeatureCache(
        train_r0=selected[0],
        train_r45=selected[1],
        val_r0=selected[2],
        val_r45=selected[3],
        val_r60=selected[4],
        oracle_pair_r0=selected[5],
        oracle_pair_r45=selected[6],
        manifest=manifest,
        root=root,
    )


def _load_manifest(
    root: Path,
    expected_source: str | None,
    expected_pair: str | None,
    expected_normalization: Normalization | None,
) -> RotatedMnistFeatureCacheManifest:
    path = root / "manifest.json"
    if not path.is_file():
        raise RotatedMnistFeatureCacheValidationError(f"manifest is missing: {path}")
    try:
        manifest = RotatedMnistFeatureCacheManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise RotatedMnistFeatureCacheValidationError(
            f"manifest is invalid: {path}"
        ) from error
    if (
        expected_source is not None
        and manifest.source_manifest_digest != expected_source
    ):
        raise RotatedMnistFeatureCacheValidationError("source manifest does not match")
    if expected_pair is not None and manifest.pair_manifest_digest != expected_pair:
        raise RotatedMnistFeatureCacheValidationError("pair manifest does not match")
    if (
        expected_normalization is not None
        and manifest.normalization != expected_normalization
    ):
        raise RotatedMnistFeatureCacheValidationError("normalization does not match")
    return manifest


def _write_table(
    root: Path,
    table: RenderedRotatedMnistTable,
    features: torch.Tensor,
) -> RotatedMnistFeatureTableManifest:
    table_dir = root / table.name
    table_dir.mkdir(parents=True, exist_ok=True)
    arrays: tuple[tuple[ArrayField, torch.Tensor], ...] = (
        ("features", features),
        ("targets", table.targets.to(torch.int64)),
        ("angles", table.angles.to(torch.int64)),
    )
    files: list[RotatedMnistArrayFileManifest] = []
    for field_name, tensor in arrays:
        path = table_dir / f"{field_name}.npy"
        array = cast(_TensorToNumpy, tensor.detach().cpu().numpy)()
        np.save(path, array, allow_pickle=False)
        files.append(
            RotatedMnistArrayFileManifest(
                field_name=field_name,
                relative_path=path.relative_to(root).as_posix(),
                sha256=_file_sha256(path),
                shape=tuple(int(value) for value in tensor.shape),
                dtype=str(array.dtype),
            )
        )
    return RotatedMnistFeatureTableManifest(
        table_name=table.name,
        role=table.role,
        source_ids=table.source_ids,
        source_ids_digest=canonical_digest_value(table.source_ids),
        row_count=len(table.source_ids),
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        files=tuple(files),
    )


def _load_table(
    root: Path, manifest: RotatedMnistFeatureTableManifest
) -> RotatedMnistFeatureTable:
    loaded: dict[str, torch.Tensor] = {}
    for item in manifest.files:
        path = root / item.relative_path
        if not path.is_file() or _file_sha256(path) != item.sha256:
            raise RotatedMnistFeatureCacheValidationError(
                f"cache file is missing or damaged: {path}"
            )
        array = np.load(path, allow_pickle=False)
        if tuple(int(value) for value in array.shape) != item.shape:
            raise RotatedMnistFeatureCacheValidationError(
                f"cache file shape mismatch: {path}"
            )
        if str(array.dtype) != item.dtype:
            raise RotatedMnistFeatureCacheValidationError(
                f"cache file dtype mismatch: {path}"
            )
        copied = cast(NDArray[np.generic], np.array(array, copy=True))
        loaded[item.field_name] = _torch_from_numpy(copied)
    features = loaded["features"].to(torch.float32)
    targets = loaded["targets"].to(torch.int64)
    angles = loaded["angles"].to(torch.int64)
    _validate_feature_matrix(features, manifest.row_count)
    if bool(((targets < 0) | (targets > 9)).any()):
        raise RotatedMnistFeatureCacheValidationError("targets must lie in 0 through 9")
    return RotatedMnistFeatureTable(
        name=manifest.table_name,
        role=manifest.role,
        source_ids=manifest.source_ids,
        features=features,
        targets=targets,
        angles=angles,
    )


def _normalized_features(
    features: torch.Tensor, normalization: Normalization
) -> torch.Tensor:
    result = features.detach().cpu().to(torch.float32)
    _validate_feature_matrix(result, int(result.shape[0]))
    if normalization == "l2":
        norms = result.square().sum(dim=1, keepdim=True).sqrt()
        if bool((norms == 0).any()):
            raise ValueError("cannot normalize a zero feature vector")
        result = result / norms
    return result.contiguous()


def _validate_feature_matrix(features: torch.Tensor, rows: int) -> None:
    if tuple(features.shape) != (rows, FEATURE_DIMENSION):
        raise RotatedMnistFeatureCacheValidationError(
            f"features must have shape ({rows}, {FEATURE_DIMENSION})"
        )
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise RotatedMnistFeatureCacheValidationError("features must be finite")


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


class _TorchFromNumpy(Protocol):
    def __call__(self, array: NDArray[np.generic]) -> torch.Tensor: ...


_torch_from_numpy = cast(_TorchFromNumpy, torch.from_numpy)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _tensor_ints(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value.item()) for value in values.detach().cpu().reshape(-1))
