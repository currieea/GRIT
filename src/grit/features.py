"""Narrow frozen-feature cache and official OpenAI CLIP adapter for CMNIST."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field, StrictInt, StrictStr, model_validator

from grit.cmnist import (
    CmnistConstruction,
    CmnistOraclePairSet,
    MnistPool,
    RenderedCmnistTable,
)
from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
)
from grit.data import (
    ExampleIdentity,
    FinalTestHandle,
    FinalTestSplitDescriptor,
    FinalTestView,
    issue_final_test_handle,
)
from grit.schemas import StrictBoundaryModel, canonical_digest_value

OPENAI_CLIP_MODEL = "ViT-B/32"
OPENAI_CLIP_WEIGHTS_SHA256 = OPENAI_CLIP_WEIGHTS_IDENTITY.removeprefix("sha256:")
FEATURE_DIMENSION = 512

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
Normalization: TypeAlias = Literal["none", "l2"]
TableRole: TypeAlias = Literal[
    "training", "validation", "final_test", "pair_projection"
]
ArrayField: TypeAlias = Literal[
    "features", "digits", "clean_labels", "targets", "colors"
]


class FeatureCacheValidationError(ValueError):
    """A CMNIST feature cache is missing, damaged, or scientifically mismatched."""


class EncoderIdentity(StrictBoundaryModel):
    implementation: NonEmptyStr
    implementation_revision: NonEmptyStr
    model_name: NonEmptyStr
    weights_identity: NonEmptyStr
    preprocessing_identity: NonEmptyStr
    raw_output_dimension: NonNegativeInt


class ImageEncoder(Protocol):
    @property
    def identity(self) -> EncoderIdentity: ...

    def encode(self, images: torch.Tensor) -> torch.Tensor: ...


class _ClipModel(Protocol):
    def eval(self) -> object: ...

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...


class _ClipPreprocess(Protocol):
    def __call__(self, image: Image.Image) -> torch.Tensor: ...


class _ClipModule(Protocol):
    def load(
        self,
        name: str,
        device: str,
        jit: bool,
        download_root: str,
    ) -> tuple[_ClipModel, _ClipPreprocess]: ...


class _ToPilImage(Protocol):
    def __call__(self, image: torch.Tensor, mode: str | None = None) -> Image.Image: ...


class _MnistDataset(Protocol):
    data: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int: ...


class _MnistFactory(Protocol):
    def __call__(
        self,
        root: str,
        train: bool,
        download: bool,
    ) -> _MnistDataset: ...


class _TensorToNumpy(Protocol):
    def __call__(self, *, force: bool = False) -> NDArray[np.generic]: ...


class _TorchFromNumpy(Protocol):
    def __call__(self, array: NDArray[np.generic]) -> torch.Tensor: ...


_to_pil_image = cast(
    _ToPilImage,
    importlib.import_module("torchvision.transforms.functional").to_pil_image,
)
_mnist_factory = cast(
    _MnistFactory,
    importlib.import_module("torchvision.datasets").MNIST,
)
_torch_from_numpy = cast(_TorchFromNumpy, torch.from_numpy)


@dataclass(slots=True)
class OfficialOpenAiClipEncoder:
    """Pinned official OpenAI CLIP ViT-B/32 with unnormalized raw outputs."""

    weights_root: Path
    allow_download: bool
    batch_size: int = 256
    _model: _ClipModel | None = None
    _preprocess: _ClipPreprocess | None = None

    @property
    def identity(self) -> EncoderIdentity:
        return EncoderIdentity(
            implementation="openai/CLIP",
            implementation_revision=OPENAI_CLIP_REVISION,
            model_name=OPENAI_CLIP_MODEL,
            weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
            preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
            raw_output_dimension=FEATURE_DIMENSION,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError("CLIP input must have shape [N, 3, H, W]")
        if self.batch_size <= 0:
            raise ValueError("CLIP batch_size must be positive")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("CLIP input images must be finite floating-point tensors")
        if float(images.min()) < 0.0 or float(images.max()) > 1.0:
            raise ValueError("CLIP input images must lie in [0, 1]")
        model, preprocess = self._resolved_model()
        batches: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, int(images.shape[0]), self.batch_size):
                stop = min(start + self.batch_size, int(images.shape[0]))
                inputs = torch.stack(
                    [
                        preprocess(_to_pil_image(image))
                        for image in images[start:stop]
                    ]
                )
                batches.append(
                    model.encode_image(inputs).detach().cpu().to(torch.float32)
                )
        if not batches:
            raise ValueError("CLIP input must contain at least one image")
        features = torch.cat(batches, dim=0)
        _validate_feature_matrix(features, expected_rows=int(images.shape[0]))
        return features

    def _resolved_model(self) -> tuple[_ClipModel, _ClipPreprocess]:
        if self._model is not None and self._preprocess is not None:
            return self._model, self._preprocess
        self.weights_root.mkdir(parents=True, exist_ok=True)
        weights_path = self.weights_root / "ViT-B-32.pt"
        if not self.allow_download and not weights_path.is_file():
            raise FileNotFoundError(
                "official OpenAI CLIP weights are absent; rerun preparation with "
                "--allow-download"
            )
        module = cast(_ClipModule, importlib.import_module("clip"))
        model, preprocess = module.load(
            OPENAI_CLIP_MODEL,
            device="cpu",
            jit=False,
            download_root=str(self.weights_root),
        )
        if _file_sha256(weights_path) != f"sha256:{OPENAI_CLIP_WEIGHTS_SHA256}":
            raise FeatureCacheValidationError(
                "official OpenAI CLIP ViT-B/32 weight digest does not match"
            )
        _ = model.eval()
        self._model = model
        self._preprocess = preprocess
        return model, preprocess


@dataclass(frozen=True, slots=True)
class DeterministicFakeEncoder:
    """Explicitly non-reportable 512-D encoder used only by smoke/tests."""

    seed: int = 0

    @property
    def identity(self) -> EncoderIdentity:
        return EncoderIdentity(
            implementation="grit.synthetic",
            implementation_revision="deterministic-fake-encoder-v1",
            model_name="fake-512",
            weights_identity=f"seed:{self.seed}",
            preprocessing_identity="identity-rgb",
            raw_output_dimension=FEATURE_DIMENSION,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or int(images.shape[1]) != 3:
            raise ValueError("fake encoder input must have shape [N, 3, H, W]")
        flattened = images.detach().cpu().to(torch.float32).flatten(start_dim=1)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        weights = torch.randn(
            (int(flattened.shape[1]), FEATURE_DIMENSION),
            generator=generator,
            dtype=torch.float32,
        )
        features = flattened @ weights / max(int(flattened.shape[1]), 1) ** 0.5
        features[:, 0] += 1.0
        _validate_feature_matrix(features, expected_rows=int(images.shape[0]))
        return features


class ArrayFileManifest(StrictBoundaryModel):
    field_name: Literal["features", "digits", "clean_labels", "targets", "colors"]
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    shape: tuple[NonNegativeInt, ...]
    dtype: NonEmptyStr


class FeatureTableManifest(StrictBoundaryModel):
    table_name: NonEmptyStr
    role: TableRole
    source_ids: tuple[NonEmptyStr, ...]
    source_ids_digest: NonEmptyStr
    row_count: NonNegativeInt
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    files: tuple[ArrayFileManifest, ...]

    @model_validator(mode="after")
    def _validate_table(self) -> FeatureTableManifest:
        if len(self.source_ids) != self.row_count:
            raise ValueError("feature table source count is inconsistent")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("feature table source IDs must be unique")
        if self.source_ids_digest != canonical_digest_value(self.source_ids):
            raise ValueError("feature table source digest is inconsistent")
        expected_fields = ("features", "digits", "clean_labels", "targets", "colors")
        if tuple(file.field_name for file in self.files) != expected_fields:
            raise ValueError("feature table files must use canonical field ordering")
        expected_shapes = ((self.row_count, 512),) + ((self.row_count,),) * 4
        expected_dtypes = ("float32", "int64", "int64", "int64", "int64")
        if tuple(file.shape for file in self.files) != expected_shapes:
            raise ValueError("feature table array shapes are inconsistent")
        if tuple(file.dtype for file in self.files) != expected_dtypes:
            raise ValueError("feature table array dtypes are inconsistent")
        relative_paths = tuple(file.relative_path for file in self.files)
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("feature table array paths must be unique")
        return self


class CmnistFeatureCacheManifest(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-features/v1"]
    dataset_id: Literal["cmnist"]
    source_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    encoder: EncoderIdentity
    normalization: Normalization
    feature_dimension: Literal[512]
    feature_dtype: Literal["float32"]
    tables: tuple[FeatureTableManifest, ...]

    @model_validator(mode="after")
    def _validate_tables(self) -> CmnistFeatureCacheManifest:
        expected = (
            "train_e01",
            "train_e02",
            "val_e01",
            "val_e02",
            "val_e05",
            "test_ood",
            "oracle_pair_red",
            "oracle_pair_green",
        )
        if tuple(table.table_name for table in self.tables) != expected:
            raise ValueError("CMNIST feature cache must contain the fixed table set")
        expected_roles: tuple[TableRole, ...] = (
            "training",
            "training",
            "validation",
            "validation",
            "validation",
            "final_test",
            "pair_projection",
            "pair_projection",
        )
        if tuple(table.role for table in self.tables) != expected_roles:
            raise ValueError("CMNIST feature table roles are inconsistent")
        validation_sources = {table.source_ids for table in self.tables[2:5]}
        if len(validation_sources) != 1:
            raise ValueError("CMNIST cached validation tables must share source order")
        if self.tables[6].source_ids != self.tables[7].source_ids:
            raise ValueError(
                "CMNIST cached oracle pair endpoints must share source order"
            )
        training_sources = set(self.tables[0].source_ids) | set(
            self.tables[1].source_ids
        )
        if not set(self.tables[6].source_ids) <= training_sources:
            raise ValueError("CMNIST cached oracle pairs must use training sources")
        return self


@dataclass(frozen=True, slots=True)
class FeatureTable:
    name: str
    role: TableRole
    source_ids: tuple[str, ...]
    features: torch.Tensor
    digits: torch.Tensor
    clean_labels: torch.Tensor
    targets: torch.Tensor
    colors: torch.Tensor


@dataclass(frozen=True, slots=True)
class CmnistFeatureCache:
    train_e01: FeatureTable
    train_e02: FeatureTable
    val_e01: FeatureTable
    val_e02: FeatureTable
    val_e05: FeatureTable
    _test_ood: FeatureTable
    oracle_pair_red: FeatureTable
    oracle_pair_green: FeatureTable
    manifest: CmnistFeatureCacheManifest
    root: Path

    def training_tables(self) -> tuple[FeatureTable, FeatureTable]:
        return self.train_e01, self.train_e02

    def validation_tables(self) -> tuple[FeatureTable, FeatureTable, FeatureTable]:
        return self.val_e01, self.val_e02, self.val_e05

    def pair_tables(self) -> tuple[FeatureTable, FeatureTable]:
        return self.oracle_pair_red, self.oracle_pair_green

    def issue_final_handle(
        self,
        *,
        run_id: str,
        candidate_id: str,
        scientific_config_digest: str,
    ) -> FinalTestHandle:
        """Issue identities only; cached test features remain unopened."""

        descriptor = FinalTestSplitDescriptor(
            dataset_id="cmnist",
            manifest_id=self.manifest.source_manifest_digest,
            name="test_ood",
            role="final_test",
            source_partition_id="test_sources",
            view_id="test_ood",
        )
        examples = tuple(
            ExampleIdentity(
                example_id=f"{source_id}:view:test_ood",
                source_id=source_id,
                view_id="test_ood",
            )
            for source_id in self._test_ood.source_ids
        )
        return issue_final_test_handle(
            handle_id=f"final-handle:{run_id}",
            run_id=run_id,
            candidate_id=candidate_id,
            scientific_config_digest=scientific_config_digest,
            descriptor=descriptor,
            examples=examples,
        )

    def open_final_table(self, view: FinalTestView) -> FeatureTable:
        if view.descriptor.name != "test_ood":
            raise ValueError(
                "final feature access requires an authorized test_ood view"
            )
        expected = tuple(example.source_id for example in view.examples)
        if expected != self._test_ood.source_ids:
            raise FeatureCacheValidationError(
                "authorized final view does not match cached test source identities"
            )
        return self._test_ood


def prepare_cmnist_feature_cache(
    construction: CmnistConstruction,
    pairs: CmnistOraclePairSet,
    encoder: ImageEncoder,
    output_dir: Path,
    *,
    normalization: Normalization,
    overwrite: bool = False,
) -> CmnistFeatureCacheManifest:
    """Encode and write the fixed CMNIST environment/pair feature tables."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"feature cache directory is not empty: {output_dir}; pass overwrite=True"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_inputs = tuple(
        _from_rendered_table(table) for table in construction.preparation_tables()
    ) + (
        _from_pair_endpoint(pairs, "oracle_pair_red", "red"),
        _from_pair_endpoint(pairs, "oracle_pair_green", "green"),
    )
    manifests: list[FeatureTableManifest] = []
    for table in table_inputs:
        raw_features = encoder.encode(table.images)
        features = _normalized_features(raw_features, normalization)
        manifests.append(_write_feature_table(output_dir, table, features))
    manifest = CmnistFeatureCacheManifest(
        schema_version="grit.cmnist-features/v1",
        dataset_id="cmnist",
        source_manifest_digest=construction.manifest.canonical_digest(),
        pair_manifest_digest=pairs.manifest.canonical_digest(),
        encoder=encoder.identity,
        normalization=normalization,
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        tables=tuple(manifests),
    )
    (output_dir / "manifest.json").write_text(
        manifest.canonical_json() + "\n", encoding="utf-8"
    )
    return manifest


def load_cmnist_feature_cache(
    root: Path,
    *,
    expected_source_manifest_digest: str | None = None,
    expected_pair_manifest_digest: str | None = None,
    expected_normalization: Normalization | None = None,
) -> CmnistFeatureCache:
    """Load a verified CMNIST cache and fail helpfully on absent/mixed artifacts."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FeatureCacheValidationError(
            f"CMNIST feature cache manifest is missing: {manifest_path}"
        )
    try:
        manifest = CmnistFeatureCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise FeatureCacheValidationError(
            f"CMNIST feature cache manifest is invalid: {manifest_path}"
        ) from error
    if (
        expected_source_manifest_digest is not None
        and manifest.source_manifest_digest != expected_source_manifest_digest
    ):
        raise FeatureCacheValidationError(
            "feature cache source manifest does not match"
        )
    if (
        expected_pair_manifest_digest is not None
        and manifest.pair_manifest_digest != expected_pair_manifest_digest
    ):
        raise FeatureCacheValidationError("feature cache pair manifest does not match")
    if (
        expected_normalization is not None
        and manifest.normalization != expected_normalization
    ):
        raise FeatureCacheValidationError("feature cache normalization does not match")
    loaded = tuple(_load_feature_table(root, table) for table in manifest.tables)
    return CmnistFeatureCache(
        train_e01=loaded[0],
        train_e02=loaded[1],
        val_e01=loaded[2],
        val_e02=loaded[3],
        val_e05=loaded[4],
        _test_ood=loaded[5],
        oracle_pair_red=loaded[6],
        oracle_pair_green=loaded[7],
        manifest=manifest,
        root=root,
    )


def load_torchvision_mnist_pools(
    data_root: Path,
    *,
    allow_download: bool,
) -> tuple[MnistPool, MnistPool]:
    """Explicit disk/download adapter; core construction accepts injected pools."""

    try:
        train = _mnist_factory(
            root=str(data_root), train=True, download=allow_download
        )
        test = _mnist_factory(
            root=str(data_root), train=False, download=allow_download
        )
    except RuntimeError as error:
        raise FileNotFoundError(
            "MNIST artifacts are absent; rerun preparation with --allow-download"
        ) from error
    return (
        MnistPool(
            official_split="train",
            source_indices=torch.arange(len(train), dtype=torch.int64),
            images=train.data,
            digits=train.targets,
        ),
        MnistPool(
            official_split="test",
            source_indices=torch.arange(len(test), dtype=torch.int64),
            images=test.data,
            digits=test.targets,
        ),
    )


@dataclass(frozen=True, slots=True)
class _TableInput:
    name: str
    role: TableRole
    source_ids: tuple[str, ...]
    images: torch.Tensor
    digits: torch.Tensor
    clean_labels: torch.Tensor
    targets: torch.Tensor
    colors: torch.Tensor


def _from_rendered_table(table: RenderedCmnistTable) -> _TableInput:
    return _TableInput(
        name=table.name,
        role=table.role,
        source_ids=table.source_ids,
        images=table.images,
        digits=table.digits,
        clean_labels=table.clean_labels,
        targets=table.targets,
        colors=table.colors,
    )


def _from_pair_endpoint(
    pairs: CmnistOraclePairSet,
    name: Literal["oracle_pair_red", "oracle_pair_green"],
    color: Literal["red", "green"],
) -> _TableInput:
    is_red = color == "red"
    source_ids = tuple(record.source_id for record in pairs.records)
    return _TableInput(
        name=name,
        role="pair_projection",
        source_ids=source_ids,
        images=pairs.left_red if is_red else pairs.right_green,
        digits=torch.tensor([record.digit for record in pairs.records]),
        clean_labels=torch.tensor([record.clean_label for record in pairs.records]),
        targets=torch.tensor([record.noisy_target for record in pairs.records]),
        colors=torch.full((len(pairs.records),), 0 if is_red else 1),
    )


def _normalized_features(
    features: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    _validate_feature_matrix(features, expected_rows=int(features.shape[0]))
    result = features.detach().cpu().to(torch.float32)
    if normalization == "l2":
        norms = torch.sqrt(result.square().sum(dim=1, keepdim=True))
        if bool((norms == 0).any()):
            raise ValueError("cannot L2-normalize a zero feature vector")
        result = result / norms
    return result.contiguous()


def _validate_feature_matrix(features: torch.Tensor, *, expected_rows: int) -> None:
    if features.ndim != 2 or tuple(features.shape) != (
        expected_rows,
        FEATURE_DIMENSION,
    ):
        raise ValueError(
            f"encoder must return shape ({expected_rows}, {FEATURE_DIMENSION})"
        )
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("encoder features must be finite floating-point values")


def _write_feature_table(
    root: Path,
    table: _TableInput,
    features: torch.Tensor,
) -> FeatureTableManifest:
    table_dir = root / table.name
    table_dir.mkdir(parents=True, exist_ok=True)
    arrays: tuple[tuple[ArrayField, torch.Tensor], ...] = (
        ("features", features),
        ("digits", table.digits.to(torch.int64)),
        ("clean_labels", table.clean_labels.to(torch.int64)),
        ("targets", table.targets.to(torch.int64)),
        ("colors", table.colors.to(torch.int64)),
    )
    files: list[ArrayFileManifest] = []
    for field_name, tensor in arrays:
        path = table_dir / f"{field_name}.npy"
        to_numpy = cast(_TensorToNumpy, tensor.detach().cpu().numpy)
        array = to_numpy()
        np.save(path, array, allow_pickle=False)
        files.append(
            ArrayFileManifest(
                field_name=field_name,
                relative_path=path.relative_to(root).as_posix(),
                sha256=_file_sha256(path),
                shape=tuple(int(value) for value in tensor.shape),
                dtype=str(array.dtype),
            )
        )
    return FeatureTableManifest(
        table_name=table.name,
        role=table.role,
        source_ids=table.source_ids,
        source_ids_digest=canonical_digest_value(table.source_ids),
        row_count=len(table.source_ids),
        feature_dimension=FEATURE_DIMENSION,
        feature_dtype="float32",
        files=tuple(files),
    )


def _load_feature_table(root: Path, manifest: FeatureTableManifest) -> FeatureTable:
    loaded: dict[str, torch.Tensor] = {}
    for file in manifest.files:
        path = root / file.relative_path
        if not path.is_file():
            raise FeatureCacheValidationError(f"feature cache file is missing: {path}")
        if _file_sha256(path) != file.sha256:
            raise FeatureCacheValidationError(
                f"feature cache file digest mismatch: {path}"
            )
        array = np.load(path, allow_pickle=False)
        if tuple(int(value) for value in array.shape) != file.shape:
            raise FeatureCacheValidationError(
                f"feature cache file shape mismatch: {path}"
            )
        if str(array.dtype) != file.dtype:
            raise FeatureCacheValidationError(
                f"feature cache file dtype mismatch: {path}"
            )
        copied = cast(NDArray[np.generic], np.array(array, copy=True))
        loaded[file.field_name] = _torch_from_numpy(copied)
    features = loaded["features"].to(torch.float32)
    _validate_feature_matrix(features, expected_rows=manifest.row_count)
    digits = loaded["digits"].to(torch.int64)
    clean_labels = loaded["clean_labels"].to(torch.int64)
    targets = loaded["targets"].to(torch.int64)
    colors = loaded["colors"].to(torch.int64)
    if bool(((digits < 0) | (digits > 9)).any()):
        raise FeatureCacheValidationError("feature cache digit labels are invalid")
    for name, values in (
        ("clean_labels", clean_labels),
        ("targets", targets),
        ("colors", colors),
    ):
        if bool(((values < 0) | (values > 1)).any()):
            raise FeatureCacheValidationError(
                f"feature cache {name} must contain binary values"
            )
    return FeatureTable(
        name=manifest.table_name,
        role=manifest.role,
        source_ids=tuple(manifest.source_ids),
        features=features,
        digits=digits,
        clean_labels=clean_labels,
        targets=targets,
        colors=colors,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
