"""Real-data preparation for the RotatedMNIST vertical slice."""

from __future__ import annotations

from pathlib import Path

from grit.data.rotated_mnist import (
    PRODUCTION_PARTITION_TARGETS,
    build_rotated_mnist_oracle_pairs,
    construct_rotated_mnist,
)
from grit.features.cmnist import OfficialOpenAiClipEncoder, load_torchvision_mnist_pools
from grit.features.rotated_mnist import (
    Normalization,
    prepare_rotated_mnist_feature_cache,
)

DATASET_MANIFEST_RELATIVE_PATH = Path("dataset-manifest.json")
PAIR_MANIFEST_RELATIVE_PATH = Path("pair-manifest.json")
FEATURE_CACHE_RELATIVE_ROOT = Path("feature-cache")


def prepare_official_rotated_mnist(
    *,
    data_root: Path,
    clip_weights_root: Path,
    output_root: Path,
    construction_seed: int,
    pair_seed: int,
    normalization: Normalization,
    allow_download: bool,
    feature_device: str,
    clip_batch_size: int,
    pair_count: int = 256,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"preparation output directory is not empty: {output_root}"
        )
    encoder = OfficialOpenAiClipEncoder(
        weights_root=clip_weights_root,
        allow_download=allow_download,
        device=feature_device,
        batch_size=clip_batch_size,
    )
    encoder.preflight()
    train_pool, test_pool = load_torchvision_mnist_pools(
        data_root, allow_download=allow_download
    )
    construction = construct_rotated_mnist(
        train_pool,
        test_pool,
        construction_seed=construction_seed,
        targets=PRODUCTION_PARTITION_TARGETS,
    )
    pairs = build_rotated_mnist_oracle_pairs(
        construction,
        train_pool,
        pair_seed=pair_seed,
        pair_count=pair_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / DATASET_MANIFEST_RELATIVE_PATH).write_text(
        construction.manifest.canonical_json() + "\n", encoding="utf-8"
    )
    (output_root / PAIR_MANIFEST_RELATIVE_PATH).write_text(
        pairs.manifest.canonical_json() + "\n", encoding="utf-8"
    )
    _ = prepare_rotated_mnist_feature_cache(
        construction,
        pairs,
        encoder,
        output_root / FEATURE_CACHE_RELATIVE_ROOT,
        normalization=normalization,
    )
