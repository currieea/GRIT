"""Build RotatedMNIST, 256 exact oracle pairs, and its CLIP feature cache.

    uv run scripts/prepare_rotated_mnist.py

Paths default under $PROJECT_SCRATCH (run `scratch-project` first on the servers).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from grit.paths import (
    DEFAULT_CONSTRUCTION_SEED,
    DEFAULT_PAIR_SEED,
    auto_device,
    default_clip_weights_root,
    scratch_root,
)
from grit.search.rotated_mnist_runner import prepare_official_rotated_mnist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="raw MNIST")
    parser.add_argument("--clip-weights-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--construction-seed", type=int, default=DEFAULT_CONSTRUCTION_SEED
    )
    parser.add_argument("--pair-seed", type=int, default=DEFAULT_PAIR_SEED)
    parser.add_argument("--pair-count", type=int, default=256)
    parser.add_argument("--normalization", choices=("none", "l2"), default="none")
    parser.add_argument("--device", help="cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    normalization = cast(Literal["none", "l2"], args.normalization)
    data_root = args.data_root or scratch_root() / "data" / "mnist"
    clip_root = args.clip_weights_root or default_clip_weights_root()
    output_root = args.output_root or scratch_root() / "artifacts" / (
        f"rotated-mnist-{normalization}"
    )
    device = args.device or auto_device()
    print(f"mnist:   {data_root}\nclip:    {clip_root}")
    print(f"output:  {output_root}\ndevice:  {device}")
    prepare_official_rotated_mnist(
        data_root=data_root,
        clip_weights_root=clip_root,
        output_root=output_root,
        construction_seed=args.construction_seed,
        pair_seed=args.pair_seed,
        normalization=normalization,
        allow_download=not args.no_download,
        feature_device=device,
        clip_batch_size=args.batch_size,
        pair_count=args.pair_count,
    )


if __name__ == "__main__":
    main()
