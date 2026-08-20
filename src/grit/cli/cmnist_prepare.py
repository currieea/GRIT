"""Prepare official MNIST and pinned OpenAI CLIP features explicitly."""

from __future__ import annotations

import argparse
from pathlib import Path

from grit.runner import prepare_official_cmnist


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic CMNIST partitions, pairs, and CLIP features."
    )
    _ = parser.add_argument("--data-root", type=Path, required=True)
    _ = parser.add_argument("--clip-weights-root", type=Path, required=True)
    _ = parser.add_argument("--output-root", type=Path, required=True)
    _ = parser.add_argument("--construction-seed", type=int, default=0)
    _ = parser.add_argument("--pair-seed", type=int, default=0)
    _ = parser.add_argument(
        "--normalization", choices=("none", "l2"), default="none"
    )
    _ = parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly permit torchvision MNIST and official CLIP downloads.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    prepare_official_cmnist(
        data_root=args.data_root,
        clip_weights_root=args.clip_weights_root,
        output_root=args.output_root,
        construction_seed=args.construction_seed,
        pair_seed=args.pair_seed,
        normalization=args.normalization,
        allow_download=args.allow_download,
    )


if __name__ == "__main__":
    main()
