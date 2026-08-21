"""Prepare server-provided Waterbirds-CF assets and pinned CLIP features."""

from __future__ import annotations

import argparse
from pathlib import Path

from grit.waterbirds_runner import prepare_server_waterbirds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct Waterbirds-CF from explicit local Waterbirds/CUB/Places assets."
        )
    )
    _ = parser.add_argument("--released-root", type=Path, required=True)
    _ = parser.add_argument("--cub-root", type=Path, required=True)
    _ = parser.add_argument("--masks-root", type=Path, required=True)
    _ = parser.add_argument("--places-root", type=Path, required=True)
    _ = parser.add_argument("--clip-weights-root", type=Path, required=True)
    _ = parser.add_argument("--output-root", type=Path, required=True)
    _ = parser.add_argument("--construction-seed", type=int, default=0)
    _ = parser.add_argument("--normalization", choices=("none", "l2"), default="none")
    _ = parser.add_argument(
        "--allow-clip-download",
        action="store_true",
        help="Explicitly permit only the pinned official CLIP weight download.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    prepare_server_waterbirds(
        released_root=args.released_root,
        cub_root=args.cub_root,
        masks_root=args.masks_root,
        places_root=args.places_root,
        clip_weights_root=args.clip_weights_root,
        output_root=args.output_root,
        construction_seed=args.construction_seed,
        normalization=args.normalization,
        allow_clip_download=args.allow_clip_download,
    )


if __name__ == "__main__":
    main()
