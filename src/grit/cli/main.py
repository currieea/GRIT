"""The `grit` command: prepare data, run a search, check status, run a smoke test."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from grit.paths import REPO_ROOT, scratch_root

# Seeds used by the checked-in production configs. Preparation must use the same ones.
DEFAULT_CONSTRUCTION_SEED = 1729
DEFAULT_PAIR_SEED = 2718

Normalization = Literal["none", "l2"]


def default_clip_weights_root() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home) / "clip"
    return scratch_root() / "data" / "clip-weights"


def auto_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grit",
        description=(
            "Paths default to $PROJECT_SCRATCH/{data,artifacts,outputs}. "
            "Run `scratch-project` first on the servers."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare", help="build a dataset, oracle pairs, and the CLIP feature cache"
    )
    prepare_sub = prepare.add_subparsers(dest="dataset", required=True)

    cm = prepare_sub.add_parser("cmnist", help="official MNIST -> ColoredMNIST")
    cm.add_argument("--data-root", type=Path, help="raw MNIST (default: scratch/data)")
    _add_common_prepare_arguments(cm)
    cm.add_argument("--pair-seed", type=int, default=DEFAULT_PAIR_SEED)

    wb = prepare_sub.add_parser("waterbirds", help="Waterbirds-CF from local assets")
    wb.add_argument("--released-root", type=Path, help="released Waterbirds-95")
    wb.add_argument("--cub-root", type=Path, help="CUB-200-2011 images")
    wb.add_argument("--masks-root", type=Path, help="CUB segmentation masks")
    wb.add_argument("--places-root", type=Path, help="the four Places365 categories")
    _add_common_prepare_arguments(wb)

    run = sub.add_parser("run", help="plan and run (or continue) a search from YAML")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--dry-run", action="store_true", help="validate, write the plan, print status"
    )
    run.add_argument(
        "--pilot",
        action="store_true",
        help="run one ERM and one rank>0 GRIT tuning task at the first seed, then stop",
    )
    run.add_argument("--limit", type=int, help="run at most N new tasks, then stop")
    run.add_argument(
        "--stop-after", choices=("tuning",), help="stop before confirmation/final"
    )
    run.add_argument(
        "--only", choices=("erm", "grit"), help="tuning-only method filter"
    )
    run.add_argument("--candidate-id", action="append", default=[])
    run.add_argument("--seed", type=int, help="tuning-only seed filter")

    status = sub.add_parser("status", help="report completed work without training")
    status.add_argument("config", type=Path)

    smoke = sub.add_parser("smoke", help="hermetic non-reportable end-to-end check")
    smoke.add_argument("dataset", choices=("cmnist", "waterbirds"))
    smoke.add_argument("config", type=Path, nargs="?")
    return parser


def _add_common_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip-weights-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--construction-seed", type=int, default=DEFAULT_CONSTRUCTION_SEED
    )
    parser.add_argument("--normalization", choices=("none", "l2"), default="none")
    parser.add_argument(
        "--device", help="cpu, cuda, or cuda:N for CLIP (default: cuda:0 if available)"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading missing MNIST or CLIP weights",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "run":
        return _run(args)
    if args.command == "status":
        from grit.production_search import production_search_status

        print(production_search_status(Path(args.config)).canonical_json())
        return 0
    if args.command == "smoke":
        return _smoke(args)
    raise AssertionError("unknown command")


def _prepare(args: argparse.Namespace) -> int:
    normalization = cast(Normalization, args.normalization)
    device = args.device or auto_device()
    clip_root = args.clip_weights_root or default_clip_weights_root()
    output_root = args.output_root or (
        scratch_root() / "artifacts" / f"{args.dataset}-{normalization}"
    )
    print(f"clip weights: {clip_root}")
    print(f"output:       {output_root}")
    print(f"device:       {device}")
    if args.dataset == "cmnist":
        from grit.runner import prepare_official_cmnist

        data_root = args.data_root or scratch_root() / "data" / "mnist"
        print(f"mnist:        {data_root}")
        prepare_official_cmnist(
            data_root=data_root,
            clip_weights_root=clip_root,
            output_root=output_root,
            construction_seed=args.construction_seed,
            pair_seed=args.pair_seed,
            normalization=normalization,
            allow_download=not args.no_download,
            feature_device=device,
            clip_batch_size=args.batch_size,
        )
        return 0
    from grit.waterbirds_runner import prepare_server_waterbirds

    data = scratch_root() / "data"
    prepare_server_waterbirds(
        released_root=args.released_root or data / "waterbirds",
        cub_root=args.cub_root or data / "cub",
        masks_root=args.masks_root or data / "cub-masks",
        places_root=args.places_root or data / "places",
        clip_weights_root=clip_root,
        output_root=output_root,
        construction_seed=args.construction_seed,
        normalization=normalization,
        allow_clip_download=not args.no_download,
        feature_device=device,
        clip_batch_size=args.batch_size,
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    from grit.production_search import (
        ProductionExecutionLimits,
        ProductionSearchStatus,
        plan_production_search,
        production_pilot_candidates,
        production_search_status,
        run_production_search,
    )

    config = Path(args.config)
    plan = plan_production_search(config)
    print(
        f"{plan.dataset}: {len(plan.candidates)} candidates, "
        f"output {plan.resolved_config.output_root}",
        file=sys.stderr,
    )
    if args.dry_run:
        print(production_search_status(config).canonical_json())
        return 0

    candidate_ids = tuple(args.candidate_id)
    seed = args.seed
    stop_after = args.stop_after
    limit = args.limit
    if args.pilot:
        pilot = production_pilot_candidates(config)
        candidate_ids = (pilot.erm.candidate_id, pilot.grit_nonzero_rank.candidate_id)
        seed = pilot.tuning_seed
        stop_after = "tuning"
        limit = 2
    filtered = args.only is not None or bool(candidate_ids) or seed is not None
    if filtered:
        stop_after = "tuning"
    has_limits = filtered or stop_after is not None or limit is not None
    limits = (
        ProductionExecutionLimits(
            stop_after=stop_after,
            method=args.only or "all",
            candidate_ids=candidate_ids,
            tuning_seed=seed,
            max_new_runs=limit,
        )
        if has_limits
        else None
    )
    result = run_production_search(config, limits)
    if isinstance(result, ProductionSearchStatus):
        print(result.canonical_json())
    else:
        print(f"completed {result.schema_version} for plan {result.plan_digest}")
    return 0


def _smoke(args: argparse.Namespace) -> int:
    config = Path(args.config) if args.config else (
        REPO_ROOT / "configs" / args.dataset / "smoke.yaml"
    )
    if args.dataset == "cmnist":
        from grit.runner import load_cmnist_smoke_config, run_cmnist_smoke

        print(run_cmnist_smoke(load_cmnist_smoke_config(config)).canonical_json())
        return 0
    from grit.waterbirds_runner import (
        load_waterbirds_smoke_config,
        run_waterbirds_smoke,
    )

    print(run_waterbirds_smoke(load_waterbirds_smoke_config(config)).canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
