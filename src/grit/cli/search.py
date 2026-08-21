"""Production local-search command-line boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from grit.production_search import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    plan_production_search,
    production_pilot_candidates,
    production_search_status,
    run_production_search,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grit-search")
    subcommands = parser.add_subparsers(dest="operation", required=True)
    plan = subcommands.add_parser(
        "plan",
        help="validate production artifacts and write a canonical search plan",
    )
    plan.add_argument("config", type=Path)
    run = subcommands.add_parser(
        "run",
        help="run or continue the validated local production search",
    )
    run.add_argument("config", type=Path)
    run.add_argument("--stop-after", choices=("tuning",))
    run.add_argument("--method", choices=("erm", "grit", "all"), default="all")
    run.add_argument("--candidate-id", action="append", default=[])
    run.add_argument("--tuning-seed", type=int)
    run.add_argument("--max-new-runs", type=int)
    pilot = subcommands.add_parser(
        "pilot-candidates",
        help="show canonical ERM and nonzero-rank GRIT tuning pilot candidates",
    )
    pilot.add_argument("config", type=Path)
    status = subcommands.add_parser(
        "status",
        help="inspect completed canonical runs without starting training",
    )
    status.add_argument("config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicitly selected production-search operation."""

    arguments = _parser().parse_args(argv)
    if arguments.operation == "plan":
        config_path = Path(arguments.config)
        plan = plan_production_search(config_path)
        print(
            f"wrote {plan.dataset} plan with {len(plan.candidates)} candidates "
            f"to {plan.resolved_config.output_root}"
        )
        return 0
    if arguments.operation == "run":
        has_limits = (
            arguments.stop_after is not None
            or arguments.method != "all"
            or bool(arguments.candidate_id)
            or arguments.tuning_seed is not None
            or arguments.max_new_runs is not None
        )
        limits = (
            ProductionExecutionLimits(
                stop_after=arguments.stop_after,
                method=arguments.method,
                candidate_ids=tuple(arguments.candidate_id),
                tuning_seed=arguments.tuning_seed,
                max_new_runs=arguments.max_new_runs,
            )
            if has_limits
            else None
        )
        result = run_production_search(Path(arguments.config), limits)
        if isinstance(result, ProductionSearchStatus):
            print(result.canonical_json())
            return 0
        print(
            f"completed {result.schema_version} for plan {result.plan_digest}"
        )
        return 0
    if arguments.operation == "pilot-candidates":
        selection = production_pilot_candidates(Path(arguments.config))
        print(selection.canonical_json())
        return 0
    if arguments.operation == "status":
        status = production_search_status(Path(arguments.config))
        print(status.canonical_json())
        return 0
    raise AssertionError("argparse accepted an unknown search operation")


if __name__ == "__main__":
    raise SystemExit(main())
