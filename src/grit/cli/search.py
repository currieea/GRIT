"""Production local-search command-line boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from grit.production_search import (
    plan_production_search,
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
        result = run_production_search(Path(arguments.config))
        print(
            f"completed {result.schema_version} for plan {result.plan_digest}"
        )
        return 0
    if arguments.operation == "status":
        status = production_search_status(Path(arguments.config))
        print(status.canonical_json())
        return 0
    raise AssertionError("argparse accepted an unknown search operation")


if __name__ == "__main__":
    raise SystemExit(main())
