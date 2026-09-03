"""Plan and run (or continue) an ERM/GRIT search from a YAML config.

    uv run scripts/run_search.py configs/cmnist/production-search.yaml --pilot
    uv run scripts/run_search.py configs/cmnist/production-search.yaml

Completed tasks in the output directory are reused, so interrupting and rerunning
is safe. --pilot runs one ERM and one rank>0 GRIT tuning task and stops.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from grit.methods.types import IMPLEMENTED_METHODS
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    plan_production_search,
    production_pilot_candidates,
    production_search_status,
    run_production_search,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="write the plan and print status only"
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int, help="run at most N new tasks, then stop")
    parser.add_argument("--stop-after", choices=("tuning",))
    parser.add_argument(
        "--only", choices=IMPLEMENTED_METHODS, help="tuning-only filter"
    )
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--seed", type=int, help="tuning-only seed filter")
    args = parser.parse_args(argv)

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
    limits = (
        ProductionExecutionLimits(
            stop_after=stop_after,
            method=args.only or "all",
            candidate_ids=candidate_ids,
            tuning_seed=seed,
            max_new_runs=limit,
        )
        if filtered or stop_after is not None or limit is not None
        else None
    )
    result = run_production_search(config, limits)
    if isinstance(result, ProductionSearchStatus):
        print(result.canonical_json())
    else:
        print(f"completed {result.schema_version} for plan {result.plan_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
