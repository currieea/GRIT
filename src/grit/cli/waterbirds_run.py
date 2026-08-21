"""Run the hermetic non-reportable Waterbirds ERM/oracle-GRIT smoke slice."""

from __future__ import annotations

import argparse
from pathlib import Path

from grit.waterbirds_runner import (
    load_waterbirds_smoke_config,
    run_waterbirds_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-reportable offline Waterbirds vertical-slice smoke."
    )
    _ = parser.add_argument("config", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_waterbirds_smoke(load_waterbirds_smoke_config(args.config))
    print(summary.canonical_json())


if __name__ == "__main__":
    main()
