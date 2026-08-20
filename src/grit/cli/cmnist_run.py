"""Run the validated, hermetic CMNIST ERM/oracle-GRIT smoke slice."""

from __future__ import annotations

import argparse
from pathlib import Path

from grit.runner import load_cmnist_smoke_config, run_cmnist_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-reportable hermetic CMNIST vertical-slice smoke test."
    )
    _ = parser.add_argument("config", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_cmnist_smoke_config(args.config)
    result = run_cmnist_smoke(config)
    print(result.canonical_json())


if __name__ == "__main__":
    main()
