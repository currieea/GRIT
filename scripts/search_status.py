"""Report completed search work without training.

    uv run scripts/search_status.py configs/cmnist/production-search.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grit.production_search import production_search_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(production_search_status(Path(args.config)).canonical_json())


if __name__ == "__main__":
    main()
