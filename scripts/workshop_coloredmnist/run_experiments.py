#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

SEEDS = [1001]

SHARED_ARGS = {
    "dataset": "ColoredMNIST",
    "pretrained": "true",
    "root_dir": "./data",
    "batch_size": "256",
    "lr": "0.001",
    "weight_decay": "1e-4",
    "epochs": "40",
    "featurizer": "linear",
    "no_wandb": None,
}


def build_cmd(python_cmd: str, params: Dict[str, str]) -> List[str]:
    cmd = shlex.split(python_cmd) + ["main.py"]
    for k, v in params.items():
        flag = f"--{k}"
        if v is None:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def run_one(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        return process.wait()


def make_run_name(spec: Dict[str, str]) -> str:
    method = spec["solver"]
    projection = spec.get("projection", "na")
    p1 = spec.get("param1", "na")
    p2 = spec.get("param2", "na")
    p3 = spec.get("param3", "na")
    seed = spec["seed"]
    return f"{method}__proj-{projection}__p1-{p1}__p2-{p2}__p3-{p3}__seed-{seed}"


def build_grid(methods: List[str]) -> List[Dict[str, str]]:
    runs: List[Dict[str, str]] = []

    if "ERM" in methods:
        for seed in SEEDS:
            spec = {
                **SHARED_ARGS,
                "solver": "ERM",
                "projection": "oracle",
                "seed": str(seed),
            }
            runs.append(spec)

    if "ECMP" in methods:
        for seed, projection, param1 in itertools.product(
            SEEDS,
            ["conditional", "oracle"],
            [6, 14, 22],
        ):
            spec = {
                **SHARED_ARGS,
                "solver": "ECMP",
                "projection": projection,
                "param1": str(param1),
                "seed": str(seed),
            }
            runs.append(spec)

    if "KernelGRIT" in methods:
        for seed, projection, param2, param3 in itertools.product(
            SEEDS,
            ["oracle"],
            [10000],
            [0.01, 0.03],
        ):
            spec = {
                **SHARED_ARGS,
                "solver": "KernelGRIT",
                "projection": projection,
                "param1": "2000",
                "param2": str(param2),
                "param3": str(param3),
                "seed": str(seed),
            }
            runs.append(spec)

    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run workshop ColoredMNIST experiment matrix"
    )
    parser.add_argument(
        "--python-cmd",
        default="uv run python",
        help="Python invocation prefix, e.g. 'uv run python' or '.venv/bin/python'",
    )
    parser.add_argument(
        "--results-dir",
        default="results/workshop_coloredmnist",
        help="Directory for logs and manifest",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ERM", "ECMP", "KernelGRIT"],
        choices=["ERM", "ECMP", "KernelGRIT"],
        help="Subset of methods to run",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute runs. If omitted, only print commands and write manifest.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs = build_grid(args.methods)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    manifest_path = results_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_name",
            "log_path",
            "solver",
            "projection",
            "param1",
            "param2",
            "param3",
            "seed",
            "cmd",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        failures = 0
        for spec in runs:
            run_name = make_run_name(spec)
            method_dir = logs_dir / spec["solver"]
            log_path = method_dir / f"{run_name}.log"
            cmd = build_cmd(args.python_cmd, spec)
            cmd_str = shlex.join(cmd)

            writer.writerow(
                {
                    "run_name": run_name,
                    "log_path": str(log_path),
                    "solver": spec.get("solver", ""),
                    "projection": spec.get("projection", ""),
                    "param1": spec.get("param1", ""),
                    "param2": spec.get("param2", ""),
                    "param3": spec.get("param3", ""),
                    "seed": spec.get("seed", ""),
                    "cmd": cmd_str,
                }
            )

            print(f"[{spec['solver']}] {run_name}")
            print(cmd_str)

            if args.execute:
                rc = run_one(cmd, log_path)
                if rc != 0:
                    failures += 1
                    print(f"Run failed with code {rc}: {run_name}")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Planned runs: {len(runs)}")
    if args.execute:
        print(f"Failed runs: {failures}")
        return 1 if failures > 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
