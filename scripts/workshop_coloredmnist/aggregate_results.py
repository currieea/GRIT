#!/usr/bin/env python3
import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DICT_LINE_RE = re.compile(r"^\{.*\}$")
AVG_LINE_RE = re.compile(r"^\[(?P<split>[^\]]+)\]\s+Average acc:\s+(?P<val>[0-9.]+)")
CORR_RE = re.compile(
    r"corr_fro_rel=(?P<corr_fro>[0-9.eE+-]+),\s*corr_abs_mean=(?P<corr_abs>[0-9.eE+-]+)"
)


def parse_run_name(log_path: Path) -> Dict[str, str]:
    stem = log_path.stem
    parts = stem.split("__")
    out: Dict[str, str] = {
        "run_name": stem,
        "solver": "",
        "projection": "",
        "param1": "",
        "param2": "",
        "param3": "",
        "seed": "",
    }
    if parts:
        out["solver"] = parts[0]
    for p in parts[1:]:
        if p.startswith("proj-"):
            out["projection"] = p[len("proj-") :]
        elif p.startswith("p1-"):
            out["param1"] = p[len("p1-") :]
        elif p.startswith("p2-"):
            out["param2"] = p[len("p2-") :]
        elif p.startswith("p3-"):
            out["param3"] = p[len("p3-") :]
        elif p.startswith("seed-"):
            out["seed"] = p[len("seed-") :]
    return out


def try_parse_dict_line(line: str) -> Optional[Dict]:
    s = line.strip()
    if not DICT_LINE_RE.match(s):
        return None
    try:
        obj = ast.literal_eval(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def parse_log(log_path: Path) -> Dict[str, Optional[float]]:
    in_val = None
    test = None
    corr_fro = None
    corr_abs = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            obj = try_parse_dict_line(line)
            if obj is not None:
                if "in_val" in obj and isinstance(obj["in_val"], dict):
                    in_val = float(obj["in_val"].get("acc_avg", in_val if in_val is not None else math.nan))
                if "test" in obj and isinstance(obj["test"], dict):
                    test = float(obj["test"].get("acc_avg", test if test is not None else math.nan))

            m_avg = AVG_LINE_RE.match(line.strip())
            if m_avg:
                split = m_avg.group("split")
                val = float(m_avg.group("val"))
                if split == "in_val":
                    in_val = val
                elif split == "test":
                    test = val

            m_corr = CORR_RE.search(line)
            if m_corr:
                corr_fro = float(m_corr.group("corr_fro"))
                corr_abs = float(m_corr.group("corr_abs"))

    return {
        "in_val_acc": in_val,
        "test_acc": test,
        "corr_fro_rel": corr_fro,
        "corr_abs_mean": corr_abs,
    }


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return (math.nan, math.nan)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate workshop ColoredMNIST logs")
    parser.add_argument(
        "--results-dir",
        default="results/workshop_coloredmnist",
        help="Directory containing logs and manifest",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs_root = results_dir / "logs"
    out_dir = results_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_files = sorted(logs_root.glob("*/*.log"))
    if not log_files:
        print(f"No logs found under {logs_root}")
        return 1

    parsed_rows = []
    for log_path in log_files:
        meta = parse_run_name(log_path)
        metrics = parse_log(log_path)
        parsed_rows.append({
            **meta,
            "log_path": str(log_path),
            **metrics,
        })

    all_runs_csv = out_dir / "all_runs.csv"
    with all_runs_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_name", "solver", "projection", "param1", "param2", "param3", "seed",
            "in_val_acc", "test_acc", "corr_fro_rel", "corr_abs_mean", "log_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in parsed_rows:
            writer.writerow(row)

    by_method_seed: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in parsed_rows:
        by_method_seed[(row["solver"], row["seed"])].append(row)

    selected_rows = []
    for (solver, seed), rows in by_method_seed.items():
        valid = [r for r in rows if r["in_val_acc"] is not None and r["test_acc"] is not None]
        if not valid:
            continue
        best = max(valid, key=lambda r: r["in_val_acc"])
        selected_rows.append(best)

    if not selected_rows:
        print(
            "No selectable runs with both in_val and test metrics were found. "
            "Failing aggregation."
        )
        return 1

    selected_csv = out_dir / "selected_by_seed.csv"
    with selected_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_name", "solver", "projection", "param1", "param2", "param3", "seed",
            "in_val_acc", "test_acc", "corr_fro_rel", "corr_abs_mean", "log_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    summary_rows = []
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for row in selected_rows:
        by_method[row["solver"]].append(row)

    for solver, rows in sorted(by_method.items()):
        in_vals = [float(r["in_val_acc"]) for r in rows if r["in_val_acc"] is not None]
        test_vals = [float(r["test_acc"]) for r in rows if r["test_acc"] is not None]
        corr_vals = [float(r["corr_fro_rel"]) for r in rows if r["corr_fro_rel"] is not None]

        in_mean, in_std = mean_std(in_vals)
        test_mean, test_std = mean_std(test_vals)
        corr_mean, corr_std = mean_std(corr_vals)

        summary_rows.append(
            {
                "solver": solver,
                "n_selected": len(rows),
                "in_val_acc_mean": in_mean,
                "in_val_acc_std": in_std,
                "test_acc_mean": test_mean,
                "test_acc_std": test_std,
                "corr_fro_rel_mean": corr_mean,
                "corr_fro_rel_std": corr_std,
            }
        )

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "solver", "n_selected", "in_val_acc_mean", "in_val_acc_std",
            "test_acc_mean", "test_acc_std", "corr_fro_rel_mean", "corr_fro_rel_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Method | N Seeds | In-Val Acc (mean+-std) | Test Acc (mean+-std) | Corr Fro Rel (mean+-std) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {solver} | {n_selected} | {in_val_acc_mean:.4f}+-{in_val_acc_std:.4f} | "
                "{test_acc_mean:.4f}+-{test_acc_std:.4f} | {corr_fro_rel_mean:.4f}+-{corr_fro_rel_std:.4f} |\n".format(
                    **row
                )
            )

    print(f"Wrote: {all_runs_csv}")
    print(f"Wrote: {selected_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
