"""Match exported W&B runs and sweep/config groups to GRIT Table 1 targets."""

from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMP_DATA_DIR = PROJECT_ROOT / "wandb_data" / "CMP"
VOLATILE_CONFIG_KEYS = {
    "device",
    "no_wandb",
    "root_dir",
    "seed",
    "wandb",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def as_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        nested = as_mapping(item)
        if nested is None:
            result[path] = item
        else:
            result.update(flatten(nested, path))
    return result


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_number(flat: Mapping[str, Any], aliases: Iterable[str]) -> Optional[float]:
    normalized = {
        key.lower().replace("/", "."): value for key, value in flat.items()
    }
    for alias in aliases:
        result = numeric(normalized.get(alias.lower().replace("/", ".")))
        if result is not None:
            return result
    return None


def metric_options(run: Mapping[str, Any], dataset: str) -> list[dict[str, Any]]:
    flat = flatten(run.get("summary") or {})
    if dataset == "colored_mnist":
        definitions = (
            (
                "in-domain selected scalar summary",
                ("in_test_val_best_in_test",),
                ("in_test_val_best_test",),
            ),
            (
                "latest nested split summary",
                ("in_test.acc_avg", "in_test_acc_avg"),
                ("test.acc_avg", "test_acc_avg"),
            ),
        )
    else:
        definitions = (
            (
                "latest nested test summary",
                ("test.acc_avg", "test_acc_avg"),
                ("test.acc_wg", "test_acc_wg"),
            ),
        )
    options = []
    for source, in_aliases, out_aliases in definitions:
        in_metric = first_number(flat, in_aliases)
        out_metric = first_number(flat, out_aliases)
        if in_metric is not None and out_metric is not None:
            options.append(
                {
                    "source": source,
                    "in_metric": in_metric,
                    "out_metric": out_metric,
                }
            )
    return options


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def model_matches(config: Mapping[str, Any], model: str, mapping: Mapping[str, Any]) -> bool:
    if model == "reference":
        return False
    representation = mapping["representation"][model]
    featurizer = str(config.get("featurizer", "")).lower()
    accepted_featurizers = {
        str(value).lower() for value in representation.get("featurizer_values", [])
    }
    if model == "probing":
        accepted_pretrained = representation["pretrained_values"]
        pretrained = config.get("pretrained")
        pretrained_match = any(
            truthy(pretrained) == truthy(value) for value in accepted_pretrained
        )
        return pretrained_match and featurizer in accepted_featurizers
    if featurizer in accepted_featurizers:
        return True
    if model == "finetune":
        return any(
            "finetune" in str(key).lower() and truthy(value)
            for key, value in config.items()
        )
    return False


def strict_filter(
    run: Mapping[str, Any], target: Mapping[str, Any], mapping: Mapping[str, Any]
) -> bool:
    if run.get("state") != "finished":
        return False
    config = run.get("config") or {}
    dataset_spec = mapping["dataset"][target["dataset"]]
    if config.get(mapping["dataset"]["config_key"]) not in dataset_spec["code_values"]:
        return False
    solver = mapping["method"]["paper_to_code"].get(target.get("method"))
    if config.get(mapping["method"]["config_key"]) != solver:
        return False
    if not model_matches(config, target["model"], mapping):
        return False
    pairing = target.get("pairing")
    if pairing is not None:
        expected_projection = mapping["pairing"]["paper_to_code"][pairing]
        if config.get(mapping["pairing"]["config_key"]) != expected_projection:
            return False
    expected_split = target.get("split_scheme")
    if expected_split is not None:
        actual_split = config.get("split_scheme", config.get("split"))
        if actual_split != expected_split:
            return False
    return True


def values_equal(left: Any, right: Any) -> bool:
    left_number = numeric(left)
    right_number = numeric(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def config_mismatches(
    config: Mapping[str, Any], target: Mapping[str, Any], targets: Mapping[str, Any]
) -> list[str]:
    if target["model"] != "probing":
        return []
    known = targets["known_hyperparameters"][target["dataset"]]
    mismatches = []
    for key in ("lr", "weight_decay", "batch_size", "epochs"):
        expected = known[key]
        actual = config.get(key)
        if not values_equal(actual, expected):
            mismatches.append(f"{key}={actual!r} (paper: {expected!r})")
    if target["method"] == "GRIT":
        rank = numeric(config.get("param1"))
        lower, upper = known["rank_range_inclusive"]
        if rank is None or not lower <= rank <= upper:
            mismatches.append(
                f"param1/rank={config.get('param1')!r} (paper: [{lower}, {upper}])"
            )
        # Do not narrow on param2. Although some ColoredMNIST sweeps use it as
        # a nominal pair-count setting, the checked-in ECMP pair subsampling is
        # commented out. The report preserves the full config so discrepancies
        # such as param2=1024 versus the paper's 256 pairs remain visible.
    return mismatches


def provenance_warnings(
    run: Mapping[str, Any], target: Mapping[str, Any], targets: Mapping[str, Any]
) -> list[str]:
    config = run.get("config") or {}
    warnings = []
    if target["dataset"] == "colored_mnist":
        warnings.append(
            f"dataset config is {config.get('dataset')!r}; ColoredMNIST and "
            "LISAColoredMNIST are distinct checked-in constructions"
        )
    if target["method"] == "GRIT" and target["dataset"] == "colored_mnist":
        expected_pairs = targets["known_hyperparameters"]["colored_mnist"][
            "counterfactual_pairs"
        ]
        if "param2" in config and not values_equal(config["param2"], expected_pairs):
            warnings.append(
                f"param2={config['param2']!r} differs from the paper's "
                f"{expected_pairs} counterfactual pairs; checked-in subsampling is commented out"
            )
    if not run.get("dataset_artifacts"):
        warnings.append("no W&B dataset artifact/hash is attached")
    if "optimizer" not in config:
        warnings.append("optimizer is absent from W&B config and cannot be verified from config alone")
    return warnings


def best_run_candidate(
    run: Mapping[str, Any], target: Mapping[str, Any], targets: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    options = metric_options(run, target["dataset"])
    if not options:
        return None
    for option in options:
        option["in_delta"] = option["in_metric"] - target["in_metric"]
        option["out_delta"] = option["out_metric"] - target["out_metric"]
        option["distance"] = abs(option["in_delta"]) + abs(option["out_delta"])
    best = min(options, key=lambda item: (item["distance"], item["source"]))
    return {
        "run": run,
        "metric": best,
        "config_mismatches": config_mismatches(run.get("config") or {}, target, targets),
        "provenance_warnings": provenance_warnings(run, target, targets),
    }


def narrow_by_known_config(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    exact = [candidate for candidate in candidates if not candidate["config_mismatches"]]
    if exact:
        return exact, "Restricted to candidates matching all logged appendix hyperparameters."
    return candidates, (
        "No metric-bearing candidate matched every logged appendix hyperparameter; "
        "ranking uses the strict schema-filtered fallback set."
    )


def signature_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(config.items())
        if key not in VOLATILE_CONFIG_KEYS
    }


def aggregate_candidates(
    candidates: list[dict[str, Any]],
    target: Mapping[str, Any],
    sweeps: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        run = candidate["run"]
        shared_config = signature_config(run.get("config") or {})
        signature = json.dumps(shared_config, sort_keys=True, separators=(",", ":"))
        sweep_id = run.get("sweep_id") or "no-sweep"
        key = (sweep_id, signature, candidate["metric"]["source"])
        groups[key].append(candidate)

    aggregates = []
    for (sweep_id, signature, metric_source), members in groups.items():
        in_metric = statistics.fmean(member["metric"]["in_metric"] for member in members)
        out_metric = statistics.fmean(member["metric"]["out_metric"] for member in members)
        in_delta = in_metric - target["in_metric"]
        out_delta = out_metric - target["out_metric"]
        aggregates.append(
            {
                "sweep_id": None if sweep_id == "no-sweep" else sweep_id,
                "sweep": sweeps.get(sweep_id),
                "shared_config": json.loads(signature),
                "metric_source": metric_source,
                "in_metric": in_metric,
                "out_metric": out_metric,
                "in_delta": in_delta,
                "out_delta": out_delta,
                "distance": abs(in_delta) + abs(out_delta),
                "run_ids": sorted(member["run"]["id"] for member in members),
                "run_count": len(members),
            }
        )
    return sorted(
        aggregates,
        key=lambda item: (item["distance"], -item["run_count"], item["sweep_id"] or ""),
    )


def json_block(value: Any) -> list[str]:
    return ["```json", json.dumps(value, indent=2, sort_keys=True), "```"]


def fmt(value: float) -> str:
    return f"{value:.6f}"


def reported_value_match(metric: Mapping[str, Any]) -> bool:
    """Allow half of the paper's three-decimal reporting unit per metric."""

    return (
        abs(metric["in_delta"]) <= 0.00051
        and abs(metric["out_delta"]) <= 0.00051
    )


def ranked_candidates(
    target: Mapping[str, Any],
    runs: list[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    strict_runs = [run for run in runs if strict_filter(run, target, mapping)]
    candidates = [
        candidate
        for run in strict_runs
        if (candidate := best_run_candidate(run, target, targets)) is not None
    ]
    narrowed, _ = narrow_by_known_config(candidates)
    narrowed.sort(
        key=lambda item: (
            item["metric"]["distance"],
            len(item["config_mismatches"]),
            item["run"]["created_at"] or "",
        )
    )
    return narrowed, len(strict_runs), len(candidates)


def render_executive_summary(
    targets: Mapping[str, Any],
    runs: list[Mapping[str, Any]],
    mapping: Mapping[str, Any],
) -> list[str]:
    lines = [
        "## Result overview",
        "",
        "A `reported-value match` means both logged metrics are within 0.00051 of the paper values (the tolerance implied by three-decimal reporting). It is a numerical match after the strict schema filter, not proof that no other run could share the printed values. `Closest only` is deliberately not claimed as the source run.",
        "",
        "| Dataset | Paper row | Status | Best run | Sweep | Logged in / out | Paper in / out | L1 |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    headline_conflicts = []
    reported_matches = 0
    matchable_count = 0
    for target in targets["targets"]:
        if target.get("matchable", True) is False:
            continue
        matchable_count += 1
        candidates, _, _ = ranked_candidates(target, runs, mapping, targets)
        if not candidates:
            status = "Unresolved"
            run_id = sweep_id = "—"
            logged = l1 = "—"
        else:
            candidate = candidates[0]
            run = candidate["run"]
            metric = candidate["metric"]
            is_match = reported_value_match(metric)
            status = "Reported-value match" if is_match else "Closest only"
            reported_matches += int(is_match)
            run_id = f"`{run['id']}`"
            sweep_id = f"`{run.get('sweep_id') or 'no-sweep'}`"
            logged = f"{metric['in_metric']:.6f} / {metric['out_metric']:.6f}"
            l1 = f"{metric['distance']:.6f}"
            if target.get("headline") and not is_match:
                headline_conflicts.append(
                    f"{target['dataset']} {target['label']}: closest strict run "
                    f"{run['id']} in sweep {run.get('sweep_id') or 'no-sweep'} logs "
                    f"{logged}, not {target['in_metric']:.3f} / {target['out_metric']:.3f}"
                )
        lines.append(
            f"| {target['dataset']} | {target['label']} | {status} | {run_id} | "
            f"{sweep_id} | {logged} | {target['in_metric']:.3f} / "
            f"{target['out_metric']:.3f} | {l1} |"
        )
    lines.extend(
        [
            "",
            f"Numerically matched after strict filtering: `{reported_matches}/{matchable_count}` matchable rows.",
            "",
        ]
    )
    if headline_conflicts:
        lines.extend(
            [
                "Headline conflicts:",
                "",
                *[f"- {conflict}." for conflict in headline_conflicts],
                "",
            ]
        )
    return lines


def render_target(
    target: Mapping[str, Any],
    runs: list[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    targets: Mapping[str, Any],
    sweeps: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines = [f"## {target['dataset']}: {target['label']}", ""]
    lines.append(
        f"Paper target: in `{target['in_metric']:.3f}`, out `{target['out_metric']:.3f}`."
    )
    lines.append("")
    if target.get("matchable", True) is False:
        lines.extend(
            [
                "This is a paper reference row rather than an implemented/logged method; no W&B run is selected.",
                "",
            ]
        )
        return lines

    narrowed, strict_count, candidate_count = ranked_candidates(
        target, runs, mapping, targets
    )
    lines.append(
        f"Strict schema filter: `{strict_count}` finished runs; "
        f"`{candidate_count}` expose a usable summary metric pair."
    )
    if not narrowed:
        lines.extend(
            [
                "",
                "No candidate can be selected from the exported summaries. This is reported as an unresolved row, not guessed.",
                "",
            ]
        )
        return lines

    _, narrowing_note = narrow_by_known_config(
        [
            candidate
            for run in runs
            if strict_filter(run, target, mapping)
            if (candidate := best_run_candidate(run, target, targets)) is not None
        ]
    )
    lines.extend([narrowing_note, ""])

    aggregates = aggregate_candidates(narrowed, target, sweeps)
    lines.extend(
        [
            "### Candidate sweep/config aggregates",
            "",
            "Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.",
            "",
            "| Rank | Sweep | Runs | In mean | Out mean | Δ in | Δ out | L1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, aggregate in enumerate(aggregates[:3], start=1):
        lines.append(
            "| {rank} | `{sweep}` | {count} | {in_metric} | {out_metric} | "
            "{in_delta:+.6f} | {out_delta:+.6f} | {distance:.6f} |".format(
                rank=rank,
                sweep=aggregate["sweep_id"] or "no-sweep",
                count=aggregate["run_count"],
                in_metric=fmt(aggregate["in_metric"]),
                out_metric=fmt(aggregate["out_metric"]),
                in_delta=aggregate["in_delta"],
                out_delta=aggregate["out_delta"],
                distance=aggregate["distance"],
            )
        )
    lines.append("")
    if len(aggregates) > 1 and aggregates[1]["distance"] - aggregates[0]["distance"] <= 0.002:
        lines.extend(
            [
                "**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.",
                "",
            ]
        )
    for rank, aggregate in enumerate(aggregates[:3], start=1):
        lines.extend(
            [
                f"<details><summary>Sweep/config aggregate {rank}: {aggregate['sweep_id'] or 'no-sweep'}</summary>",
                "",
                f"Metric source: `{aggregate['metric_source']}`; run IDs: `{', '.join(aggregate['run_ids'])}`.",
                "",
                "Shared run config:",
                "",
                *json_block(aggregate["shared_config"]),
                "",
            ]
        )
        if aggregate["sweep"] is not None:
            lines.extend(
                [
                    "Full sweep config:",
                    "",
                    *json_block(aggregate["sweep"].get("config") or {}),
                    "",
                ]
            )
        lines.extend(["</details>", ""])

    lines.extend(
        [
            "### Top individual runs",
            "",
            "| Rank | Run | Name | Sweep | Metric source | Logged in/out | Δ in/out | L1 |",
            "|---:|---|---|---|---|---|---|---:|",
        ]
    )
    for rank, candidate in enumerate(narrowed[:3], start=1):
        run = candidate["run"]
        metric = candidate["metric"]
        lines.append(
            f"| {rank} | `{run['id']}` | {run['name']} | `{run.get('sweep_id') or 'no-sweep'}` | "
            f"{metric['source']} | {fmt(metric['in_metric'])} / {fmt(metric['out_metric'])} | "
            f"{metric['in_delta']:+.6f} / {metric['out_delta']:+.6f} | {metric['distance']:.6f} |"
        )
    lines.append("")
    if len(narrowed) > 1 and narrowed[1]["metric"]["distance"] - narrowed[0]["metric"]["distance"] <= 0.002:
        lines.extend(
            [
                "**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.",
                "",
            ]
        )
    for rank, candidate in enumerate(narrowed[:3], start=1):
        run = candidate["run"]
        all_artifacts = run.get("artifacts") or []
        dataset_artifacts = run.get("dataset_artifacts") or []
        code_artifacts = [
            artifact for artifact in all_artifacts if artifact.get("type") == "code"
        ]
        lines.extend(
            [
                f"<details><summary>Run candidate {rank}: {run['id']} ({run['name']})</summary>",
                "",
                f"Created: `{run.get('created_at')}`; state: `{run.get('state')}`; URL: {run.get('url') or 'unavailable'}.",
                "",
                f"Appendix-config mismatches: `{candidate['config_mismatches'] or 'none'}`.",
                "",
                f"Provenance warnings: `{candidate['provenance_warnings'] or 'none'}`.",
                "",
                f"Logged dataset artifacts: `{len(dataset_artifacts)}`; code artifacts: `{len(code_artifacts)}`.",
                "",
                "Full run config:",
                "",
                *json_block(run.get("config") or {}),
                "",
                "Dataset artifacts:",
                "",
                *json_block(dataset_artifacts),
                "",
                "Code artifacts:",
                "",
                *json_block(code_artifacts),
                "",
                "</details>",
                "",
            ]
        )
    return lines


def render_report(
    run_dump: Mapping[str, Any],
    sweep_dump: Mapping[str, Any],
    mapping: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> str:
    runs = list(run_dump["runs"])
    sweep_records = list(sweep_dump["sweeps"])
    sweeps = {sweep["id"]: sweep for sweep in sweep_records}
    state_counts = Counter(run.get("state") for run in runs)
    dataset_counts = Counter((run.get("config") or {}).get("dataset") for run in runs)
    solver_counts = Counter((run.get("config") or {}).get("solver") for run in runs)
    artifact_type_counts = Counter(
        artifact.get("type")
        for run in runs
        for artifact in (run.get("artifacts") or [])
    )
    dataset_artifact_runs = [run for run in runs if run.get("dataset_artifacts")]

    lines = [
        "# GRIT Table 1 W&B match report",
        "",
        f"Export: `{run_dump.get('entity')}/{run_dump.get('project')}`, "
        f"`{len(runs)}` runs and `{len(sweep_records)}` sweeps.",
        "",
        "This report separates strict schema filtering from numerical closeness. It does not silently alias `ColoredMNIST` with `LISAColoredMNIST`, treat plain `Waterbirds` as Waterbirds-CF, or invent a finetuning flag that is absent from the base parser.",
        "",
        *render_executive_summary(targets, runs, mapping),
        "## Verified paper-to-code mapping",
        "",
        "| Paper term | Code config |",
        "|---|---|",
        "| GRIT | `solver=ECMP` |",
        "| ERM / IRM / REx / GroupDRO / Fish / SWAD / LISA / MatchDG | same value in `solver` |",
        "| random pairs | `projection=conditional` |",
        "| 1NN pairs | `projection=nearest` |",
        "| clean pairs | `projection=oracle` |",
        "| rank r for GRIT | `param1` |",
        "| frozen CLIP linear probe | `pretrained=true`, `featurizer=linear` |",
        "| ColoredMNIST code values | `ColoredMNIST`, `LISAColoredMNIST` (kept distinct in output) |",
        "| Waterbirds-CF code value | `CounterfactualWaterbirds` |",
        "",
        "The complete mapping, including overloaded `param1` meanings and optimizer caveats, is in `wandb_data/CMP/config_mapping.json`.",
        "",
        "## Project inventory",
        "",
        f"Run states: `{dict(state_counts)}`.",
        "",
        f"Observed dataset config values: `{dict(dataset_counts)}`.",
        "",
        f"Observed solver config values: `{dict(solver_counts)}`.",
        "",
        f"Runs with a logged or used W&B artifact whose type is exactly `dataset`: `{len(dataset_artifact_runs)}`.",
        "",
        f"Artifact types: `{dict(artifact_type_counts)}`.",
        "",
    ]
    if not dataset_artifact_runs:
        lines.extend(
            [
                "**Dataset provenance limitation:** no run references a W&B artifact typed `dataset`; root paths in run configs cannot establish the Waterbirds-CF metadata hash or confirm the 184/56 counterfactual composition.",
                "",
            ]
        )
    for target in targets["targets"]:
        lines.extend(render_target(target, runs, mapping, targets, sweeps))
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=CMP_DATA_DIR / "wandb_runs.json"
    )
    parser.add_argument(
        "--sweeps", type=Path, default=CMP_DATA_DIR / "wandb_sweeps.json"
    )
    parser.add_argument(
        "--mapping", type=Path, default=CMP_DATA_DIR / "config_mapping.json"
    )
    parser.add_argument(
        "--targets", type=Path, default=CMP_DATA_DIR / "paper_targets.json"
    )
    parser.add_argument(
        "--output", type=Path, default=CMP_DATA_DIR / "match_report.md"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dump = load_json(args.runs)
    sweep_dump = load_json(args.sweeps)
    if not run_dump.get("complete") or not sweep_dump.get("complete"):
        raise RuntimeError("Refusing to match an incomplete W&B export")
    report = render_report(
        run_dump,
        sweep_dump,
        load_json(args.mapping),
        load_json(args.targets),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
