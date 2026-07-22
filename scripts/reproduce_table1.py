#!/usr/bin/env python3
"""Prepare, run, and summarize the ColoredMNIST Table 1 reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reproduction" / "table1_cmnist.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reproduction_results" / "cmnist"
DATA_ENV_VAR = "GRIT_DATA_ROOT"
DATASET_DIRECTORY = "LISAColoredMNIST-cf-clip_v1.0"
REQUIRED_DATA_FILES = (
    "x_array.pth",
    "y_array.pth",
    "split_array.pth",
    "metadata_array.pth",
    "diff.pth",
    "reproduction_manifest.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_dump(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=str(output_path.parent),
        prefix=output_path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(str(temporary_path), str(output_path))


def resolve_data_root(
    cli_value: Path | None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environ = os.environ if environ is None else environ
    if cli_value is not None:
        value = cli_value
    elif environ.get(DATA_ENV_VAR):
        value = Path(environ[DATA_ENV_VAR])
    else:
        value = PROJECT_ROOT / "data"
    return value.expanduser().resolve()


def resolve_output_root(value: Path | None) -> Path:
    return (value or DEFAULT_OUTPUT_ROOT).expanduser().resolve()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open() as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction manifest schema")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The reproduction manifest must contain rows")
    row_ids = [row.get("id") for row in rows]
    if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("Every reproduction row must have a unique non-empty id")
    for row in rows:
        if row.get("status") == "run":
            if "solver" not in row.get("fixed", {}):
                raise ValueError(f"Runnable row {row['id']} has no solver")
            if "grid" not in row or "selected" not in row:
                raise ValueError(f"Runnable row {row['id']} lacks grid metadata")


def runnable_rows(
    manifest: Mapping[str, Any], row_ids: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    rows_by_id = {row["id"]: row for row in manifest["rows"]}
    if row_ids:
        unknown = sorted(set(row_ids) - set(rows_by_id))
        if unknown:
            raise ValueError(f"Unknown row id(s): {', '.join(unknown)}")
        selected = [rows_by_id[row_id] for row_id in row_ids]
    else:
        selected = list(manifest["rows"])
    return [row for row in selected if row["status"] == "run"]


def grid_values(grid: Mapping[str, Sequence[Any]]) -> Iterable[dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def stable_hash(payload: Mapping[str, Any], length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def dataset_fingerprint(manifest: Mapping[str, Any]) -> str:
    identity = {
        "dataset": manifest.get("dataset"),
        "seed": manifest.get("seed"),
        "clip_model": manifest.get("clip_model"),
        "clip_commit": manifest.get("clip_commit"),
        "files": {
            name: details.get("sha256")
            for name, details in sorted(manifest.get("files", {}).items())
        },
    }
    return stable_hash(identity, length=16)


def expand_runs(
    manifest: Mapping[str, Any],
    *,
    quick: bool = False,
    seeds: Sequence[int] = (1001,),
    row_ids: Sequence[str] | None = None,
    epochs: int | None = None,
    dataset_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    expanded = []
    for row in runnable_rows(manifest, row_ids):
        variants = [row["selected"]] if quick else list(grid_values(row["grid"]))
        for grid_index, variant in enumerate(variants):
            base_config = dict(manifest["defaults"])
            base_config.update(row.get("fixed", {}))
            base_config.update(variant)
            if epochs is not None:
                base_config["epochs"] = epochs
            identity = {
                "row_id": row["id"],
                "config": base_config,
                "dataset_fingerprint": dataset_fingerprint,
            }
            config_id = stable_hash(identity)
            for seed in seeds:
                config = dict(base_config)
                config["seed"] = int(seed)
                run_identity = dict(identity)
                run_identity["config"] = config
                digest = stable_hash(run_identity)
                expanded.append(
                    {
                        "run_id": f"{row['id']}-{digest}",
                        "config_id": config_id,
                        "row_id": row["id"],
                        "row_label": row["label"],
                        "grid_index": grid_index,
                        "dataset_fingerprint": dataset_fingerprint,
                        "config": config,
                    }
                )
    return expanded


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def training_command(
    run: Mapping[str, Any],
    *,
    data_root: Path,
    result_path: Path,
    device: str,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--root_dir",
        str(data_root),
        "--output_json",
        str(result_path),
        "--device",
        device,
        "--deterministic",
    ]
    for key, value in run["config"].items():
        command.extend((f"--{key}", _format_cli_value(value)))
    if wandb_project:
        command.extend(("--wandb_project", wandb_project))
        if wandb_entity:
            command.extend(("--wandb_entity", wandb_entity))
        if wandb_group:
            command.extend(("--wandb_group", wandb_group))
    else:
        command.append("--no_wandb")
    return command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_prepared_dataset(dataset_dir: Path, verify_hashes: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_DATA_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Prepared dataset is incomplete at {dataset_dir}; missing: {', '.join(missing)}"
        )
    with (dataset_dir / "reproduction_manifest.json").open() as handle:
        manifest = json.load(handle)
    expected_manifest_files = set(REQUIRED_DATA_FILES) - {"reproduction_manifest.json"}
    missing_manifest_files = expected_manifest_files - set(manifest.get("files", {}))
    if missing_manifest_files:
        raise ValueError(
            "Dataset manifest is incomplete; missing hashes for: "
            + ", ".join(sorted(missing_manifest_files))
        )
    if verify_hashes:
        for name, details in manifest.get("files", {}).items():
            path = dataset_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Manifest file is missing: {path}")
            actual = sha256_file(path)
            if actual != details.get("sha256"):
                raise ValueError(f"Dataset hash mismatch for {path}")
    return manifest


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_torch_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for preprocessing, but it is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset(args: argparse.Namespace) -> int:
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm.auto import tqdm

    sys.path.insert(0, str(PROJECT_ROOT))
    from datasets import LISAColoredMNISTDataset
    from models import Clip

    data_root = resolve_data_root(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    if not os.access(str(data_root), os.W_OK):
        raise PermissionError(f"Data root is not writable: {data_root}")

    dataset_dir = data_root / DATASET_DIRECTORY
    if dataset_dir.exists() and not args.force:
        validate_prepared_dataset(dataset_dir, verify_hashes=not args.skip_hash_check)
        print(f"Prepared dataset is already valid: {dataset_dir}")
        return 0
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{DATASET_DIRECTORY}-", dir=str(data_root))
    )
    device = choose_torch_device(args.device)
    clip_cache = data_root / ".cache" / "clip"
    clip_cache.mkdir(parents=True, exist_ok=True)

    try:
        seed_everything(args.seed)
        preprocessor = Clip(
            {
                "input_shape": (2, 28, 28),
                "clip_download_root": str(clip_cache),
            }
        ).to(device)
        preprocessor.eval()

        mnist = torchvision.datasets.MNIST(
            str(data_root), train=True, download=True, transform=transforms.ToTensor()
        )
        mnist_loader = DataLoader(
            mnist, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        difference_batches = []
        seen = 0
        with torch.no_grad():
            for images, _ in tqdm(mnist_loader, desc="Oracle CLIP differences"):
                if seen >= 50000:
                    break
                images = images[: 50000 - seen]
                empty = torch.zeros_like(images)
                red = torch.cat((images, empty), dim=1)
                green = torch.cat((empty, images), dim=1)
                difference_batches.append(
                    (preprocessor(red.to(device)) - preprocessor(green.to(device))).cpu()
                )
                seen += len(images)
        difference_tensor = torch.cat(difference_batches)
        torch.save(difference_tensor, temporary_dir / "diff.pth")

        # Dataset construction consumes random Bernoulli samples. Reseeding here makes
        # it independent of batching and CLIP preprocessing above.
        seed_everything(args.seed)
        dataset = LISAColoredMNISTDataset(root_dir=str(data_root), download=False)
        dataset_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        feature_batches = []
        with torch.no_grad():
            for images, _, _ in tqdm(dataset_loader, desc="LISAColoredMNIST CLIP features"):
                feature_batches.append(preprocessor(images.to(device)).cpu())
        features = torch.cat(feature_batches)

        torch.save(dataset._split_array, temporary_dir / "split_array.pth")
        torch.save(features, temporary_dir / "x_array.pth")
        torch.save(dataset._y_array, temporary_dir / "y_array.pth")
        torch.save(dataset._metadata_array, temporary_dir / "metadata_array.pth")

        tensor_shapes = {
            "x_array.pth": list(features.shape),
            "y_array.pth": list(dataset._y_array.shape),
            "split_array.pth": list(dataset._split_array.shape),
            "metadata_array.pth": list(dataset._metadata_array.shape),
            "diff.pth": list(difference_tensor.shape),
        }
        files = {}
        for name, shape in tensor_shapes.items():
            path = temporary_dir / name
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": shape,
            }
        manifest = {
            "schema_version": 1,
            "dataset": "LISAColoredMNIST",
            "directory": DATASET_DIRECTORY,
            "created_at": utc_now(),
            "seed": args.seed,
            "git_commit": git_commit(),
            "clip_model": "ViT-B/32",
            "clip_commit": "a1d071733d7111c9c014f024669f959182114e33",
            "device": str(device),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": installed_version("numpy"),
                "wilds": installed_version("wilds"),
            },
            "files": files,
        }
        atomic_json_dump(manifest, temporary_dir / "reproduction_manifest.json")

        backup_dir = None
        if dataset_dir.exists():
            backup_dir = data_root / f".{DATASET_DIRECTORY}-backup"
            if backup_dir.exists():
                raise FileExistsError(
                    f"Cannot replace the dataset while a backup exists: {backup_dir}"
                )
            os.replace(str(dataset_dir), str(backup_dir))
        try:
            os.replace(str(temporary_dir), str(dataset_dir))
        except BaseException:
            if backup_dir is not None and backup_dir.exists():
                os.replace(str(backup_dir), str(dataset_dir))
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"Prepared LISAColoredMNIST at {dataset_dir}")
    return 0


def run_experiments(args: argparse.Namespace) -> int:
    if args.wandb_entity and not args.wandb_project:
        raise ValueError("--wandb-entity requires --wandb-project")
    manifest = load_manifest(args.manifest)
    data_root = resolve_data_root(args.data_root)
    output_root = resolve_output_root(args.output_dir)
    expansion_options = {
        "quick": args.quick,
        "seeds": args.seeds,
        "row_ids": args.rows,
        "epochs": args.epochs,
    }
    runs = expand_runs(
        manifest,
        **expansion_options,
    )
    print(f"Planned runs: {len(runs)}")

    if args.dry_run:
        for run in runs:
            run_dir = output_root / "runs" / run["run_id"]
            command = training_command(
                run,
                data_root=data_root,
                result_path=run_dir / "result.json",
                device=args.device,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=args.wandb_group,
            )
            print(shlex.join(command))
        return 0

    dataset_dir = data_root / DATASET_DIRECTORY
    data_manifest = validate_prepared_dataset(
        dataset_dir, verify_hashes=not args.skip_data_hash_check
    )
    prepared_fingerprint = dataset_fingerprint(data_manifest)
    runs = expand_runs(
        manifest,
        dataset_fingerprint=prepared_fingerprint,
        **expansion_options,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "manifest": str(args.manifest.resolve()),
            "data_root": str(data_root),
            "dataset_fingerprint": prepared_fingerprint,
            "quick": args.quick,
            "seeds": args.seeds,
            "rows": args.rows,
            "run_count": len(runs),
            "run_ids": [run["run_id"] for run in runs],
        },
        output_root / "run_plan.json",
    )

    completed = skipped = failed = 0
    for position, run in enumerate(runs, 1):
        run_dir = output_root / "runs" / run["run_id"]
        result_path = run_dir / "result.json"
        record_path = run_dir / "record.json"
        if record_path.is_file() and result_path.is_file() and not args.force:
            with record_path.open() as handle:
                existing = json.load(handle)
            if existing.get("status") == "completed":
                skipped += 1
                print(f"[{position}/{len(runs)}] skip {run['run_id']}")
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = training_command(
            run,
            data_root=data_root,
            result_path=result_path,
            device=args.device,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
        )
        record = dict(run)
        record.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "command": command,
                "result_path": result_path.name,
            }
        )
        atomic_json_dump(record, record_path)
        print(f"[{position}/{len(runs)}] run {run['run_id']}")
        with (run_dir / "train.log").open("w") as log_handle:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["finished_at"] = utc_now()
        record["returncode"] = process.returncode
        if process.returncode == 0 and result_path.is_file():
            record["status"] = "completed"
            completed += 1
        else:
            record["status"] = "failed"
            failed += 1
        atomic_json_dump(record, record_path)
        if record["status"] == "failed":
            print(f"  failed; see {run_dir / 'train.log'}", file=sys.stderr)
            if args.fail_fast:
                break

    print(f"Completed: {completed}; skipped: {skipped}; failed: {failed}")
    summarize_results(manifest, output_root)
    return 1 if failed else 0


def _selected_metrics(result: Mapping[str, Any], selection: Mapping[str, str]) -> tuple[float, float, int | None]:
    selected = result["selection"]["in_domain"]
    metrics = selected["metrics"]
    in_value = float(metrics[selection["split"]][selection["metric"]])
    test_value = float(metrics[selection["paired_test_split"]][selection["metric"]])
    return in_value, test_value, selected.get("step")


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty sequence")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_results(manifest: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    run_plan_path = output_root / "run_plan.json"
    run_plan = None
    allowed_run_ids = None
    if run_plan_path.is_file():
        with run_plan_path.open() as handle:
            run_plan = json.load(handle)
        if "run_ids" in run_plan:
            allowed_run_ids = set(run_plan["run_ids"])

    records_by_row: dict[str, list[dict[str, Any]]] = {}
    for record_path in sorted((output_root / "runs").glob("*/record.json")):
        with record_path.open() as handle:
            record = json.load(handle)
        if allowed_run_ids is not None and record.get("run_id") not in allowed_run_ids:
            continue
        if record.get("status") != "completed":
            continue
        result_path = Path(record["result_path"])
        if not result_path.is_absolute():
            result_path = record_path.parent / result_path
        if not result_path.is_file():
            continue
        with result_path.open() as handle:
            result = json.load(handle)
        try:
            in_value, test_value, step = _selected_metrics(result, manifest["selection"])
        except (KeyError, TypeError, ValueError):
            continue
        item = dict(record)
        item.update({"in": in_value, "test": test_value, "selected_step": step})
        records_by_row.setdefault(record["row_id"], []).append(item)

    output_rows = []
    for row in manifest["rows"]:
        output = {
            "id": row["id"],
            "label": row["label"],
            "source_sweep": row.get("source_sweep"),
            "paper_in": row["paper"]["in"],
            "paper_test": row["paper"]["test"],
            "status": row["status"],
            "note": row.get("note"),
        }
        if row["status"] == "reference":
            output.update(
                {
                    "reproduced_in": row["reference"]["in"],
                    "reproduced_test": row["reference"]["test"],
                    "in_std": 0.0,
                    "test_std": 0.0,
                    "run_count": 0,
                }
            )
        elif row["status"] == "unresolved":
            pass
        else:
            candidates = records_by_row.get(row["id"], [])
            grouped: dict[str, list[dict[str, Any]]] = {}
            for candidate in candidates:
                grouped.setdefault(candidate["config_id"], []).append(candidate)
            aggregates = []
            for config_id, members in grouped.items():
                mean_in, std_in = _mean_std([member["in"] for member in members])
                mean_test, std_test = _mean_std([member["test"] for member in members])
                aggregates.append(
                    {
                        "config_id": config_id,
                        "grid_index": min(member["grid_index"] for member in members),
                        "config": members[0]["config"],
                        "mean_in": mean_in,
                        "std_in": std_in,
                        "mean_test": mean_test,
                        "std_test": std_test,
                        "run_count": len(members),
                        "runs": [member["run_id"] for member in members],
                    }
                )
            if aggregates:
                best = sorted(aggregates, key=lambda item: (-item["mean_in"], item["grid_index"]))[0]
                output.update(
                    {
                        "status": "reproduced",
                        "reproduced_in": best["mean_in"],
                        "reproduced_test": best["mean_test"],
                        "in_std": best["std_in"],
                        "test_std": best["std_test"],
                        "delta_in": best["mean_in"] - row["paper"]["in"],
                        "delta_test": best["mean_test"] - row["paper"]["test"],
                        "run_count": best["run_count"],
                        "selected_config": best["config"],
                        "selected_runs": best["runs"],
                    }
                )
            else:
                output["status"] = "missing"
        output_rows.append(output)

    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "selection": manifest["selection"],
        "run_plan": run_plan,
        "rows": output_rows,
    }
    atomic_json_dump(payload, output_root / "results.json")
    write_csv(output_rows, output_root / "table1_cmnist.csv")
    write_markdown(output_rows, output_root / "table1_cmnist.md")
    print(f"Wrote summary to {output_root / 'table1_cmnist.md'}")
    return output_rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "id",
        "label",
        "status",
        "source_sweep",
        "paper_in",
        "paper_test",
        "reproduced_in",
        "in_std",
        "reproduced_test",
        "test_std",
        "delta_in",
        "delta_test",
        "run_count",
        "selected_config",
        "note",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            if "selected_config" in value:
                value["selected_config"] = json.dumps(value["selected_config"], sort_keys=True)
            writer.writerow(value)


def _metric(value: Any, std: Any = None) -> str:
    if value is None:
        return "—"
    if std not in (None, 0, 0.0):
        return f"{float(value):.4f} ± {float(std):.4f}"
    return f"{float(value):.4f}"


def _config_summary(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "—"
    keys = ("solver", "projection", "param1", "param2", "latent_dim", "seed")
    return ", ".join(f"{key}={config[key]}" for key in keys if key in config)


def write_markdown(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# ColoredMNIST Table 1 reproduction",
        "",
        "Models are selected by in-domain `in_test.acc_avg`; the reported OOD value is from the same epoch.",
        "All executed rows use `LISAColoredMNIST` with frozen CLIP ViT-B/32 features.",
        "",
        "| Row | Status | Paper in / test | Reproduced in / test | Δ in / test | Selected configuration |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        paper = f"{row['paper_in']:.3f} / {row['paper_test']:.3f}"
        reproduced = (
            f"{_metric(row.get('reproduced_in'), row.get('in_std'))} / "
            f"{_metric(row.get('reproduced_test'), row.get('test_std'))}"
        )
        delta = (
            f"{_metric(row.get('delta_in'))} / {_metric(row.get('delta_test'))}"
            if row.get("delta_in") is not None
            else "—"
        )
        lines.append(
            f"| {row['label']} | {row['status']} | {paper} | {reproduced} | "
            f"{delta} | {_config_summary(row.get('selected_config'))} |"
        )
    lines.extend(
        [
            "",
            "Unresolved rows are retained to keep the output aligned with the published table; no configuration is inferred for them.",
            "",
        ]
    )
    notes = [row for row in rows if row.get("note")]
    if notes:
        lines.extend(["## Provenance notes", ""])
        lines.extend(f"- **{row['label']}:** {row['note']}" for row in notes)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def summarize_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    summarize_results(manifest, resolve_output_root(args.output_dir))
    return 0


def add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Dataset root; overrides ${DATA_ENV_VAR}, then defaults to <repo>/data",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare deterministic CLIP tensors")
    add_shared_paths(prepare)
    prepare.add_argument("--seed", type=int, default=1001)
    prepare.add_argument("--batch-size", type=int, default=250)
    prepare.add_argument("--num-workers", type=int, default=4)
    prepare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--skip-hash-check", action="store_true")
    prepare.set_defaults(handler=prepare_dataset)

    run = subparsers.add_parser("run", help="Run the audited Table 1 configurations")
    add_shared_paths(run)
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--rows", nargs="+", default=None)
    run.add_argument("--seeds", nargs="+", type=int, default=[1001])
    run.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke tests")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--skip-data-hash-check", action="store_true")
    run.add_argument("--wandb-project", default=None)
    run.add_argument("--wandb-entity", default=None)
    run.add_argument("--wandb-group", default="table1-cmnist-reproduction")
    run.set_defaults(handler=run_experiments)

    summarize = subparsers.add_parser("summarize", help="Aggregate completed runs")
    summarize.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    summarize.add_argument("--output-dir", type=Path, default=None)
    summarize.set_defaults(handler=summarize_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
