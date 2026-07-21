"""Export public W&B runs, sweeps, artifacts, and optional histories.

The export intentionally preserves raw config, summary, and history values.
Older projects sometimes store nested metric dictionaries as strings, so
downstream code must not assume that W&B has already decoded them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar

import wandb


DEFAULT_ENTITY = "inouye-lab"
DEFAULT_PROJECT = "CMP"
DEFAULT_OUTPUT_DIR = Path("wandb_data") / DEFAULT_PROJECT
T = TypeVar("T")


def error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def retry_call(
    operation: Callable[[], T],
    max_retries: int,
    retry_backoff: float,
) -> T:
    """Run an API operation with bounded exponential-backoff retries."""

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception:
            if attempt == max_retries:
                raise
            if retry_backoff > 0:
                time.sleep(retry_backoff * (2**attempt))
    raise AssertionError("retry loop exited without returning or raising")


def json_safe(value: Any) -> Any:
    """Recursively convert W&B values to deterministic JSON-compatible data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        items = value.items()
    except Exception:
        items = None
    if items is not None:
        try:
            return {str(key): json_safe(item) for key, item in items}
        except Exception:
            pass
    # wandb.old.summary.SummarySubDict implements __getattr__ by raising
    # KeyError, which makes Python's hasattr() unsafe. Its payload is in _dict.
    try:
        object_attributes = vars(value)
    except Exception:
        object_attributes = {}
    if isinstance(object_attributes.get("_dict"), dict):
        return json_safe(object_attributes["_dict"])
    try:
        tolist = getattr(value, "tolist")
    except Exception:
        tolist = None
    if callable(tolist):
        try:
            return json_safe(tolist())
        except Exception:
            pass
    try:
        isoformat = getattr(value, "isoformat")
    except Exception:
        isoformat = None
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return str(value)


def atomic_json_dump(path: Path, value: Any) -> None:
    """Write JSON atomically so an interrupted export cannot corrupt a dump."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def atomic_jsonl_dump(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write newline-delimited JSON records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for record in records:
            json.dump(json_safe(record), handle, sort_keys=True)
            handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return None


def artifact_record(artifact: Any, relationship: str) -> dict[str, Any]:
    artifact_type = attr(artifact, "type")
    return {
        "relationship": relationship,
        "id": attr(artifact, "id"),
        "name": attr(artifact, "name"),
        "type": artifact_type,
        "is_dataset": str(artifact_type).lower() == "dataset",
        "version": attr(artifact, "version"),
        "digest": attr(artifact, "digest"),
        "state": attr(artifact, "state"),
        "size": attr(artifact, "size"),
        "aliases": attr(artifact, "aliases"),
        "created_at": attr(artifact, "created_at"),
        "updated_at": attr(artifact, "updated_at"),
        "entity": attr(artifact, "entity"),
        "project": attr(artifact, "project"),
        "qualified_name": attr(artifact, "qualified_name"),
    }


def collect_artifacts(
    run: Any, max_retries: int, retry_backoff: float
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect output and input artifacts without failing the whole export."""

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    # Public API Run.use_artifact() requires an already-known artifact name and
    # therefore cannot enumerate dependencies. used_artifacts() is its listing
    # counterpart; logged_artifacts() lists outputs produced by the run.
    sources: tuple[tuple[str, Any], ...] = (
        ("logged", run.logged_artifacts),
        ("used", run.used_artifacts),
    )
    for relationship, loader in sources:
        try:
            artifacts = retry_call(
                lambda loader=loader: list(loader()), max_retries, retry_backoff
            )
            records.extend(
                artifact_record(artifact, relationship) for artifact in artifacts
            )
        except Exception as error:
            errors.append(f"{relationship}: {error_text(error)}")
    return records, errors


def run_record(
    run: Any, max_retries: int = 0, retry_backoff: float = 0
) -> dict[str, Any]:
    record_errors: list[str] = []
    artifacts, artifact_errors = collect_artifacts(
        run, max_retries, retry_backoff
    )
    sweep = attr(run, "sweep")
    try:
        config = retry_call(
            lambda: json_safe(attr(run, "config") or {}),
            max_retries,
            retry_backoff,
        )
    except Exception as error:
        config = {}
        record_errors.append(f"config: {error_text(error)}")
    try:
        summary = retry_call(
            lambda: json_safe(dict(attr(run, "summary") or {})),
            max_retries,
            retry_backoff,
        )
    except Exception as error:
        summary = {}
        record_errors.append(f"summary: {error_text(error)}")
    return {
        "name": attr(run, "name"),
        "id": attr(run, "id"),
        "config": config,
        "summary": summary,
        "tags": json_safe(attr(run, "tags") or []),
        "state": attr(run, "state"),
        "sweep_id": attr(sweep, "id") if sweep is not None else None,
        "created_at": attr(run, "created_at"),
        "url": attr(run, "url"),
        "artifacts": artifacts,
        "dataset_artifacts": [
            artifact for artifact in artifacts if artifact["is_dataset"]
        ],
        "artifact_errors": artifact_errors,
        "record_errors": record_errors,
    }


def failed_run_record(run: Any, error: BaseException) -> dict[str, Any]:
    """Preserve a run identity when the full record cannot be serialized."""

    sweep = attr(run, "sweep")
    return {
        "name": attr(run, "name"),
        "id": attr(run, "id"),
        "config": {},
        "summary": {},
        "tags": [],
        "state": attr(run, "state"),
        "sweep_id": attr(sweep, "id") if sweep is not None else None,
        "created_at": attr(run, "created_at"),
        "url": attr(run, "url"),
        "artifacts": [],
        "dataset_artifacts": [],
        "artifact_errors": [],
        "record_errors": [f"record: {error_text(error)}"],
    }


def sweep_record(
    sweep: Any,
    fallback_run_ids: Optional[list[str]] = None,
    runs: Optional[list[Any]] = None,
    record_errors: Optional[list[str]] = None,
) -> dict[str, Any]:
    loaded_runs = runs if runs is not None else list(attr(sweep, "runs") or [])
    run_ids = [attr(run, "id") for run in loaded_runs]
    run_ids_source = "sweep.runs"
    if not run_ids and fallback_run_ids:
        run_ids = sorted(fallback_run_ids)
        run_ids_source = "run.sweep.id fallback"
    return {
        "id": attr(sweep, "id"),
        "name": attr(sweep, "name"),
        "state": attr(sweep, "state"),
        "config": json_safe(attr(sweep, "config") or {}),
        "run_ids": run_ids,
        "run_count": len(run_ids),
        "run_ids_source": run_ids_source,
        "url": attr(sweep, "url"),
        "record_errors": record_errors or [],
    }


def export_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    output_path: Path,
    checkpoint_every: int,
    workers: int,
    max_retries: int,
    retry_backoff: float,
    resume: bool = False,
) -> dict[str, Any]:
    exported_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    run_objects = retry_call(
        lambda: list(api.runs(f"{entity}/{project}")),
        max_retries,
        retry_backoff,
    )
    current_run_ids = {str(attr(run, "id")) for run in run_objects}
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("entity") != entity or previous.get("project") != project:
            raise ValueError(
                f"cannot resume {output_path}: entity/project does not match"
            )
        exported_at = previous.get("exported_at") or exported_at
        records_by_id = {
            str(run.get("id")): run
            for run in previous.get("runs", [])
            if run.get("id") is not None
            and str(run.get("id")) in current_run_ids
            and not run.get("record_errors")
            and not run.get("artifact_errors")
        }
        records = list(records_by_id.values())
    completed_run_ids = {str(run["id"]) for run in records}
    pending_runs = [
        run for run in run_objects if str(attr(run, "id")) not in completed_run_ids
    ]
    if completed_run_ids:
        print(
            f"runs: resuming with {len(completed_run_ids)}/{len(run_objects)} complete",
            file=sys.stderr,
            flush=True,
        )
    def accept_record(record: dict[str, Any], index: int) -> None:
        records.append(record)
        print(
            f"runs: {index}/{len(run_objects)} "
            f"({record['id']} {record['state']})",
            file=sys.stderr,
            flush=True,
        )
        if checkpoint_every > 0 and index % checkpoint_every == 0:
            atomic_json_dump(
                output_path,
                {
                    "entity": entity,
                    "project": project,
                    "exported_at": exported_at,
                    "complete": False,
                    "expected_run_count": len(run_objects),
                    "run_count": len(records),
                    "error_count": sum(
                        bool(run["record_errors"] or run["artifact_errors"])
                        for run in records
                    ),
                    "runs": records,
                },
            )

    if workers == 1:
        for offset, run in enumerate(pending_runs, start=1):
            index = len(completed_run_ids) + offset
            try:
                record = run_record(run, max_retries, retry_backoff)
            except Exception as error:
                record = failed_run_record(run, error)
            accept_record(record, index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_run = {
                executor.submit(
                    run_record, run, max_retries, retry_backoff
                ): run
                for run in pending_runs
            }
            for offset, future in enumerate(as_completed(future_to_run), start=1):
                index = len(completed_run_ids) + offset
                try:
                    record = future.result()
                except Exception as error:
                    record = failed_run_record(future_to_run[future], error)
                accept_record(record, index)
    records.sort(key=lambda run: (run.get("created_at") or "", run.get("id") or ""), reverse=True)
    error_count = sum(
        bool(run["record_errors"] or run["artifact_errors"])
        for run in records
    )
    payload = {
        "entity": entity,
        "project": project,
        "exported_at": exported_at,
        "complete": error_count == 0 and len(records) == len(run_objects),
        "expected_run_count": len(run_objects),
        "run_count": len(records),
        "error_count": error_count,
        "runs": records,
    }
    atomic_json_dump(output_path, payload)
    return payload


def export_sweeps(
    api: wandb.Api,
    entity: str,
    project: str,
    output_path: Path,
    workers: int,
    run_payload: Optional[Mapping[str, Any]] = None,
    max_retries: int = 0,
    retry_backoff: float = 0,
    resume: bool = False,
) -> dict[str, Any]:
    project_object = retry_call(
        lambda: api.project(project, entity), max_retries, retry_backoff
    )
    sweep_summaries = retry_call(
        lambda: list(project_object.sweeps()), max_retries, retry_backoff
    )
    current_sweep_ids = {str(attr(sweep, "id")) for sweep in sweep_summaries}
    run_ids_by_sweep: dict[str, list[str]] = {}
    for run in (run_payload or {}).get("runs", []):
        if run.get("sweep_id"):
            run_ids_by_sweep.setdefault(run["sweep_id"], []).append(run["id"])

    def load_sweep(summary: Any) -> dict[str, Any]:
        sweep_id = attr(summary, "id")
        detailed = retry_call(
            lambda: api.sweep(f"{entity}/{project}/{sweep_id}"),
            max_retries,
            retry_backoff,
        )
        record_errors: list[str] = []
        try:
            detailed_runs = retry_call(
                lambda: list(attr(detailed, "runs") or []),
                max_retries,
                retry_backoff,
            )
        except Exception as error:
            detailed_runs = []
            record_errors.append(f"runs: {error_text(error)}")
        return sweep_record(
            detailed,
            run_ids_by_sweep.get(sweep_id),
            detailed_runs,
            record_errors,
        )

    records = []
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("entity") != entity or previous.get("project") != project:
            raise ValueError(
                f"cannot resume {output_path}: entity/project does not match"
            )
        records_by_id = {
            str(sweep.get("id")): sweep
            for sweep in previous.get("sweeps", [])
            if sweep.get("id") is not None
            and str(sweep.get("id")) in current_sweep_ids
            and not sweep.get("record_errors")
        }
        records = list(records_by_id.values())
    completed_sweep_ids = {str(sweep["id"]) for sweep in records}
    pending_summaries = [
        sweep
        for sweep in sweep_summaries
        if str(attr(sweep, "id")) not in completed_sweep_ids
    ]
    if completed_sweep_ids:
        print(
            f"sweeps: resuming with {len(completed_sweep_ids)}/"
            f"{len(sweep_summaries)} complete",
            file=sys.stderr,
            flush=True,
        )

    def failed_sweep_record(summary: Any, error: BaseException) -> dict[str, Any]:
        sweep_id = attr(summary, "id")
        fallback_ids = sorted(run_ids_by_sweep.get(sweep_id, []))
        return {
            "id": sweep_id,
            "name": attr(summary, "name"),
            "state": attr(summary, "state"),
            "config": json_safe(attr(summary, "config") or {}),
            "run_ids": fallback_ids,
            "run_count": len(fallback_ids),
            "run_ids_source": "run.sweep.id fallback",
            "url": attr(summary, "url"),
            "record_errors": [f"record: {error_text(error)}"],
        }

    def accept_record(record: dict[str, Any], index: int) -> None:
        records.append(record)
        print(
            f"sweeps: {index}/{len(sweep_summaries)} "
            f"({record['id']} {record['run_count']} runs)",
            file=sys.stderr,
            flush=True,
        )

    if workers == 1:
        for offset, summary in enumerate(pending_summaries, start=1):
            index = len(completed_sweep_ids) + offset
            try:
                record = load_sweep(summary)
            except Exception as error:
                record = failed_sweep_record(summary, error)
            accept_record(record, index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_summary = {
                executor.submit(load_sweep, sweep): sweep
                for sweep in pending_summaries
            }
            for offset, future in enumerate(
                as_completed(future_to_summary), start=1
            ):
                index = len(completed_sweep_ids) + offset
                try:
                    record = future.result()
                except Exception as error:
                    record = failed_sweep_record(
                        future_to_summary[future], error
                    )
                accept_record(record, index)
    records.sort(key=lambda sweep: sweep.get("id") or "")
    error_count = sum(bool(sweep["record_errors"]) for sweep in records)
    payload = {
        "entity": entity,
        "project": project,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "complete": error_count == 0 and len(records) == len(sweep_summaries),
        "expected_sweep_count": len(sweep_summaries),
        "sweep_count": len(records),
        "error_count": error_count,
        "sweeps": records,
    }
    atomic_json_dump(output_path, payload)
    return payload


def history_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    step = row.get("_step")
    if isinstance(step, (int, float)) and not isinstance(step, bool):
        step_key: tuple[Any, ...] = (0, float(step))
    else:
        step_key = (1, str(step))
    return (*step_key, json.dumps(json_safe(row), sort_keys=True))


def iter_history_records(
    histories: Mapping[str, list[Mapping[str, Any]]]
) -> Iterable[dict[str, Any]]:
    for run_id in sorted(histories):
        for row in sorted(histories[run_id], key=history_row_sort_key):
            yield {"run_id": run_id, "history": row}


def history_manifest(
    entity: str,
    project: str,
    exported_at: str,
    expected_run_count: int,
    histories: Mapping[str, list[Mapping[str, Any]]],
    scanned_run_ids: Iterable[str],
    errors: Mapping[str, str],
    complete: bool,
) -> dict[str, Any]:
    scanned = sorted(scanned_run_ids)
    return {
        "entity": entity,
        "project": project,
        "exported_at": exported_at,
        "complete": complete,
        "expected_run_count": expected_run_count,
        "scanned_run_count": len(scanned),
        "scanned_run_ids": scanned,
        "runs_with_history_count": sum(bool(rows) for rows in histories.values()),
        "runs_with_history_ids": sorted(
            run_id for run_id, rows in histories.items() if rows
        ),
        "history_row_count": sum(len(rows) for rows in histories.values()),
        "error_count": len(errors),
        "errors": dict(sorted(errors.items())),
    }


def write_history_snapshot(
    output_path: Path,
    manifest_path: Path,
    entity: str,
    project: str,
    exported_at: str,
    expected_run_count: int,
    histories: Mapping[str, list[Mapping[str, Any]]],
    scanned_run_ids: Iterable[str],
    errors: Mapping[str, str],
    complete: bool,
) -> dict[str, Any]:
    atomic_jsonl_dump(output_path, iter_history_records(histories))
    manifest = history_manifest(
        entity,
        project,
        exported_at,
        expected_run_count,
        histories,
        scanned_run_ids,
        errors,
        complete,
    )
    atomic_json_dump(manifest_path, manifest)
    return manifest


def export_history(
    api: wandb.Api,
    entity: str,
    project: str,
    output_path: Path,
    manifest_path: Path,
    checkpoint_every: int,
    workers: int,
    max_retries: int,
    retry_backoff: float,
    resume: bool = False,
) -> dict[str, Any]:
    exported_at = datetime.now(timezone.utc).isoformat()
    run_objects = retry_call(
        lambda: list(api.runs(f"{entity}/{project}")),
        max_retries,
        retry_backoff,
    )
    histories: dict[str, list[Mapping[str, Any]]] = {}
    scanned_run_ids: set[str] = set()
    errors: dict[str, str] = {}
    current_run_ids = {str(attr(run, "id")) for run in run_objects}
    if resume and output_path.exists() and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if (
            previous_manifest.get("entity") != entity
            or previous_manifest.get("project") != project
        ):
            raise ValueError(
                f"cannot resume {manifest_path}: entity/project does not match"
            )
        exported_at = previous_manifest.get("exported_at") or exported_at
        previous_errors = set((previous_manifest.get("errors") or {}).keys())
        scanned_run_ids = {
            str(run_id)
            for run_id in previous_manifest.get("scanned_run_ids", [])
            if str(run_id) in current_run_ids and str(run_id) not in previous_errors
        }
        histories = {run_id: [] for run_id in scanned_run_ids}
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                run_id = str(record.get("run_id"))
                if run_id in scanned_run_ids:
                    histories[run_id].append(record.get("history") or {})
    pending_runs = [
        run for run in run_objects if str(attr(run, "id")) not in scanned_run_ids
    ]
    if scanned_run_ids:
        print(
            f"history: resuming with {len(scanned_run_ids)}/"
            f"{len(run_objects)} complete",
            file=sys.stderr,
            flush=True,
        )
    starting_history_count = len(scanned_run_ids)

    def load_history(run: Any) -> tuple[str, list[Mapping[str, Any]]]:
        run_id = str(attr(run, "id"))
        rows = retry_call(
            lambda: list(run.scan_history()), max_retries, retry_backoff
        )
        return run_id, [json_safe(dict(row)) for row in rows]

    def accept_history(
        run_id: str,
        rows: list[Mapping[str, Any]],
        index: int,
        error: Optional[BaseException] = None,
    ) -> None:
        histories[run_id] = rows
        if error is not None:
            errors[run_id] = error_text(error)
        scanned_run_ids.add(run_id)
        print(
            f"history: {index}/{len(run_objects)} "
            f"({run_id} {len(rows)} rows)",
            file=sys.stderr,
            flush=True,
        )
        if checkpoint_every > 0 and index % checkpoint_every == 0:
            write_history_snapshot(
                output_path,
                manifest_path,
                entity,
                project,
                exported_at,
                len(run_objects),
                histories,
                scanned_run_ids,
                errors,
                False,
            )

    if workers == 1:
        for offset, run in enumerate(pending_runs, start=1):
            index = starting_history_count + offset
            fallback_id = str(attr(run, "id"))
            try:
                run_id, rows = load_history(run)
                accept_history(run_id, rows, index)
            except Exception as error:
                accept_history(fallback_id, [], index, error)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_run = {
                executor.submit(load_history, run): run for run in pending_runs
            }
            for offset, future in enumerate(as_completed(future_to_run), start=1):
                index = starting_history_count + offset
                fallback_id = str(attr(future_to_run[future], "id"))
                try:
                    run_id, rows = future.result()
                    accept_history(run_id, rows, index)
                except Exception as error:
                    accept_history(fallback_id, [], index, error)

    complete = len(scanned_run_ids) == len(run_objects) and not errors
    return write_history_snapshot(
        output_path,
        manifest_path,
        entity,
        project,
        exported_at,
        len(run_objects),
        histories,
        scanned_run_ids,
        errors,
        complete,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--only", choices=("all", "runs", "sweeps", "history"), default="all"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete run, sweep, and history checkpoints",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum concurrent per-run artifact/summary requests",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Atomically checkpoint the run dump after this many runs (0 disables)",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Export every run's full scan_history() stream to JSONL",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries after the initial attempt for each W&B API operation",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Initial exponential retry delay in seconds",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff cannot be negative")
    api = wandb.Api(timeout=args.timeout)
    runs_path = args.output_dir / "wandb_runs.json"
    if args.only in {"all", "runs"}:
        runs = export_runs(
            api,
            args.entity,
            args.project,
            runs_path,
            args.checkpoint_every,
            args.workers,
            args.max_retries,
            args.retry_backoff,
            args.resume,
        )
    elif args.only == "sweeps":
        with runs_path.open(encoding="utf-8") as handle:
            runs = json.load(handle)
    else:
        runs = {"run_count": "not exported", "runs": []}
    if args.only in {"all", "sweeps"}:
        sweeps = export_sweeps(
            api,
            args.entity,
            args.project,
            args.output_dir / "wandb_sweeps.json",
            args.workers,
            runs,
            args.max_retries,
            args.retry_backoff,
            args.resume,
        )
    else:
        sweeps = {"sweep_count": "not exported"}
    if args.include_history or args.only == "history":
        history = export_history(
            api,
            args.entity,
            args.project,
            args.output_dir / "wandb_history.jsonl",
            args.output_dir / "wandb_history_manifest.json",
            args.checkpoint_every,
            args.workers,
            args.max_retries,
            args.retry_backoff,
            args.resume,
        )
    else:
        history = {"history_row_count": "not exported"}
    print(
        f"exported {runs['run_count']} runs, {sweeps['sweep_count']} sweeps, "
        f"and {history['history_row_count']} history rows",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
