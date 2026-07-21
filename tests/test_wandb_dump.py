from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import wandb_dump


class FakeArtifact:
    def __init__(self, name: str, artifact_type: str) -> None:
        self.id = f"id-{name}"
        self.name = name
        self.type = artifact_type
        self.version = "v0"
        self.digest = f"digest-{name}"
        self.state = "COMMITTED"
        self.size = 10
        self.aliases = ["latest"]
        self.created_at = "2026-01-01T00:00:00Z"
        self.updated_at = "2026-01-01T00:00:00Z"
        self.entity = "entity"
        self.project = "project"
        self.qualified_name = f"entity/project/{name}:v0"


class FakeRun:
    def __init__(
        self,
        run_id: str,
        *,
        history: list[dict[str, Any]] | None = None,
        history_error: Exception | None = None,
        logged_error: Exception | None = None,
    ) -> None:
        self.id = run_id
        self.name = f"run-{run_id}"
        self.config = {"dataset": "example"}
        self.summary = {"metric": 0.5}
        self.tags = ["test"]
        self.state = "finished"
        self.sweep = SimpleNamespace(id="sweep-1")
        self.created_at = "2026-01-01T00:00:00Z"
        self.url = f"https://wandb.invalid/runs/{run_id}"
        self._history = history or []
        self._history_error = history_error
        self._logged_error = logged_error
        self.logged_calls = 0
        self.history_calls = 0

    def logged_artifacts(self) -> list[FakeArtifact]:
        self.logged_calls += 1
        if self._logged_error is not None:
            raise self._logged_error
        return [FakeArtifact("code", "code")]

    def used_artifacts(self) -> list[FakeArtifact]:
        return [FakeArtifact("dataset", "dataset")]

    def scan_history(self) -> list[dict[str, Any]]:
        self.history_calls += 1
        if self._history_error is not None:
            raise self._history_error
        return self._history


class FakeProject:
    def __init__(self, sweeps: list[Any]) -> None:
        self._sweeps = sweeps

    def sweeps(self) -> list[Any]:
        return self._sweeps


class FakeApi:
    def __init__(self, runs: list[FakeRun], sweeps: list[Any] | None = None) -> None:
        self._runs = runs
        self._sweeps = sweeps or []

    def runs(self, path: str) -> list[FakeRun]:
        assert "/" in path
        return self._runs

    def project(self, project: str, entity: str) -> FakeProject:
        assert project and entity
        return FakeProject(self._sweeps)

    def sweep(self, path: str) -> Any:
        sweep_id = path.rsplit("/", 1)[-1]
        return next(sweep for sweep in self._sweeps if sweep.id == sweep_id)


def test_retry_call_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr(wandb_dump.time, "sleep", lambda _: None)
    assert wandb_dump.retry_call(flaky, max_retries=2, retry_backoff=1) == "ok"
    assert attempts == 3


def test_run_record_preserves_artifact_relationships() -> None:
    record = wandb_dump.run_record(FakeRun("abc"))

    assert record["record_errors"] == []
    assert record["artifact_errors"] == []
    assert {(item["relationship"], item["type"]) for item in record["artifacts"]} == {
        ("logged", "code"),
        ("used", "dataset"),
    }
    assert [item["name"] for item in record["dataset_artifacts"]] == ["dataset"]


def test_export_runs_marks_artifact_failure_incomplete(tmp_path: Path) -> None:
    api = FakeApi([FakeRun("bad", logged_error=RuntimeError("unavailable"))])
    output = tmp_path / "runs.json"

    payload = wandb_dump.export_runs(
        api,
        "entity",
        "project",
        output,
        checkpoint_every=0,
        workers=1,
        max_retries=1,
        retry_backoff=0,
    )

    assert payload["complete"] is False
    assert payload["error_count"] == 1
    assert payload["runs"][0]["artifact_errors"] == [
        "logged: RuntimeError: unavailable"
    ]
    assert json.loads(output.read_text())["complete"] is False


def test_sweep_export_uses_run_membership_fallback(tmp_path: Path) -> None:
    sweep = SimpleNamespace(
        id="sweep-1",
        name="example",
        state="FINISHED",
        config={"method": "grid"},
        runs=[],
        url="https://wandb.invalid/sweeps/sweep-1",
    )
    run_payload = {"runs": [{"id": "abc", "sweep_id": "sweep-1"}]}

    payload = wandb_dump.export_sweeps(
        FakeApi([], [sweep]),
        "entity",
        "project",
        tmp_path / "sweeps.json",
        workers=1,
        run_payload=run_payload,
    )

    assert payload["complete"] is True
    assert payload["sweeps"][0]["run_ids"] == ["abc"]
    assert payload["sweeps"][0]["run_ids_source"] == "run.sweep.id fallback"


def test_history_export_is_deterministic_and_auditable(tmp_path: Path) -> None:
    runs = [
        FakeRun("b", history=[{"_step": 1, "acc": 0.2}]),
        FakeRun(
            "a",
            history=[{"_step": 2, "acc": 0.3}, {"_step": 0, "acc": 0.1}],
        ),
    ]
    output = tmp_path / "history.jsonl"
    manifest_path = tmp_path / "history_manifest.json"

    manifest = wandb_dump.export_history(
        FakeApi(runs),
        "entity",
        "project",
        output,
        manifest_path,
        checkpoint_every=1,
        workers=2,
        max_retries=0,
        retry_backoff=0,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert [(record["run_id"], record["history"]["_step"]) for record in records] == [
        ("a", 0),
        ("a", 2),
        ("b", 1),
    ]
    assert manifest["complete"] is True
    assert manifest["scanned_run_count"] == 2
    assert manifest["history_row_count"] == 3
    assert json.loads(manifest_path.read_text()) == manifest


def test_history_export_records_exhausted_error(tmp_path: Path) -> None:
    run = FakeRun("bad", history_error=RuntimeError("history unavailable"))

    manifest = wandb_dump.export_history(
        FakeApi([run]),
        "entity",
        "project",
        tmp_path / "history.jsonl",
        tmp_path / "manifest.json",
        checkpoint_every=0,
        workers=1,
        max_retries=1,
        retry_backoff=0,
    )

    assert manifest["complete"] is False
    assert manifest["error_count"] == 1
    assert manifest["errors"] == {"bad": "RuntimeError: history unavailable"}


def test_run_export_resume_skips_completed_records(tmp_path: Path) -> None:
    first = FakeRun("first")
    second = FakeRun("second")
    output = tmp_path / "runs.json"
    common = {
        "checkpoint_every": 0,
        "workers": 1,
        "max_retries": 0,
        "retry_backoff": 0,
    }

    wandb_dump.export_runs(
        FakeApi([first]), "entity", "project", output, **common
    )
    payload = wandb_dump.export_runs(
        FakeApi([first, second]),
        "entity",
        "project",
        output,
        resume=True,
        **common,
    )

    assert first.logged_calls == 1
    assert second.logged_calls == 1
    assert payload["complete"] is True
    assert {run["id"] for run in payload["runs"]} == {"first", "second"}


def test_history_export_resume_preserves_completed_rows(tmp_path: Path) -> None:
    first = FakeRun("first", history=[{"_step": 0, "metric": 1}])
    second = FakeRun("second", history=[{"_step": 0, "metric": 2}])
    output = tmp_path / "history.jsonl"
    manifest_path = tmp_path / "manifest.json"
    common = {
        "checkpoint_every": 0,
        "workers": 1,
        "max_retries": 0,
        "retry_backoff": 0,
    }

    wandb_dump.export_history(
        FakeApi([first]),
        "entity",
        "project",
        output,
        manifest_path,
        **common,
    )
    manifest = wandb_dump.export_history(
        FakeApi([first, second]),
        "entity",
        "project",
        output,
        manifest_path,
        resume=True,
        **common,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert first.history_calls == 1
    assert second.history_calls == 1
    assert [record["run_id"] for record in records] == ["first", "second"]
    assert manifest["complete"] is True
