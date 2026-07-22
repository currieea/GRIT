from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as training_main  # noqa: E402
from scripts import reproduce_table1  # noqa: E402


def test_data_root_precedence(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"

    assert reproduce_table1.resolve_data_root(
        explicit, {reproduce_table1.DATA_ENV_VAR: str(environment)}
    ) == explicit.resolve()
    assert reproduce_table1.resolve_data_root(
        None, {reproduce_table1.DATA_ENV_VAR: str(environment)}
    ) == environment.resolve()
    assert reproduce_table1.resolve_data_root(None, {}) == (
        reproduce_table1.PROJECT_ROOT / "data"
    ).resolve()


def test_manifest_expands_expected_runs() -> None:
    manifest = reproduce_table1.load_manifest()
    full = reproduce_table1.expand_runs(manifest)
    quick = reproduce_table1.expand_runs(manifest, quick=True)

    assert len(reproduce_table1.runnable_rows(manifest)) == 13
    assert len(full) == 164
    assert len(quick) == 13
    assert {run["config"]["dataset"] for run in full} == {"LISAColoredMNIST"}
    assert {run["config"]["seed"] for run in full} == {1001}


def test_quick_mode_uses_audited_grit_configuration() -> None:
    manifest = reproduce_table1.load_manifest()
    runs = reproduce_table1.expand_runs(
        manifest,
        quick=True,
        row_ids=["colored_grit_clean_probing"],
    )

    assert len(runs) == 1
    assert runs[0]["config"]["solver"] == "ECMP"
    assert runs[0]["config"]["projection"] == "oracle"
    assert runs[0]["config"]["param1"] == 8
    assert runs[0]["config"]["param2"] == 1024


def test_dataset_fingerprint_changes_run_identity() -> None:
    manifest = reproduce_table1.load_manifest()
    first = reproduce_table1.expand_runs(
        manifest, quick=True, row_ids=["colored_erm"], dataset_fingerprint="first"
    )
    second = reproduce_table1.expand_runs(
        manifest, quick=True, row_ids=["colored_erm"], dataset_fingerprint="second"
    )

    assert first[0]["run_id"] != second[0]["run_id"]
    assert first[0]["config_id"] != second[0]["config_id"]


def test_dataset_fingerprint_ignores_volatile_metadata() -> None:
    base = {
        "dataset": "LISAColoredMNIST",
        "seed": 1001,
        "clip_model": "ViT-B/32",
        "clip_commit": "commit",
        "files": {"x_array.pth": {"sha256": "abc", "bytes": 10}},
        "created_at": "first",
        "device": "cpu",
    }
    changed = dict(base, created_at="second", device="cuda")

    assert reproduce_table1.dataset_fingerprint(base) == reproduce_table1.dataset_fingerprint(
        changed
    )


def test_dry_run_does_not_require_prepared_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = reproduce_table1.main(
        [
            "run",
            "--dry-run",
            "--quick",
            "--rows",
            "colored_erm",
            "--data-root",
            str(tmp_path / "missing"),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Planned runs: 1" in output
    assert "--dataset LISAColoredMNIST" in output
    assert "--no_wandb" in output


def _write_completed_run(
    output_root: Path,
    run: dict,
    *,
    in_accuracy: float,
    test_accuracy: float,
) -> None:
    run_dir = output_root / "runs" / run["run_id"]
    run_dir.mkdir(parents=True)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "selection": {
                    "in_domain": {
                        "step": 3,
                        "metrics": {
                            "in_test": {"acc_avg": in_accuracy},
                            "test": {"acc_avg": test_accuracy},
                        },
                    }
                }
            }
        )
    )
    record = dict(run)
    record.update({"status": "completed", "result_path": str(result_path)})
    (run_dir / "record.json").write_text(json.dumps(record))


def test_summary_selects_by_in_domain_not_ood(tmp_path: Path) -> None:
    manifest = reproduce_table1.load_manifest()
    runs = reproduce_table1.expand_runs(
        manifest, row_ids=["colored_irm"], seeds=[1001]
    )
    _write_completed_run(tmp_path, runs[0], in_accuracy=0.80, test_accuracy=0.90)
    _write_completed_run(tmp_path, runs[1], in_accuracy=0.81, test_accuracy=0.10)

    rows = reproduce_table1.summarize_results(manifest, tmp_path)
    irm = next(row for row in rows if row["id"] == "colored_irm")

    assert irm["reproduced_in"] == pytest.approx(0.81)
    assert irm["reproduced_test"] == pytest.approx(0.10)
    assert irm["selected_config"]["param1"] == 0.1
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "table1_cmnist.csv").is_file()
    assert (tmp_path / "table1_cmnist.md").is_file()


def test_summary_ties_follow_manifest_order(tmp_path: Path) -> None:
    manifest = reproduce_table1.load_manifest()
    runs = reproduce_table1.expand_runs(manifest, row_ids=["colored_groupdro"])
    _write_completed_run(tmp_path, runs[0], in_accuracy=0.80, test_accuracy=0.10)
    _write_completed_run(tmp_path, runs[1], in_accuracy=0.80, test_accuracy=0.90)

    rows = reproduce_table1.summarize_results(manifest, tmp_path)
    groupdro = next(row for row in rows if row["id"] == "colored_groupdro")

    assert groupdro["selected_config"]["param1"] == 0.001
    assert groupdro["reproduced_test"] == pytest.approx(0.10)


def test_summary_uses_only_current_run_plan(tmp_path: Path) -> None:
    manifest = reproduce_table1.load_manifest()
    runs = reproduce_table1.expand_runs(manifest, row_ids=["colored_groupdro"])
    _write_completed_run(tmp_path, runs[0], in_accuracy=0.70, test_accuracy=0.20)
    _write_completed_run(tmp_path, runs[1], in_accuracy=0.99, test_accuracy=0.90)
    (tmp_path / "run_plan.json").write_text(
        json.dumps({"run_ids": [runs[0]["run_id"]]})
    )

    rows = reproduce_table1.summarize_results(manifest, tmp_path)
    groupdro = next(row for row in rows if row["id"] == "colored_groupdro")

    assert groupdro["reproduced_in"] == pytest.approx(0.70)
    assert groupdro["selected_config"]["param1"] == 0.001


def test_main_parser_accepts_dataset_path(tmp_path: Path) -> None:
    args = training_main.build_parser().parse_args(["--root_dir", str(tmp_path)])
    assert args.root_dir == tmp_path


def test_structured_result_includes_selected_steps() -> None:
    args = training_main.build_parser().parse_args(["--no_wandb"])
    solver = SimpleNamespace(
        device="cpu",
        dataset=SimpleNamespace(data_dir=Path("/does/not/exist")),
        best_id_step=4,
        best_val_step=3,
        best_oracle_step=7,
        best_id_log={"in_test": {"acc_avg": 0.8}, "test": {"acc_avg": 0.2}},
        best_val_log={"val": {"acc_avg": 0.7}},
        best_oracle_log={"test": {"acc_avg": 0.3}},
    )

    result = training_main.build_result(args, solver, 1.5)

    assert result["selection"]["in_domain"]["step"] == 4
    assert result["selection"]["in_domain"]["metrics"]["test"]["acc_avg"] == 0.2
    assert result["elapsed_seconds"] == 1.5


def test_incomplete_prepared_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        reproduce_table1.validate_prepared_dataset(tmp_path)


def test_prepared_dataset_hashes_are_verified(tmp_path: Path) -> None:
    for name in reproduce_table1.REQUIRED_DATA_FILES:
        if name != "reproduction_manifest.json":
            (tmp_path / name).write_bytes(name.encode())
    files = {
        name: {"sha256": reproduce_table1.sha256_file(tmp_path / name)}
        for name in reproduce_table1.REQUIRED_DATA_FILES
        if name != "reproduction_manifest.json"
    }
    (tmp_path / "reproduction_manifest.json").write_text(json.dumps({"files": files}))

    reproduce_table1.validate_prepared_dataset(tmp_path)
    (tmp_path / "x_array.pth").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="hash mismatch"):
        reproduce_table1.validate_prepared_dataset(tmp_path)


def test_seed_everything_is_repeatable() -> None:
    import torch

    reproduce_table1.seed_everything(1001)
    first = torch.rand(4)
    reproduce_table1.seed_everything(1001)
    second = torch.rand(4)

    assert torch.equal(first, second)
