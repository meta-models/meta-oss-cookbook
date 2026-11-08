"""CPU regressions for the standalone rollout plotting helper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("data_analyst_plot_results", SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot)


def _write_run(path: Path, num_rows: int = 4) -> Path:
    path.mkdir()
    log = path / "console.log"
    log.write_text(
        "Train | Step: 1 rollout_reward/_mean: 0.25 loss/mean: 1.2e-3\n"
        "Train | Step: 2 rollout_reward/_mean: 0.75 loss/mean: 2.0e-6\n"
    )
    rows = [
        {
            "group_id": i,
            "rollout_id": 0,
            "is_validation": False,
            "status": "completed",
            "reward": 1.0 if i >= num_rows // 2 else 0.05,
            "turns": [{"completion_message": {"tool_calls": []}}],
        }
        for i in range(num_rows)
    ]
    rows.append(
        {"group_id": 999, "is_validation": True, "status": "completed", "turns": []}
    )
    (path / "rollout_samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n{incomplete json\n"
    )
    return log


def _run(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path), *args],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


def test_parser_accepts_prefixed_colored_scientific_metrics(tmp_path: Path) -> None:
    log = tmp_path / "console.log"
    log.write_text(
        "[INFO] \x1b[31mTrain | Step: 2\x1b[0m "
        "rollout_reward/_mean: 7.5E-1 loss/mean: 2.0e-6\n"
    )
    assert plot.parse_steps(str(log)) == [
        (2, {"rollout_reward/_mean": 0.75, "loss/mean": 2e-6})
    ]


@pytest.mark.parametrize("prefix", ["run-", "offline-run-"])
def test_find_console_log_supports_both_wandb_layouts(
    tmp_path: Path, prefix: str
) -> None:
    log = tmp_path / "wandb" / f"{prefix}fixture" / "files" / "output.log"
    log.parent.mkdir(parents=True)
    log.write_text("metrics")
    assert plot.find_console_log(str(tmp_path)) == str(log)


def test_saved_console_stream_has_priority(tmp_path: Path) -> None:
    log = _write_run(tmp_path / "run")
    assert plot.find_console_log(str(log.parent)) == str(log)


def test_metric_counts_calls_not_turns() -> None:
    row = {"turns": [{"completion_message": {"tool_calls": [{}, {}]}}]}
    assert plot._num_tool_calls(row) == 2


def test_error_outcomes_are_not_silently_dropped() -> None:
    rows = [
        {"group_id": 0, "status": "completed"},
        {"group_id": 1, "status": "error_timeout"},
    ]
    bins = plot.outcome_bins(rows, num_bins=1)
    assert bins == [
        {
            "completed": 50.0,
            "env_budget": 0.0,
            "out_of_turns": 0.0,
            "out_of_tokens": 0.0,
            "error_or_other": 50.0,
        }
    ]
    assert sum(bins[0].values()) == 100.0


@pytest.mark.parametrize("num_rows", [0, 1])
def test_short_runs_fail_with_actionable_messages(
    tmp_path: Path, num_rows: int
) -> None:
    path = tmp_path / "run"
    _write_run(path, num_rows=num_rows)
    result = _run(path)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    expected = "no training rollouts" if num_rows == 0 else "at least 2"
    assert expected in result.stderr


def test_plotting_writes_pngs_without_claiming_a_stop_reason(tmp_path: Path) -> None:
    path = tmp_path / "run"
    _write_run(path)
    result = _run(path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reward per step: 0.25 0.75" in result.stdout
    assert "stopped by saturation" not in result.stdout
    assert "late recorded rollouts all correct: True" in result.stdout
    assert len(plot.load_train_rollouts(str(path))) == 4
    assert "rollout scope:" in result.stdout
    assert "median_requested_tool_calls" in result.stdout
    assert "env_budget" in result.stdout
    assert (path / "reward.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (path / "outcomes.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_replayed_ids_preserve_append_order_and_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "replay"
    _write_run(path)
    replay = {
        "group_id": 0,
        "rollout_id": 0,
        "status": "completed",
        "reward": 1.0,
        "turns": [],
        "is_validation": False,
    }
    with (path / "rollout_samples.jsonl").open("a") as file:
        file.write(json.dumps(replay) + "\n")
    rows = plot.load_train_rollouts(str(path))
    assert rows[-1] == replay
    result = _run(path)
    assert result.returncode == 0, result.stderr
    assert "1 repeated (group_id, rollout_id) occurrences" in result.stdout


def test_environment_budget_completion_has_its_own_outcome() -> None:
    row = {
        "group_id": 0,
        "status": "completed",
        "turns": [{"env_rewards": {"turn_budget_exhausted": 1.0}}],
    }
    bins = plot.outcome_bins([row], num_bins=1)
    assert bins[0]["completed"] == 0.0
    assert bins[0]["env_budget"] == 100.0
    assert sum(bins[0].values()) == 100.0
