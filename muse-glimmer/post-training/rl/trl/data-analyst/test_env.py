# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the tasks, the bash tool and the reward. Run: pytest test_env.py"""

import csv
import inspect
import io
import json
import os
import re
import resource
import signal
import time
from collections import Counter
from pathlib import Path

import pytest
from transformers.utils import get_json_schema

from env import DataAnalystEnv, accuracy, answered, make_dataset, make_task


def _recompute_golden(task: dict) -> float:
    """Independently recompute the answer by parsing the file the model sees."""
    question, content = task["prompt"][0]["content"], task["input_content"]
    if task["task_type"] == "category_revenue":
        category = re.search(
            r"revenue \(quantity x unit_price\) of the `(\w+)`", question
        ).group(1)
        rows = csv.DictReader(io.StringIO(content))
        return round(
            sum(
                int(r["quantity"]) * float(r["unit_price"])
                for r in rows
                if r["category"] == category
            ),
            2,
        )
    line_re = re.compile(
        r'^(\S+) \S+ \S+ \[(\d\d/\w+/\d{4}:\d\d:\d\d):\d\d [^\]]*\] "[^"]*" (\d{3}) '
    )
    parsed = [line_re.match(line).groups() for line in content.splitlines()]
    if task["task_type"] == "peak_error_minute":
        return float(
            max(
                Counter(
                    minute for _, minute, status in parsed if status.startswith("5")
                ).values()
            )
        )
    return float(
        max(Counter(ip for ip, _, status in parsed if status == "404").values())
    )


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_golden_answers_match_an_independent_computation(split):
    for i in range(200):
        task = make_task(split, i)
        assert task["golden_value"] == pytest.approx(
            _recompute_golden(task), abs=1e-9
        ), task["task_id"]


def test_tasks_are_deterministic_and_splits_disjoint():
    assert make_task("train", 7) == make_task("train", 7)
    ids = {
        split: set(make_dataset(split, 300)["task_id"])
        for split in ("train", "validation", "test")
    }
    assert not ids["train"] & ids["validation"] and not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    counts = Counter(make_dataset("test", 256)["task_type"]).values()
    assert len(counts) == 3 and max(counts) - min(counts) <= 1


def test_only_bash_is_exposed_as_a_tool():
    public = [
        n
        for n, _ in inspect.getmembers(DataAnalystEnv, inspect.isfunction)
        if not n.startswith("_")
    ]
    assert sorted(public) == [
        "bash",
        "get_reward",
        "reset",
    ]  # TRL skips reset and get_reward
    schema = get_json_schema(DataAnalystEnv().bash)["function"]
    assert schema["name"] == "bash" and schema["parameters"]["required"] == ["command"]


def _env(task_index: int = 0) -> tuple[DataAnalystEnv, dict]:
    task = make_task("test", task_index)
    env = DataAnalystEnv()
    env.reset(**task)
    return env, task


def test_reward_tiers():
    env, task = _env()
    assert env.get_reward() == 0.0  # no progress
    env.bash("ls")
    assert env.get_reward() == pytest.approx(0.05)  # ran a command, no file
    env.bash("echo 'not json' > answer.json")
    assert env.get_reward() == pytest.approx(0.20)  # malformed file
    env.bash(f"echo '{{\"answer\": {task['golden_value'] + 100}}}' > answer.json")
    assert env.get_reward() == pytest.approx(0.45)  # well-formed, wrong number
    assert accuracy([env]) == [0.0] and answered([env]) == [1.0]
    env.bash(f"echo '{json.dumps({'answer': task['golden_value']})}' > answer.json")
    assert env.get_reward() == 1.0 and accuracy([env]) == [1.0]


def test_repeated_failures_are_penalized_but_never_below_zero_or_on_correct():
    env, task = _env(1)
    for _ in range(3):
        env.bash("false")
    assert env.get_reward() == 0.0
    env.bash(f"echo '{json.dumps({'answer': task['golden_value']})}' > answer.json")
    assert env.get_reward() == 1.0


def test_workspace_is_private_and_reset_starts_fresh(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "do-not-leak")
    env, task = _env()
    output = env.bash("env; pwd; ls")
    assert "do-not-leak" not in output
    assert task["input_filename"] in output and "HOME=" + str(env._workdir) in output
    first = env._workdir
    env.bash("echo '{\"answer\": 1}' > answer.json")
    env.reset(**make_task("test", 2))
    assert not first.exists() and "answer.json" not in env.bash("ls")


def test_failures_timeouts_and_long_output_are_reported(monkeypatch):
    env, _ = _env()
    assert "[exit code: 3]" in env.bash("echo oops >&2; exit 3")
    assert len(env.bash("seq 1 100000")) < 1000
    monkeypatch.setattr("env.COMMAND_TIMEOUT_S", 1)
    assert "timed out" in env.bash("sleep 30")


def test_endless_output_is_bounded_in_time_and_memory(monkeypatch):
    monkeypatch.setattr("env.COMMAND_TIMEOUT_S", 1)
    env, _ = _env()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.monotonic()
    output = env.bash("yes")
    assert time.monotonic() - start < 5
    assert "timed out" in output and "truncated" in output and len(output) < 1000
    growth_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_before
    assert growth_kib < 64 * 1024  # without the cap, one second of `yes` buffers GiBs


def test_background_jobs_do_not_outlive_the_call():
    env, _ = _env()
    pid = int(env.bash("sleep 300 >/dev/null 2>&1 & echo $!").split()[0])
    for _ in range(50):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("background job survived the tool call")


def test_file_writes_are_capped(monkeypatch):
    monkeypatch.setattr("env.MAX_FILE_BYTES", 64 * 1024)
    env, _ = _env()
    env.bash("head -c 1000000 /dev/zero > big.bin")
    assert (env._workdir / "big.bin").stat().st_size <= 64 * 1024
