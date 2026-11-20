# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data-analyst tasks and a one-tool bash environment for TRL's GRPOTrainer.

This is the TitanRL data-analyst environment (one ``bash`` tool, a private
workspace, an ``answer.json`` artifact and the same tiered reward) ported to
TRL's ``environment_factory`` interface, with harder task files. Each task asks
for one number computed from a realistic file whose format trips naive shell
parsing: a CSV with quoted commas, Apache timestamps that must be bucketed by
minute, and a combined-format access log whose request lines can contain
spaces. The questions themselves are unambiguous, and the golden answer is
computed from the generated records, so grading is deterministic and offline.

Commands run in a private temporary workspace with a stripped environment, a
wall-clock timeout and output limits. This is not container isolation: train on
infrastructure without secrets or unrelated writable data.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import Dataset

OUTPUT_FILENAME = "answer.json"

# Disjoint index ranges keep train, validation and test tasks apart.
SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}

# Reward tiers (identical to the TitanRL recipe). A correct answer scores 1.0.
# Otherwise partial credits accumulate, so a well-formed wrong answer scores
# 0.45, and each repeat of an already-failed command costs 0.05.
RAN_OK_CREDIT = 0.05
FILE_EXISTS_CREDIT = 0.15
FILE_PARSES_CREDIT = 0.25
REPEATED_FAILURE_PENALTY = 0.05

COMMAND_TIMEOUT_S = 10
MAX_OUTPUT_CHARS = 800
MAX_FILE_BYTES = 16 * 1024 * 1024

_WRITE_PROMPTLY_HINT = (
    "\n\nWork efficiently: you have a limited number of tool calls. As soon as you "
    "have computed the number, write it to the output file in that same command -- "
    "do not explore further first."
)
_PRODUCTS = {
    "office": ["Desk lamp, brass", "Stapler", "Notebook, A5 (pack of 3)", '12" ruler', 'Pen set, "fine tip"', "Monitor stand"],
    "kitchen": ["Kettle", "Knife set, 5 pieces", "Cutting board, bamboo", 'Mug "World\'s best"', "Toaster"],
    "garden": ["Hose, 15 m", "Trowel", "Seed mix, wildflower", "Gloves", 'Planter, 30" round'],
    "toys": ["Puzzle, 1000 pieces", "Yo-yo", "Kite", 'Board game "Castles"', "Blocks, wooden"],
}
_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/131.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
    "curl/8.5.0",
    "python-requests/2.32",
]


def _clock(seconds: int) -> str:
    """Seconds after 09:00:00 as HH:MM:SS."""
    return f"{9 + seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _category_revenue(rng: random.Random) -> dict:
    """CSV sum with a filter. Product names contain quoted commas, so splitting on ',' misaligns columns."""
    category = rng.choice(list(_PRODUCTS))
    rows, total = [], 0.0
    for i in range(rng.randint(30, 60)):
        row_category = rng.choice(list(_PRODUCTS))
        quantity, price = rng.randint(1, 12), round(rng.uniform(2, 150), 2)
        rows.append([5000 + i, rng.choice(_PRODUCTS[row_category]), row_category, quantity, f"{price:.2f}"])
        if row_category == category:
            total += quantity * price
    buf = io.StringIO()
    csv.writer(buf).writerows([["order_id", "product", "category", "quantity", "unit_price"], *rows])
    prompt = (
        "The file `orders.csv` (already in your workspace) lists order lines with columns `order_id`, "
        "`product`, `category`, `quantity` and `unit_price`.\n\n"
        f"Compute the total revenue (quantity x unit_price) of the `{category}` category. Write it to "
        f'`{OUTPUT_FILENAME}` as JSON: {{"answer": <number>}}. Round to 2 decimal places.'
    )
    return dict(prompt=prompt, input_filename="orders.csv", input_content=buf.getvalue(),
                golden_value=round(total, 2), tolerance=0.02)


def _peak_error_minute(rng: random.Random) -> dict:
    """Per-minute count of 5xx responses. Apache timestamps must be cut at exactly the minute."""
    lines, per_minute = [], {}
    base_rate = rng.uniform(0.03, 0.10)
    t = rng.randint(1, 600)
    burst_minute, burst_rate = t // 60 + rng.randint(1, 8), rng.uniform(0.3, 0.8)
    for _ in range(rng.randint(150, 300)):
        t += rng.randint(0, 8)
        error_rate = burst_rate if t // 60 == burst_minute else base_rate
        if rng.random() < error_rate:
            status = rng.choice([500, 502, 503, 504])
            per_minute[t // 60] = per_minute.get(t // 60, 0) + 1
        else:
            status = rng.choices([200, 201, 304, 404], [70, 5, 15, 10])[0]
        lines.append(f'10.0.{rng.randint(1, 4)}.{rng.randint(1, 9)} - - [03/Aug/2026:{_clock(t)} +0000] '
                     f'"GET /api/x/{rng.randint(1, 999)} HTTP/1.1" {status} {rng.randint(100, 9000)}')
    prompt = (
        "The file `access.log` (already in your workspace) is a web server access log.\n\n"
        "Group requests by calendar minute (for example everything from 10:15:00 to 10:15:59) and find the minute "
        "with the most server errors (status 500-599). Write that minute's number of server errors to "
        f'`{OUTPUT_FILENAME}` as JSON: {{"answer": <number>}}.'
    )
    return dict(prompt=prompt, input_filename="access.log", input_content="\n".join(lines) + "\n",
                golden_value=float(max(per_minute.values())), tolerance=0.5)


def _top_not_found_client(rng: random.Random) -> dict:
    """Most 404s per client IP. Some request lines contain spaces, which shifts whitespace-split fields."""
    ips = [f"10.0.{rng.randint(1, 4)}.{rng.randint(1, 40)}" for _ in range(rng.randint(6, 10))]
    counts, lines = {}, []
    t = rng.randint(1, 600)
    for _ in range(rng.randint(120, 220)):
        t += rng.randint(0, 30)
        ip = rng.choice(ips)
        if rng.random() < 0.1:
            path = rng.choice(["/search?q=red shoes", "/files/annual report.pdf", "/p/summer sale"])
        else:
            path = rng.choice(["/", "/cart", "/login", f"/p/{rng.randint(1, 300)}", "/static/app.js"])
        status = rng.choices([200, 304, 404, 500], [70, 10, 17, 3])[0]
        counts[ip] = counts.get(ip, 0) + (status == 404)
        lines.append(f'{ip} - - [03/Aug/2026:{_clock(t)} +0000] "GET {path} HTTP/1.1" {status} {rng.randint(100, 9000)} '
                     f'"https://shop.example.com/" "{rng.choice(_AGENTS)}"')
    prompt = (
        "The file `access.log` (already in your workspace) is a web server access log in combined log format "
        "(client IP, timestamp, request line, status, size, referer and user agent).\n\n"
        "Which client IP received the most 404 responses? Write that IP's number of 404 responses to "
        f'`{OUTPUT_FILENAME}` as JSON: {{"answer": <number>}}.'
    )
    return dict(prompt=prompt, input_filename="access.log", input_content="\n".join(lines) + "\n",
                golden_value=float(max(counts.values())), tolerance=0.5)


TEMPLATES = [_category_revenue, _peak_error_minute, _top_not_found_client]


def make_task(split: str, position: int, seed: int = 42) -> dict:
    """The ``position``-th task of a split, reproducible on its own."""
    template = TEMPLATES[position % len(TEMPLATES)]
    index = SPLIT_OFFSETS[split] + position
    task = template(random.Random(f"{seed}:{index}"))
    task["prompt"] = [{"role": "user", "content": task["prompt"] + _WRITE_PROMPTLY_HINT}]
    task["task_type"] = template.__name__.lstrip("_")
    task["task_id"] = f"{task['task_type']}_{index}"
    return task


def make_dataset(split: str, size: int, seed: int = 42) -> Dataset:
    return Dataset.from_list([make_task(split, i, seed) for i in range(size)])


_SANDBOX_ENV = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/local/bin", "LANG": "C.UTF-8"}


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated {len(text) - MAX_OUTPUT_CHARS} chars]"


class DataAnalystEnv:
    """One rollout's private workspace. TRL exposes every public method except
    ``reset`` and ``get_reward`` as a tool, so ``bash`` is the only tool."""

    def __init__(self) -> None:
        self._workdir: Path | None = None

    def reset(self, input_filename: str, input_content: str, golden_value: float, tolerance: float, **_) -> None:
        self._remove_workdir()
        self._workdir = Path(tempfile.mkdtemp(prefix="glimmer_data_analyst_"))
        (self._workdir / input_filename).write_text(input_content)
        self._golden_value = golden_value
        self._tolerance = tolerance
        self._ran_ok = False
        self._failed_commands: set[str] = set()
        self._repeated_failures = 0

    def bash(self, command: str) -> str:
        """Run a shell command in your task workspace and return its stdout/stderr.

        Your input data file is already there; write your output file there too.

        Args:
            command: The shell command to run (via bash -c).

        Returns:
            The command's output, followed by its exit code if it failed.
        """
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=self._workdir,
            env={**_SANDBOX_ENV, "HOME": str(self._workdir), "TMPDIR": str(self._workdir)},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_limit_child,
        )
        annotation = ""
        try:
            stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            annotation = f"[error] command timed out after {COMMAND_TIMEOUT_S}s"
        failed = bool(annotation) or process.returncode != 0
        if failed:
            self._repeated_failures += command in self._failed_commands
            self._failed_commands.add(command)
        else:
            self._ran_ok = True
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        err = f"[stderr]\n{err}" if err else ""
        parts = [_truncate("\n".join(s for s in ((err, out) if failed else (out, err)) if s)), annotation]
        if process.returncode and not annotation:
            parts.append(f"[exit code: {process.returncode}]")
        return "\n".join(p for p in parts if p) or "[no output]"

    def get_reward(self) -> float:
        status = self._answer_status()
        if status == "correct":
            return 1.0
        reward = RAN_OK_CREDIT * self._ran_ok
        reward += FILE_EXISTS_CREDIT * (status != "missing")
        reward += FILE_PARSES_CREDIT * (status == "wrong")
        return max(0.0, reward - REPEATED_FAILURE_PENALTY * self._repeated_failures)

    def _answer_status(self) -> str:
        """``correct``, ``wrong`` (well-formed, wrong number), ``malformed`` or ``missing``."""
        path = self._workdir / OUTPUT_FILENAME
        if not path.is_file():
            return "missing"
        try:
            value = float(json.loads(path.read_text())["answer"])
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return "malformed"
        return "correct" if abs(value - self._golden_value) <= self._tolerance else "wrong"

    def _remove_workdir(self) -> None:
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)

    def __del__(self) -> None:
        self._remove_workdir()


def accuracy(environments: list[DataAnalystEnv], **kwargs) -> list[float]:
    """1.0 when ``answer.json`` holds the right number. Logged with weight 0."""
    return [float(env._answer_status() == "correct") for env in environments]


def answered(environments: list[DataAnalystEnv], **kwargs) -> list[float]:
    """1.0 when ``answer.json`` is well-formed, right or wrong. Logged with weight 0."""
    return [float(env._answer_status() in ("correct", "wrong")) for env in environments]
