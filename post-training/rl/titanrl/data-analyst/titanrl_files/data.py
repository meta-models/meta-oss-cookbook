# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic, fully offline csv/log data-analysis tasks with a verifiable golden answer.

Each sample gives the model a small data file (a csv table or a server log) already
present in its workspace, plus a question whose answer is a single number. The model
must compute the answer and write it to a required output file; ``rubric.py`` grades
the written file against the golden value computed here at generation time -- no
external dataset or network access is needed, unlike ``search_r1``.
"""

from __future__ import annotations

import csv
import io
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from torchtitan.config import Configurable

OUTPUT_FILENAME = "answer.json"
"""Filename every task asks the model to write its answer to, as ``{"answer": <number>}``."""

_WRITE_PROMPTLY_HINT = (
    "\n\nWork efficiently: you have a limited number of tool calls. As soon as you "
    "have computed the number, write it to the output file in that same command -- "
    "do not explore further first."
)
"""Appended to every task prompt. Observed rollouts frequently computed the correct
number and then ran out of turn/token budget before writing the artifact, scoring
zero; this tells the model to persist the result as soon as it has it."""

_CSV_REGIONS = ["north", "south", "east", "west", "central"]

_LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
_LOG_STATUSES = [200, 200, 200, 301, 404, 500, 503]
_LOG_HOSTS = [f"10.0.{n}.{m}" for n in range(1, 5) for m in (1, 2, 3)]


@dataclass(frozen=True, kw_only=True, slots=True)
class GlimmerDataAnalystSample:
    """One templated csv/log task: the workspace file to write, the question, and
    the golden numeric answer to grade against."""

    task_id: str
    """Unique id for this instance (``<template>_<index>``); used for logging and
    for the train/validation disjointness check, not for grading."""

    prompt: str
    """The user-facing task description, naming the input and required output files."""

    input_filename: str
    """Filename the input data is written under in the rollout's workspace."""

    input_content: str
    """Full text content of the input data file."""

    golden_value: float
    """The correct numeric answer, computed directly from ``input_content``."""

    tolerance: float = 1e-6
    """Absolute tolerance for a numeric match against ``golden_value``."""


def _make_csv_sum_task(rng: random.Random, index: int) -> GlimmerDataAnalystSample:
    """A city metrics table; question = sum of one numeric column, filtered to one region."""
    num_rows = rng.randint(6, 14)
    filter_region = rng.choice(_CSV_REGIONS)
    rows = []
    total = 0.0
    for i in range(num_rows):
        region = rng.choice(_CSV_REGIONS)
        population = rng.randint(10_000, 900_000)
        gdp_millions = round(rng.uniform(50.0, 5000.0), 2)
        rows.append((f"city_{i:03d}", region, population, gdp_millions))
        if region == filter_region:
            total += gdp_millions

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["city", "region", "population", "gdp_millions"])
    writer.writerows(rows)

    prompt = (
        "The file `data.csv` (already in your workspace) lists cities with their "
        "region, population, and GDP in millions of dollars.\n\n"
        f"Compute the total `gdp_millions` for every city in the `{filter_region}` "
        f"region. Write the result to `{OUTPUT_FILENAME}` as JSON: "
        '{"answer": <number>}. Round to 2 decimal places.' + _WRITE_PROMPTLY_HINT
    )
    return GlimmerDataAnalystSample(
        task_id=f"csv_sum_{index}",
        prompt=prompt,
        input_filename="data.csv",
        input_content=buf.getvalue(),
        golden_value=round(total, 2),
        tolerance=0.02,
    )


def _make_csv_groupby_max_task(rng: random.Random, index: int) -> GlimmerDataAnalystSample:
    """A pension table; question = the highest average value across regions."""
    num_rows = rng.randint(8, 16)
    sums: dict[str, float] = {r: 0.0 for r in _CSV_REGIONS}
    counts: dict[str, int] = {r: 0 for r in _CSV_REGIONS}
    rows = []
    for i in range(num_rows):
        region = rng.choice(_CSV_REGIONS)
        amount = round(rng.uniform(500.0, 4000.0), 2)
        rows.append((f"account_{i:04d}", region, amount))
        sums[region] += amount
        counts[region] += 1

    averages = [sums[r] / counts[r] for r in _CSV_REGIONS if counts[r] > 0]
    best_avg = max(averages)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["account_id", "region", "monthly_pension"])
    writer.writerows(rows)

    prompt = (
        "The file `data.csv` (already in your workspace) lists pension accounts "
        "with their region and monthly pension amount.\n\n"
        "Find the region with the highest AVERAGE `monthly_pension`, and write that "
        f'average to `{OUTPUT_FILENAME}` as JSON: {{"answer": <number>}}. '
        "Round to 2 decimal places." + _WRITE_PROMPTLY_HINT
    )
    return GlimmerDataAnalystSample(
        task_id=f"csv_groupby_max_{index}",
        prompt=prompt,
        input_filename="data.csv",
        input_content=buf.getvalue(),
        golden_value=round(best_avg, 2),
        tolerance=0.02,
    )


def _make_log_count_task(rng: random.Random, index: int) -> GlimmerDataAnalystSample:
    """A synthetic server log; question = how many lines are at a given level."""
    num_lines = rng.randint(25, 60)
    target_level = rng.choice(_LOG_LEVELS)
    lines = []
    count = 0
    for i in range(num_lines):
        level = rng.choice(_LOG_LEVELS)
        host = rng.choice(_LOG_HOSTS)
        lines.append(
            f"2026-08-{(i % 28) + 1:02d}T00:{i % 60:02d}:00Z {level} {host} request handled"
        )
        if level == target_level:
            count += 1

    prompt = (
        "The file `access.log` (already in your workspace) has one log line per "
        "request, each starting with a timestamp and a level "
        f"({'/'.join(_LOG_LEVELS)}).\n\n"
        f"Count how many lines have level `{target_level}`. Write the count to "
        f'`{OUTPUT_FILENAME}` as JSON: {{"answer": <number>}}.' + _WRITE_PROMPTLY_HINT
    )
    return GlimmerDataAnalystSample(
        task_id=f"log_count_{index}",
        prompt=prompt,
        input_filename="access.log",
        input_content="\n".join(lines) + "\n",
        golden_value=float(count),
        tolerance=0.5,
    )


def _make_log_top_status_task(rng: random.Random, index: int) -> GlimmerDataAnalystSample:
    """A synthetic access log; question = how many lines have the most common HTTP
    status code (the count, not the code itself, so the answer stays a single number
    even when two codes tie for most common)."""
    num_lines = rng.randint(25, 60)
    counts: dict[int, int] = {}
    lines = []
    for i in range(num_lines):
        status = rng.choice(_LOG_STATUSES)
        host = rng.choice(_LOG_HOSTS)
        counts[status] = counts.get(status, 0) + 1
        lines.append(
            f'{host} - - [{(i % 28) + 1:02d}/Aug/2026] "GET /r/{i} HTTP/1.1" '
            f"{status} {rng.randint(100, 9000)}"
        )

    top_count = max(counts.values())

    prompt = (
        "The file `access.log` (already in your workspace) is an Apache-style "
        "access log; each line ends with an HTTP status code and a response size.\n\n"
        "Find the most frequent HTTP status code, and write the NUMBER OF LINES that "
        f'have that status code to `{OUTPUT_FILENAME}` as JSON: {{"answer": <number>}}.'
        + _WRITE_PROMPTLY_HINT
    )
    return GlimmerDataAnalystSample(
        task_id=f"log_top_status_{index}",
        prompt=prompt,
        input_filename="access.log",
        input_content="\n".join(lines) + "\n",
        golden_value=float(top_count),
        tolerance=0.5,
    )


_TEMPLATES = [
    _make_csv_sum_task,
    _make_csv_groupby_max_task,
    _make_log_count_task,
    _make_log_top_status_task,
]


class GlimmerDataAnalystDataset(Configurable):
    """Endless, seeded stream of synthetic csv/log data-analysis tasks.

    Fully offline and self-contained: every sample's data file and golden answer are
    generated in-process from ``seed`` -- no HF dataset download, no network access.
    ``split`` selects a disjoint index range, so train and validation never draw the
    same instance (see ``__next__``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42
        """Seed for both the per-template RNG and the round-robin template order."""

        split: Literal["train", "validation"] = "train"
        """``"train"`` uses indices below ``validation_offset``; ``"validation"`` uses
        indices at or above it, so the two splits never share an instance."""

        validation_offset: int = 1_000_000
        """First index reserved for ``"validation"``; large enough that the two
        ranges never collide for any realistic run length."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._template_order = list(range(len(_TEMPLATES)))
        random.Random(config.seed).shuffle(self._template_order)
        self._cursor = 0
        self._next_index = 0 if config.split == "train" else config.validation_offset

    def __iter__(self) -> Iterator[GlimmerDataAnalystSample]:
        return self

    def __next__(self) -> GlimmerDataAnalystSample:
        template_idx = self._template_order[self._cursor % len(self._template_order)]
        self._cursor += 1
        index = self._next_index
        self._next_index += 1
        # Each instance gets its own RNG, seeded from the dataset seed + its global
        # index (as a string -- random.Random doesn't accept a tuple seed), so any
        # index is independently reproducible regardless of draw order.
        instance_rng = random.Random(f"{self._config.seed}:{index}")
        return _TEMPLATES[template_idx](instance_rng, index)

    def state_dict(self) -> dict:
        return {"cursor": self._cursor, "next_index": self._next_index}

    def load_state_dict(self, state_dict: dict) -> None:
        self._cursor = state_dict["cursor"]
        self._next_index = state_dict["next_index"]
