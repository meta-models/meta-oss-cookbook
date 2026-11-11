"""Plot training rewards and recorded rollout outcomes from a TitanRL run.

    python plot_results.py outputs/rl/glimmer_data_analyst [--log FILE] [--out DIR]

Reads console.log (when saved with tee), online/offline W&B output.log, and
rollout_samples.jsonl. The recipe records every sibling; other configurations
may record a biased subset, so these statistics always describe the JSONL rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STEP_RE = re.compile(r"\bTrain \| Step:\s*(\d+)\s+(.*)")
_KV_RE = re.compile(r"([A-Za-z0-9_/]+):\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WINDOW = 128


def find_console_log(dump_folder: str) -> str:
    console = os.path.join(dump_folder, "console.log")
    if os.path.isfile(console):
        return console
    logs = [
        path
        for pattern in ("run-*", "offline-run-*")
        for path in glob.glob(
            os.path.join(dump_folder, "wandb", pattern, "files", "output.log")
        )
    ]
    if not logs:
        raise SystemExit(
            f"no console.log or W&B output.log under {dump_folder}; "
            "pass the saved console log with --log"
        )
    return max(logs, key=os.path.getmtime)


def parse_steps(log_path: str) -> list[tuple[int, dict[str, float]]]:
    steps = {}
    with open(log_path, errors="replace") as file:
        for line in file:
            match = _STEP_RE.search(_ANSI_RE.sub("", line))
            if match:
                values = {
                    key: float(value) for key, value in _KV_RE.findall(match.group(2))
                }
                steps[int(match.group(1))] = values
    return sorted(steps.items())


def load_train_rollouts(dump_folder: str) -> list[dict]:
    rows = []
    with open(os.path.join(dump_folder, "rollout_samples.jsonl")) as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("is_validation"):
                rows.append(row)
    # Preserve physical recording order: resumed runs may replay older IDs.
    return rows


def outcome_bins(rows: list[dict], num_bins: int = 8) -> list[dict[str, float]]:
    bins: list[list[dict]] = [[] for _ in range(num_bins)]
    for index, row in enumerate(rows):
        # Recording order remains meaningful when a resume replays older IDs.
        bins[min(num_bins - 1, index * num_bins // len(rows))].append(row)
    out = []
    for bucket in bins:
        if not bucket:
            continue
        counts = Counter(row["status"] for row in bucket)
        env_budget = sum(
            row["status"] == "completed"
            and any(
                (turn.get("env_rewards") or {}).get("turn_budget_exhausted", 0)
                for turn in row.get("turns", [])
            )
            for row in bucket
        )
        completed = counts.get("completed", 0) - env_budget
        turns = counts.get("truncated_max_turns", 0)
        tokens = counts.get("truncated_length", 0) + counts.get(
            "truncated_prompt_too_long", 0
        )
        n = len(bucket)
        out.append(
            {
                "completed": 100 * completed / n,
                "env_budget": 100 * env_budget / n,
                "out_of_turns": 100 * turns / n,
                "out_of_tokens": 100 * tokens / n,
                "error_or_other": 100
                * (n - completed - env_budget - turns - tokens)
                / n,
            }
        )
    return out


def _num_tool_calls(rollout: dict) -> int:
    return sum(
        len((turn.get("completion_message") or {}).get("tool_calls") or [])
        for turn in rollout["turns"]
    )


def _first_turn_reasoning_chars(rollout: dict) -> int:
    if not rollout["turns"]:
        return 0
    message = rollout["turns"][0].get("completion_message") or {}
    return len(message.get("reasoning_content") or "")


def behavior(window: list[dict]) -> dict[str, float]:
    calls = [_num_tool_calls(row) for row in window]
    return {
        "zero_requested_tool_calls_pct": 100 * sum(n == 0 for n in calls) / len(window),
        "median_first_turn_reasoning_chars": statistics.median(
            _first_turn_reasoning_chars(row) for row in window
        ),
        # Parsed calls can be retained when truncation prevents env execution.
        "median_requested_tool_calls": statistics.median(calls),
        "correct_pct": 100
        * sum((row.get("reward") or 0) >= 1.0 for row in window)
        / len(window),
    }


def plot_reward(steps, late_rollouts_all_correct: bool, path: str) -> None:
    xs = [step for step, _ in steps]
    ys = [values["rollout_reward/_mean"] for _, values in steps]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.plot(xs, ys, marker="o", ms=5, lw=2.2, color="#1f6feb", zorder=3)
    ax.fill_between(xs, 0, ys, color="#1f6feb", alpha=0.08, zorder=1)
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("mean rollout reward")
    ax.set_title("Muse Glimmer 30B: data-analyst rollout reward", loc="left")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, max(xs) + 0.5)
    ax.set_xticks(xs)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(
        f"{ys[0]:.2f}", (xs[0], ys[0]), textcoords="offset points", xytext=(4, -14)
    )
    ax.annotate(
        f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(-18, 8)
    )
    if late_rollouts_all_correct:
        ax.text(
            0.99,
            0.06,
            "Late recorded rollouts are all correct.\n"
            "Check the console log for the actual stop reason.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_outcomes(bins, path: str) -> None:
    xs = list(range(len(bins)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.stackplot(
        xs,
        [bucket["completed"] for bucket in bins],
        [bucket["env_budget"] for bucket in bins],
        [bucket["out_of_turns"] for bucket in bins],
        [bucket["out_of_tokens"] for bucket in bins],
        [bucket["error_or_other"] for bucket in bins],
        labels=[
            "completed rollout",
            "environment budget",
            "turn limit",
            "token/context limit",
            "error/other",
        ],
        colors=["#2da44e", "#bf8700", "#d4a72c", "#cf222e", "#8250df"],
        alpha=0.85,
    )
    ax.set_xlabel("training progress (recorded rollouts, binned)")
    ax.set_ylabel("% of recorded rollouts")
    ax.set_title("How recorded rollouts end", loc="left")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.1, max(0.1, len(bins) - 1))
    ax.set_xticks([])
    ax.grid(alpha=0.2, lw=0.6, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump_folder")
    parser.add_argument("--log", help="saved console log (otherwise auto-detected)")
    parser.add_argument("--out", help="PNG directory (default: the dump folder)")
    args = parser.parse_args()
    log_path = args.log or find_console_log(args.dump_folder)
    steps = parse_steps(log_path)
    if not steps:
        raise SystemExit(f"no 'Train | Step:' lines in {log_path}")
    if any("rollout_reward/_mean" not in values for _, values in steps):
        raise SystemExit("console steps lack rollout_reward/_mean; enable that metric")
    rows = load_train_rollouts(args.dump_folder)
    if not rows:
        raise SystemExit(f"no training rollouts in {args.dump_folder}")
    if len(rows) < 2:
        raise SystemExit(
            "at least 2 training rollouts are required for before/after statistics"
        )
    bins = outcome_bins(rows)
    n = min(_WINDOW, len(rows) // 2)
    before, after = behavior(rows[:n]), behavior(rows[-n:])
    late_correct = after["correct_pct"] == 100.0
    out_dir = args.out or args.dump_folder
    os.makedirs(out_dir, exist_ok=True)
    reward_png = os.path.join(out_dir, "reward.png")
    outcomes_png = os.path.join(out_dir, "outcomes.png")
    plot_reward(steps, late_correct, reward_png)
    plot_outcomes(bins, outcomes_png)
    print(f"metrics source: {log_path}")
    occurrences = Counter((row["group_id"], row.get("rollout_id", 0)) for row in rows)
    repeats = sum(count - 1 for count in occurrences.values())
    print(
        f"rollout scope: entire append-only JSONL in recording order; "
        f"{repeats} repeated (group_id, rollout_id) occurrences"
    )
    print(
        "metrics and rollout histories may span different attempts; use matching inputs"
    )
    print(f"steps: {len(steps)} (late recorded rollouts all correct: {late_correct})")
    print(
        "reward per step: "
        + " ".join(f"{values['rollout_reward/_mean']:.2f}" for _, values in steps)
    )
    print("how recorded rollouts end, first bin -> last bin (%):")
    for key in (
        "completed",
        "env_budget",
        "out_of_turns",
        "out_of_tokens",
        "error_or_other",
    ):
        print(f"  {key:14s} {bins[0][key]:5.0f} -> {bins[-1][key]:5.0f}")
    print(f"before/after (first vs last {n} training rollouts):")
    for key in before:
        print(f"  {key:34s} {before[key]:8.1f} -> {after[key]:8.1f}")
    print(f"wrote {reward_png}\nwrote {outcomes_png}")


if __name__ == "__main__":
    main()
