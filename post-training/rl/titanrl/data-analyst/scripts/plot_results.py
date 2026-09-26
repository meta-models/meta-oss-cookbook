"""Plot a data-analyst training run and print its before/after numbers.

Reads only the run's --dump-folder:
  * per-step metrics from W&B's local console log (wandb/run-*/files/output.log),
    or from a saved console log passed with --log;
  * per-rollout outcomes from rollout_samples.jsonl.

    python plot_results.py outputs/rl/glimmer_data_analyst [--out DIR] [--log FILE]

Writes reward.png and outcomes.png, then prints the numbers the recipe README
quotes (reward per step, how rollouts end, before/after behavior).
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

_STEP_RE = re.compile(r"^Train \| Step:\s*(\d+)\s+(.*)")
_KV_RE = re.compile(r"([A-Za-z0-9_/]+):\s*(-?[\d.]+(?:e[-+]?\d+)?)")
# Before/after windows: the first and last N training rollouts of the run.
_WINDOW = 128


def find_console_log(dump_folder: str) -> str:
    logs = sorted(
        glob.glob(os.path.join(dump_folder, "wandb", "run-*", "files", "output.log")),
        key=os.path.getmtime,
    )
    if not logs:
        raise SystemExit(
            f"no wandb/run-*/files/output.log under {dump_folder}; "
            "pass the saved console log with --log"
        )
    return logs[-1]


def parse_steps(log_path: str) -> list[tuple[int, dict[str, float]]]:
    steps = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _STEP_RE.match(line.strip())
            if m:
                kv = {k: float(v) for k, v in _KV_RE.findall(m.group(2))}
                steps[int(m.group(1))] = kv
    return sorted(steps.items())


def load_train_rollouts(dump_folder: str) -> list[dict]:
    rows = []
    with open(os.path.join(dump_folder, "rollout_samples.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("is_validation"):
                rows.append(row)
    rows.sort(key=lambda r: r["group_id"])
    return rows


def outcome_bins(rows: list[dict], num_bins: int = 8) -> list[dict[str, float]]:
    lo, hi = rows[0]["group_id"], rows[-1]["group_id"]
    width = (hi - lo + 1) / num_bins
    bins: list[list[dict]] = [[] for _ in range(num_bins)]
    for r in rows:
        bins[min(num_bins - 1, int((r["group_id"] - lo) / width))].append(r)
    out = []
    for b in bins:
        if not b:
            continue
        c = Counter(r["status"] for r in b)
        n = len(b)
        out.append(
            {
                "completed": 100 * c.get("completed", 0) / n,
                "out_of_turns": 100 * c.get("truncated_max_turns", 0) / n,
                "out_of_tokens": 100
                * (c.get("truncated_length", 0) + c.get("truncated_prompt_too_long", 0))
                / n,
            }
        )
    return out


def _num_tool_calls(rollout: dict) -> int:
    return sum(
        1 for t in rollout["turns"] if (t.get("completion_message") or {}).get("tool_calls")
    )


def _first_turn_reasoning_chars(rollout: dict) -> int:
    if not rollout["turns"]:
        return 0
    msg = rollout["turns"][0].get("completion_message") or {}
    return len(msg.get("reasoning_content") or "")


def behavior(window: list[dict]) -> dict[str, float]:
    calls = [_num_tool_calls(r) for r in window]
    return {
        "zero_tool_calls_pct": 100 * sum(1 for n in calls if n == 0) / len(window),
        "median_first_turn_reasoning_chars": statistics.median(
            _first_turn_reasoning_chars(r) for r in window
        ),
        "median_tool_calls": statistics.median(calls),
        "correct_pct": 100 * sum(1 for r in window if (r.get("reward") or 0) >= 1.0) / len(window),
    }


def plot_reward(steps, stopped_by_saturation: bool, path: str) -> None:
    xs = [s for s, _ in steps]
    ys = [kv["rollout_reward/_mean"] for _, kv in steps]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.plot(xs, ys, marker="o", ms=5, lw=2.2, color="#1f6feb", zorder=3)
    ax.fill_between(xs, 0, ys, color="#1f6feb", alpha=0.08, zorder=1)
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("mean rollout reward")
    ax.set_title(
        "Muse Glimmer 30B, data-analyst recipe: mean rollout reward per step",
        fontsize=11,
        loc="left",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, max(xs) + 0.5)
    ax.set_xticks(xs)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), textcoords="offset points",
                xytext=(4, -14), fontsize=9, color="#57606a")
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-18, 8), fontsize=9, color="#1f6feb", weight="bold")
    if stopped_by_saturation:
        ax.text(0.99, 0.06,
                f"run ends at step {xs[-1]}: every rollout now succeeds,\n"
                "so GRPO has no reward variance left to learn from",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="#57606a", style="italic")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_outcomes(bins, path: str) -> None:
    xs = list(range(len(bins)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    ax.stackplot(
        xs,
        [b["completed"] for b in bins],
        [b["out_of_turns"] for b in bins],
        [b["out_of_tokens"] for b in bins],
        labels=["completed the task", "ran out of turns", "ran out of tokens mid-answer"],
        colors=["#2da44e", "#d4a72c", "#cf222e"],
        alpha=0.85,
    )
    ax.set_xlabel("training progress  (rollouts, binned; left = start of run)")
    ax.set_ylabel("% of rollouts")
    ax.set_title("How rollouts end, over the course of training", fontsize=11, loc="left")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(bins) - 1)
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
    parser.add_argument("--log", help="saved console log (default: W&B's local output.log)")
    parser.add_argument("--out", help="where to write the PNGs (default: the dump folder)")
    args = parser.parse_args()

    log_path = args.log or find_console_log(args.dump_folder)
    out_dir = args.out or args.dump_folder
    os.makedirs(out_dir, exist_ok=True)

    steps = parse_steps(log_path)
    if not steps:
        raise SystemExit(f"no 'Train | Step:' lines in {log_path}")
    rows = load_train_rollouts(args.dump_folder)
    bins = outcome_bins(rows)
    n = min(_WINDOW, len(rows) // 2)
    before, after = behavior(rows[:n]), behavior(rows[-n:])
    # Saturated: every late rollout scores full marks, so GRPO groups have no
    # reward variance left (TitanRL then stops with "consecutive untrainable
    # batches", which W&B's console copy does not always capture).
    saturated = after["correct_pct"] >= 99.0

    reward_png = os.path.join(out_dir, "reward.png")
    outcomes_png = os.path.join(out_dir, "outcomes.png")
    plot_reward(steps, saturated, reward_png)
    plot_outcomes(bins, outcomes_png)

    rewards = [kv["rollout_reward/_mean"] for _, kv in steps]
    print(f"steps: {len(steps)}  (stopped by saturation: {saturated})")
    print("reward per step: " + " ".join(f"{r:.2f}" for r in rewards))
    print(f"how rollouts end, first bin -> last bin (%):")
    for key in ("completed", "out_of_turns", "out_of_tokens"):
        print(f"  {key:14s} {bins[0][key]:5.0f} -> {bins[-1][key]:5.0f}")
    print(f"before/after (first vs last {n} training rollouts):")
    for key in before:
        print(f"  {key:34s} {before[key]:8.1f} -> {after[key]:8.1f}")
    print(f"wrote {reward_png}\nwrote {outcomes_png}")


if __name__ == "__main__":
    main()
