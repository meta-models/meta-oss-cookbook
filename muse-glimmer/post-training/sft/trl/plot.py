# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rebuild the README plots from one or more training run directories.

    python plot.py outputs/run-seed42 outputs/run-seed1234 --out assets

Each directory needs the ``trainer_state.json`` and ``test_results.json`` that
``train.py`` writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE, TUNED = "#9aa5b1", "#1f6feb"


def load(run: Path) -> dict:
    state = json.loads((run / "trainer_state.json").read_text())
    results = json.loads((run / "test_results.json").read_text())
    history = state["log_history"]
    return {
        "label": f"seed {results['seed']}",
        "train": [(row["step"], row["loss"]) for row in history if "loss" in row],
        "validation": [
            (row["step"], row["eval_loss"]) for row in history if "eval_loss" in row
        ],
        "results": results,
    }


def plot_loss(runs: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, run in enumerate(runs):
        color = plt.cm.tab10(i)
        steps, loss = zip(*run["train"])
        ax.plot(steps, loss, color=color, alpha=0.35, lw=1)
        steps, loss = zip(*run["validation"])
        ax.plot(
            steps,
            loss,
            color=color,
            marker="o",
            lw=2,
            label=f"{run['label']} validation",
        )
    ax.plot([], [], color="gray", alpha=0.5, label="training (per step)")
    ax.set(xlabel="optimizer step", ylabel="loss on assistant tokens", ylim=(0, None))
    ax.set_title("Completion-only loss, 256 held-out validation questions")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=150)
    plt.close(fig)


def plot_test(runs: list[dict], out: Path) -> None:
    first = runs[0]["results"]["base"]
    n_all, n_read = first["num_questions"], first["read_off"]["num_questions"]
    groups = [
        (f"Exact match\nall {n_all}", None, "exact_match"),
        (f"Contains answer\nall {n_all}", None, "contains_answer"),
        (f"Exact match\nread-off {n_read}", "read_off", "exact_match"),
        (f"Contains answer\nread-off {n_read}", "read_off", "contains_answer"),
    ]

    def values(which: str, subset: str | None, key: str) -> list[float]:
        rows = [run["results"][which] for run in runs]
        return [100 * (row[subset][key] if subset else row[key]) for row in rows]

    fig, (ax, ax_tokens) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]}
    )
    width = 0.38
    for i, (label, subset, key) in enumerate(groups):
        for offset, which, color in (
            (-width / 2, "base", BASE),
            (width / 2, "tuned", TUNED),
        ):
            vals = values(which, subset, key)
            mean = sum(vals) / len(vals)
            ax.bar(
                i + offset, mean, width, color=color, label=which if i == 0 else None
            )
            ax.scatter([i + offset] * len(vals), vals, color="black", s=10, zorder=3)
            ax.text(i + offset, max(vals) + 1.5, f"{mean:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set(ylabel="% of held-out questions", ylim=(0, 100))
    ax.set_title("Held-out test: base vs tuned (dots are seeds)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for x, which, color in ((0, "base", BASE), (1, "tuned", TUNED)):
        vals = [run["results"][which]["mean_generated_tokens"] for run in runs]
        mean = sum(vals) / len(vals)
        ax_tokens.bar(x, mean, 0.6, color=color)
        ax_tokens.text(x, mean * 1.02, f"{mean:.0f}", ha="center", fontsize=8)
    ax_tokens.set_xticks([0, 1], ["base", "tuned"])
    ax_tokens.set_title("Tokens per answer")
    ax_tokens.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = [load(run) for run in args.runs]
    plot_loss(runs, args.out)
    plot_test(runs, args.out)
    print(f"wrote {args.out / 'loss.png'} and {args.out / 'test.png'}")


if __name__ == "__main__":
    main()
