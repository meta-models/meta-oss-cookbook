# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plot accuracy, reward, loss and held-out test scores from one or more runs.

    python plot.py outputs/muse-glimmer-data-analyst [more run dirs ...] --out assets

Reads ``trainer_state.json`` and ``test_results.json`` from each run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_PNG_METADATA = {
    "Copyright": "Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the LICENSE file "
    "in the root directory of the meta-oss-cookbook repository."
}


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(h["step"], h[key]) for h in history if h.get(key) is not None]
    return [s for s, _ in points], [v for _, v in points]


def _smooth(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, i - window + 1) : i + 1]) / len(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))]


def _curve(ax, history: list[dict], key: str, color: str | None = None, label: str | None = None) -> str:
    """Plot one logged metric per update: faint raw values plus a 5-update mean."""
    steps, values = _series(history, key)
    (line,) = ax.plot(steps, _smooth(values), color=color, label=label)
    ax.plot(steps, values, color=line.get_color(), alpha=0.2)
    return line.get_color()


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata=_PNG_METADATA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for run in args.runs:
        history = json.loads((run / "trainer_state.json").read_text())["log_history"]
        test = json.loads((run / "test_results.json").read_text())
        label = f"seed {test['trained']['seed']}" if "seed" in test["trained"] else run.name
        runs.append((label, history, test))
    tests = [(label, test) for label, _, test in runs]

    # Accuracy: training rollouts and held-out validation.
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 4))
    for label, history, _ in runs:
        color = _curve(ax_train, history, "rewards/accuracy/mean", label=label)
        steps, val_acc = _series(history, "eval_rewards/accuracy/mean")
        ax_val.plot(steps, val_acc, marker="o", color=color, label=label)
    ax_train.set(title="Training rollouts (5-update mean)", xlabel="GRPO update", ylabel="correct answer rate", ylim=(0, 1.02))
    ax_val.set(title="Held-out validation", xlabel="GRPO update", ylim=(0, 1.02))
    for ax in (ax_train, ax_val):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    _save(fig, args.out / "training.png")

    # Reward: the optimized signal, and how many prompt groups still carry a learning signal.
    fig, (ax_reward, ax_flat) = plt.subplots(1, 2, figsize=(11, 4))
    for label, history, _ in runs:
        color = _curve(ax_reward, history, "reward", label=label)
        _curve(ax_flat, history, "frac_reward_zero_std", color=color, label=label)
    ax_reward.set(title="Mean training reward (5-update mean)", xlabel="GRPO update", ylabel="reward", ylim=(0.5, 1.02))
    ax_flat.set(title="Prompt groups whose 8 rollouts all got the same reward", xlabel="GRPO update",
                ylabel="fraction of groups", ylim=(0, 1.02))
    for ax in (ax_reward, ax_flat):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    _save(fig, args.out / "reward.png")

    # Loss: the GRPO objective and the gradient it produces.
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(11, 4))
    for label, history, _ in runs:
        color = _curve(ax_loss, history, "loss", label=label)
        _curve(ax_grad, history, "grad_norm", color=color, label=label)
    ax_loss.axhline(0, color="black", linewidth=0.8)
    ax_loss.set(title="GRPO loss (5-update mean)", xlabel="GRPO update", ylabel="loss")
    ax_grad.set(title="Gradient norm (5-update mean)", xlabel="GRPO update", ylabel="L2 norm", ylim=(0, None))
    for ax in (ax_loss, ax_grad):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    _save(fig, args.out / "loss.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.38
    for i, (label, test) in enumerate(tests):
        for j, (name, color) in enumerate([("base", "#9e9e9e"), ("trained", "#1f77b4")]):
            r = test[name]
            n = r["num_tasks"] * r["samples_per_task"]
            ci = 1.96 * (r["accuracy"] * (1 - r["accuracy"]) / n) ** 0.5
            bar = ax.bar(i + (j - 0.5) * width, r["accuracy"], width, yerr=ci, capsize=3, color=color,
                         label=name if i == 0 else None)
            ax.bar_label(bar, labels=[f"{r['accuracy']:.1%}"], padding=2)
    ax.set_xticks(range(len(tests)), [label for label, _ in tests])
    r = tests[0][1]["base"]
    ax.set(title=f"Held-out test ({r['num_tasks']} tasks x {r['samples_per_task']} samples, 95% CI)",
           ylabel="correct answer rate", ylim=(0, 1.1))
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, args.out / "test.png")
    print(json.dumps({label: t for label, t in tests}, indent=2))


if __name__ == "__main__":
    main()
