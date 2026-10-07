#!/usr/bin/env python3
"""Plot exported training histories and deterministic evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

_HISTORY_ALIASES: dict[str, tuple[str, ...]] = {
    "loss": (
        "train/loss",
        "train_loss",
        "loss",
    ),
    "validation_loss": (
        "eval/loss",
        "eval_loss",
        "validation/loss",
        "validation_loss",
        "val/loss",
        "val_loss",
    ),
    "memory": (
        "sampled_peak_gpu_memory_gib",
        "train/memory/max_active (GiB)",
        "train/memory/max_allocated (GiB)",
        "train/memory/device_reserved (GiB)",
        "train/gpu_memory_allocated_gb",
        "gpu_memory_allocated_gb",
        "gpu_memory_gb",
        "memory_gb",
        "system/gpu.0.memoryallocatedbytes",
        "system/gpu.0.memory",
    ),
    "throughput": (
        "train/tokens/train_per_sec_per_gpu",
        "train/tokens_per_second",
        "tokens_per_second",
        "train/samples_per_second",
        "samples_per_second",
        "train_steps_per_second",
        "steps_per_second",
    ),
}
_STEP_ALIASES: tuple[str, ...] = (
    "train/global_step",
    "trainer/global_step",
    "global_step",
    "step",
    "_step",
)
_EVALUATION_ALIASES: dict[str, tuple[str, ...]] = {
    "normalized_exact_match": (
        "normalized_exact_match",
        "exact_match",
        "em",
    ),
    "token_f1": ("token_f1", "f1"),
}


def _flatten_mapping(
    value: Mapping[str, object], prefix: str = ""
) -> dict[str, object]:
    flattened = {}
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        flattened[name.lower()] = child
        if isinstance(child, Mapping):
            flattened.update(_flatten_mapping(child, name))
    return flattened


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        if not all(isinstance(row, dict) for row in value):
            raise ValueError("JSON history arrays must contain objects")
        return list(value)
    if not isinstance(value, dict):
        raise TypeError("JSON history must be an object or array")
    for key in ("history", "records", "rows", "data"):
        rows = value.get(key)
        if isinstance(rows, list):
            return _json_rows(rows)
    columns = {key: child for key, child in value.items() if isinstance(child, list)}
    if columns and len(columns) == len(value):
        lengths = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            raise ValueError("column-oriented JSON history has unequal lengths")
        return [
            {key: column[index] for key, column in columns.items()}
            for index in range(next(iter(lengths), 0))
        ]
    return [value]


def load_history(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    with path.open(encoding="utf-8") as handle:
        return _json_rows(json.load(handle))


def _first_number(
    flattened: Mapping[str, object], aliases: Iterable[str]
) -> float | None:
    for alias in aliases:
        value = _finite_float(flattened.get(alias.lower()))
        if value is not None:
            return value
    return None


def _memory_in_gib(value: float, source_name: str) -> float:
    return value / (1024**3) if "byte" in source_name.lower() else value


def _history_metric(flattened: Mapping[str, object], metric: str) -> float | None:
    for alias in _HISTORY_ALIASES[metric]:
        value = _finite_float(flattened.get(alias.lower()))
        if value is not None:
            return _memory_in_gib(value, alias) if metric == "memory" else value
    return _fallback_metric(flattened, metric)


def _fallback_metric(flattened: Mapping[str, object], metric: str) -> float | None:
    for key in sorted(flattened):
        lowered = key.lower()
        if (
            metric == "loss"
            and "loss" in lowered
            and not any(marker in lowered for marker in ("eval", "validation", "val/"))
        ):
            return _finite_float(flattened[key])
        if (
            metric == "validation_loss"
            and "loss" in lowered
            and any(marker in lowered for marker in ("eval", "validation", "val/"))
        ):
            return _finite_float(flattened[key])
        if (
            metric == "memory"
            and "memory" in lowered
            and any(marker in lowered for marker in ("gpu", "cuda", "vram"))
        ):
            value = _finite_float(flattened[key])
            return _memory_in_gib(value, lowered) if value is not None else None
        if metric == "throughput" and any(
            marker in lowered
            for marker in (
                "tokens_per_second",
                "samples_per_second",
                "steps_per_second",
                "throughput",
            )
        ):
            return _finite_float(flattened[key])
    return None


def aggregate_history(
    records: Iterable[Mapping[str, object]],
) -> dict[str, list[tuple[float, float]]]:
    """Normalize run-history aliases into deterministic, sorted metric series."""
    series = {metric: {} for metric in _HISTORY_ALIASES}
    for ordinal, record in enumerate(records):
        flattened = _flatten_mapping(record)
        step = _first_number(flattened, _STEP_ALIASES)
        if step is None:
            step = float(ordinal)
        for metric in _HISTORY_ALIASES:
            value = _history_metric(flattened, metric)
            if value is not None:
                series[metric][step] = value
    return {metric: sorted(points.items()) for metric, points in series.items()}


def load_evaluation(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"evaluation JSON must contain an object: {path}")
    return value


def _evaluation_summary(payload: Mapping[str, object]) -> dict[str, float]:
    for container_key in ("metrics", "summary"):
        container = payload.get(container_key)
        if isinstance(container, Mapping):
            flattened = _flatten_mapping(container)
            result = {}
            for metric, aliases in _EVALUATION_ALIASES.items():
                value = _first_number(flattened, aliases)
                if value is not None:
                    result[metric] = value
            if result:
                return result

    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        return {}
    result = {}
    for metric, aliases in _EVALUATION_ALIASES.items():
        values = []
        for example in examples:
            if not isinstance(example, Mapping):
                continue
            value = _first_number(_flatten_mapping(example), aliases)
            if value is not None:
                values.append(value)
        if values:
            result[metric] = sum(values) / len(values)
    return result


def aggregate_evaluations(
    evaluations: Iterable[tuple[str, Mapping[str, object]]],
) -> dict[str, dict[str, float]]:
    """Return metric-oriented values sorted by evaluation label."""
    aggregated = {metric: {} for metric in _EVALUATION_ALIASES}
    seen_labels = set()
    for label, payload in sorted(evaluations, key=lambda item: item[0]):
        if label in seen_labels:
            raise ValueError(f"duplicate evaluation label: {label}")
        seen_labels.add(label)
        summary = _evaluation_summary(payload)
        for metric, value in summary.items():
            aggregated[metric][label] = value
    return aggregated


def _labeled_path(specification: str) -> tuple[str, Path]:
    if "=" in specification:
        label, raw_path = specification.split("=", 1)
        if not label:
            raise ValueError(f"empty label in input: {specification}")
    else:
        raw_path = specification
        label = Path(raw_path).stem
    return label, Path(raw_path)


def _save_line_plot(
    pyplot: object,
    runs: Mapping[str, Mapping[str, list[tuple[float, float]]]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    figure, axis = pyplot.subplots(figsize=(8, 4.5), dpi=120)
    plotted = False
    for label in sorted(runs):
        points = runs[label].get(metric, [])
        if not points:
            continue
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=label,
            linewidth=1.8,
            marker="o" if len(points) == 1 else None,
        )
        plotted = True
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No matching data", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(
        output,
        dpi=120,
        metadata={"Software": "meta-oss-cookbook"},
    )
    pyplot.close(figure)


def _save_evaluation_plot(
    pyplot: object,
    metrics: Mapping[str, Mapping[str, float]],
    output: Path,
) -> None:
    labels = sorted({label for values in metrics.values() for label in values})
    metric_names = [metric for metric in _EVALUATION_ALIASES if metrics.get(metric)]
    figure, axis = pyplot.subplots(figsize=(8, 4.5), dpi=120)
    if labels and metric_names:
        width = 0.8 / len(metric_names)
        centers = list(range(len(labels)))
        for metric_index, metric in enumerate(metric_names):
            offset = (metric_index - (len(metric_names) - 1) / 2) * width
            axis.bar(
                [center + offset for center in centers],
                [metrics[metric].get(label, 0.0) for label in labels],
                width=width,
                label=metric.replace("_", " "),
            )
        axis.set_xticks(centers, labels, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
    axis.set_title("Base versus tuned evaluation")
    axis.set_ylabel("Score")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output,
        dpi=120,
        metadata={"Software": "meta-oss-cookbook"},
    )
    pyplot.close(figure)


def plot_results(
    histories: Sequence[str], evaluations: Sequence[str], output_directory: Path
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError as error:
        raise RuntimeError("plotting requires matplotlib") from error

    output_directory.mkdir(parents=True, exist_ok=True)
    run_series = {}
    for specification in histories:
        label, path = _labeled_path(specification)
        if label in run_series:
            raise ValueError(f"duplicate history label: {label}")
        run_series[label] = aggregate_history(load_history(path))

    evaluation_payloads = []
    for specification in evaluations:
        label, path = _labeled_path(specification)
        payload = load_evaluation(path)
        if "=" not in specification:
            stored_label = payload.get("label")
            if isinstance(stored_label, str) and stored_label:
                label = stored_label
        evaluation_payloads.append((label, payload))
    evaluation_metrics = aggregate_evaluations(evaluation_payloads)

    specifications = (
        ("loss", "Training loss", "Loss", "loss.png"),
        (
            "validation_loss",
            "Validation loss",
            "Loss",
            "validation_loss.png",
        ),
        ("memory", "Accelerator memory", "Memory (GiB)", "memory.png"),
        ("throughput", "Training throughput", "Tokens/s/GPU", "throughput.png"),
    )
    outputs = []
    for metric, title, ylabel, filename in specifications:
        output = output_directory / filename
        _save_line_plot(
            pyplot,
            run_series,
            metric=metric,
            title=title,
            ylabel=ylabel,
            output=output,
        )
        outputs.append(output)
    evaluation_output = output_directory / "base_vs_tuned_metrics.png"
    _save_evaluation_plot(pyplot, evaluation_metrics, evaluation_output)
    outputs.append(evaluation_output)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Exported run-history JSON or CSV; repeat for multiple runs",
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help="Evaluation JSON; repeat for base and tuned results",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.history and not args.evaluation:
        parser.error("provide at least one --history or --evaluation input")
    try:
        outputs = plot_results(args.history, args.evaluation, args.output_dir)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
