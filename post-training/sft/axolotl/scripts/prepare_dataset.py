#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare small, reproducible SFT datasets for Muse Glimmer.

The output is JSONL in the OpenAI Messages shape accepted by Axolotl. Image
bytes are decoded once, normalized to RGB PNG files, and referenced by local
paths. Imports of datasets, Pillow, and Transformers are lazy so the pure data
validation and splitting helpers remain easy to test.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.05
DEFAULT_IMAGE_SIZE = 512
DEFAULT_MAX_IMAGES = 4
DEFAULT_MAX_IMAGE_PIXELS = 100_000_000
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "prepared"
DEFAULT_TOKENIZER_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
RECIPE_ROOT = Path(__file__).resolve().parents[1]


def _find_repository_root(recipe_root: Path) -> Path:
    for candidate in (recipe_root, *recipe_root.parents):
        if (candidate / "README.md").is_file() and (
            candidate / "post-training"
        ).is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root above {recipe_root}")


REPOSITORY_ROOT = _find_repository_root(RECIPE_ROOT)

LengthCounter = Callable[[list[dict[str, Any]]], int]


class PreparationError(ValueError):
    """Raised when source data cannot be prepared without ambiguity."""


@dataclass(frozen=True)
class SourceSpec:
    """Immutable source metadata for one preparation track."""

    dataset_id: str
    revision: str
    config_name: str | None
    source_splits: tuple[str, ...]
    expected_rows: Mapping[str, int]
    output_name: str
    license_name: str


SOURCES: dict[str, SourceSpec] = {
    "tulu": SourceSpec(
        dataset_id=("allenai/tulu-3-sft-personas-instruction-following"),
        revision="fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        config_name=None,
        source_splits=("train",),
        expected_rows={"train": 29_980},
        output_name="tulu",
        license_name="ODC-BY-1.0",
    ),
    "cosyn-circuit": SourceSpec(
        dataset_id="allenai/CoSyn-400K",
        revision="86e46e1fd5e754d056169f0fb38f06c6997ff7de",
        config_name="circuit",
        source_splits=("train", "validation"),
        expected_rows={"train": 10_470, "validation": 128},
        output_name="cosyn-circuit",
        license_name="ODC-BY-1.0",
    ),
    "docmatix": SourceSpec(
        dataset_id="HuggingFaceM4/Docmatix",
        revision="0725b65616e0e5f6024be10e38ddf8d8c48664fd",
        config_name="zero-shot-exp",
        source_splits=("train", "test"),
        expected_rows={"train": 1_700, "test": 200},
        output_name="docmatix",
        license_name="MIT",
    ),
}


@dataclass
class ExampleMetrics:
    """Preflight measurements for one normalized conversation."""

    messages: int
    assistant_messages: int
    images: int
    text_chars: int
    tokens: int | None = None


@dataclass
class SplitStats:
    """Aggregate measurements written to the preparation manifest."""

    examples: int = 0
    messages: int = 0
    assistant_messages: int = 0
    images: int = 0
    text_chars: int = 0
    min_images: int | None = None
    max_images: int = 0
    min_text_chars: int | None = None
    max_text_chars: int = 0
    min_tokens: int | None = None
    max_tokens: int | None = None
    image_count_histogram: dict[str, int] = field(default_factory=dict)

    def add(self, metrics: ExampleMetrics) -> None:
        """Add one validated example to the aggregate."""

        self.examples += 1
        self.messages += metrics.messages
        self.assistant_messages += metrics.assistant_messages
        self.images += metrics.images
        self.min_images = _minimum(self.min_images, metrics.images)
        self.max_images = max(self.max_images, metrics.images)
        self.text_chars += metrics.text_chars
        self.min_text_chars = _minimum(
            self.min_text_chars,
            metrics.text_chars,
        )
        self.max_text_chars = max(self.max_text_chars, metrics.text_chars)
        key = str(metrics.images)
        self.image_count_histogram[key] = (
            self.image_count_histogram.get(key, 0) + 1
        )
        if metrics.tokens is not None:
            self.min_tokens = _minimum(self.min_tokens, metrics.tokens)
            self.max_tokens = max(self.max_tokens or 0, metrics.tokens)


def _minimum(current: int | None, candidate: int) -> int:
    return candidate if current is None else min(current, candidate)


def deterministic_validation_assignment(
    example_id: str,
    *,
    seed: int,
    validation_fraction: float,
) -> bool:
    """Return a stable train/validation assignment for an example id."""

    if not 0.0 <= validation_fraction < 1.0:
        raise PreparationError(
            "validation_fraction must be at least 0 and less than 1"
        )
    if validation_fraction == 0.0:
        return False
    digest = hashlib.sha256(f"{seed}\0{example_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big")
    cutoff = int(validation_fraction * (1 << 64))
    return bucket < cutoff


def normalize_tulu(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one Tulu row without changing its conversational content."""

    example_id = _required_string(row, "id")
    source_messages = row.get("messages")
    if not isinstance(source_messages, Sequence) or isinstance(
        source_messages,
        (str, bytes),
    ):
        raise PreparationError(f"{example_id}: messages must be a list")

    messages = []
    for message in source_messages:
        if not isinstance(message, Mapping):
            raise PreparationError(
                f"{example_id}: every message must be an object"
            )
        messages.append(
            {
                "role": _required_string(message, "role"),
                "content": _required_message_content(message, "content"),
            }
        )
    return {"id": example_id, "messages": messages}


def normalize_cosyn(
    row: Mapping[str, Any],
    *,
    image_path: str,
) -> list[dict[str, Any]]:
    """Turn one CoSyn row into one two-turn dialogue per QA pair."""

    example_id = _required_string(row, "id")
    pairs = list(_iter_cosyn_pairs(row.get("qa_pairs"), example_id))
    if not pairs:
        raise PreparationError(f"{example_id}: qa_pairs must not be empty")

    examples = []
    for index, (question, explanation, answer) in enumerate(pairs):
        examples.append(
            {
                "id": f"{example_id}-qa-{index:02d}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": _answer_with_explanation(
                            explanation,
                            answer,
                        ),
                    },
                ],
            }
        )
    return examples


def normalize_docmatix(
    row: Mapping[str, Any],
    *,
    example_id: str,
    image_paths: Sequence[str],
) -> dict[str, Any]:
    """Turn an ordered Docmatix page list and QA list into one dialogue."""

    turns = row.get("texts")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise PreparationError(f"{example_id}: texts must be a list")
    if not turns:
        raise PreparationError(f"{example_id}: texts must not be empty")

    messages: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise PreparationError(
                f"{example_id}: every texts item must be an object"
            )
        question = _required_string(turn, "user")
        answer = _required_string(turn, "assistant")
        if index == 0:
            content: str | list[dict[str, str]] = [
                {"type": "image", "path": path} for path in image_paths
            ]
            content.append({"type": "text", "text": question})
        else:
            content = question
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": answer})
    return {"id": example_id, "messages": messages}


def _iter_cosyn_pairs(
    value: Any,
    example_id: str,
) -> Iterable[tuple[str, str, str]]:
    """Accept both HF Sequence(dict) layouts used by CoSyn readers."""

    if isinstance(value, Mapping):
        questions = _string_list(value.get("question"), "question", example_id)
        explanations = _string_list(
            value.get("explanation"),
            "explanation",
            example_id,
            allow_empty=True,
        )
        answers = _string_list(value.get("answer"), "answer", example_id)
        lengths = {len(questions), len(explanations), len(answers)}
        if len(lengths) != 1:
            raise PreparationError(
                f"{example_id}: qa_pairs fields have different lengths"
            )
        yield from zip(questions, explanations, answers, strict=True)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if not isinstance(item, Mapping):
                raise PreparationError(
                    f"{example_id}: qa_pairs items must be objects"
                )
            yield (
                _required_string(item, "question"),
                _required_string(item, "explanation", allow_empty=True),
                _required_string(item, "answer"),
            )
        return

    raise PreparationError(f"{example_id}: qa_pairs has an unknown shape")


def _string_list(
    value: Any,
    field_name: str,
    example_id: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PreparationError(
            f"{example_id}: qa_pairs.{field_name} must be a list"
        )
    result = []
    for item in value:
        if not isinstance(item, str):
            raise PreparationError(
                f"{example_id}: qa_pairs.{field_name} must contain strings"
            )
        cleaned = item.strip()
        if not allow_empty and not cleaned:
            raise PreparationError(
                f"{example_id}: qa_pairs.{field_name} must not contain empty strings"
            )
        result.append(cleaned)
    return result


def _answer_with_explanation(explanation: str, answer: str) -> str:
    explanation = explanation.strip()
    answer = answer.strip()
    if not explanation or explanation == answer:
        return answer
    return f"{explanation}\n\nAnswer: {answer}"


def _required_message_content(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise PreparationError(f"{key} must be a string")
    if not value.strip():
        raise PreparationError(f"{key} must not be empty")
    return value


def _required_string(
    row: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise PreparationError(f"{key} must be a string")
    value = value.strip()
    if not allow_empty and not value:
        raise PreparationError(f"{key} must not be empty")
    return value


def preflight_messages(
    messages: Any,
    *,
    expected_images: tuple[int, int],
    max_images: int,
    max_text_chars: int | None,
    length_counter: LengthCounter | None = None,
    max_tokens: int | None = None,
    check_paths: bool = True,
) -> ExampleMetrics:
    """Validate one conversation and return useful size measurements."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise PreparationError("messages must be a list")
    if not messages:
        raise PreparationError("messages must not be empty")

    image_count = 0
    text_chars = 0
    assistant_count = 0
    for message in messages:
        if not isinstance(message, Mapping):
            raise PreparationError("each message must be an object")
        role = _required_string(message, "role")
        if role == "assistant":
            assistant_count += 1
        content = message.get("content")
        if isinstance(content, str):
            if not content.strip():
                raise PreparationError("message content must not be empty")
            text_chars += len(content)
            continue
        if not isinstance(content, Sequence) or isinstance(content, bytes):
            raise PreparationError("message content must be text or a list")
        if not content:
            raise PreparationError("message content parts must not be empty")
        for part in content:
            if not isinstance(part, Mapping):
                raise PreparationError("message content parts must be objects")
            part_type = _required_string(part, "type")
            if part_type == "text":
                text_chars += len(_required_string(part, "text"))
            elif part_type == "image":
                image_count += 1
                media_keys = [
                    key
                    for key in ("image", "path", "url", "base64")
                    if part.get(key) is not None
                ]
                if len(media_keys) != 1:
                    raise PreparationError(
                        "each image part must contain exactly one media source"
                    )
                if (
                    check_paths
                    and media_keys[0] == "path"
                    and not Path(str(part["path"])).is_file()
                ):
                    raise PreparationError(
                        f"image path does not exist: {part['path']}"
                    )
            else:
                raise PreparationError(
                    f"unsupported message content type: {part_type}"
                )

    if assistant_count == 0:
        raise PreparationError("conversation has no assistant target")
    minimum_images, scenario_maximum = expected_images
    allowed_maximum = min(scenario_maximum, max_images)
    if not minimum_images <= image_count <= allowed_maximum:
        raise PreparationError(
            "image count is outside the configured range: "
            f"got {image_count}, expected {minimum_images}..{allowed_maximum}"
        )
    if max_text_chars is not None and text_chars > max_text_chars:
        raise PreparationError(
            f"text length {text_chars} exceeds max_text_chars={max_text_chars}"
        )

    token_count = length_counter(list(messages)) if length_counter else None
    if token_count is not None and token_count <= 0:
        raise PreparationError("length_counter returned a non-positive value")
    if (
        max_tokens is not None
        and token_count is not None
        and token_count > max_tokens
    ):
        raise PreparationError(
            f"token length {token_count} exceeds max_tokens={max_tokens}"
        )

    return ExampleMetrics(
        messages=len(messages),
        assistant_messages=assistant_count,
        images=image_count,
        text_chars=text_chars,
        tokens=token_count,
    )


def build_processor_length_counter(
    model_id: str,
    *,
    revision: str = DEFAULT_TOKENIZER_REVISION,
    local_files_only: bool = False,
) -> LengthCounter:
    """Build an exact multimodal token counter without loading model weights."""

    try:
        from transformers import AutoProcessor
    except ImportError as error:
        raise PreparationError(
            "Transformers is required when --tokenizer-model is set"
        ) from error

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )

    def count(messages: list[dict[str, Any]]) -> int:
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    return count


def prepare_dataset(
    scenario: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    max_examples_per_split: int | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_text_chars: int | None = None,
    length_counter: LengthCounter | None = None,
    tokenizer_model: str | None = None,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    tokenizer_local_files_only: bool = False,
    max_tokens: int | None = None,
    drop_over_max_tokens: bool = False,
    overwrite: bool = False,
    source_splits: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    verify_source_counts: bool = True,
) -> Path:
    """Prepare one named source and return its final output directory."""

    if scenario not in SOURCES:
        raise PreparationError(f"unknown dataset scenario: {scenario}")
    if not 0.0 <= validation_fraction < 1.0:
        raise PreparationError(
            "validation_fraction must be at least 0 and less than 1"
        )
    if max_examples_per_split is not None and max_examples_per_split <= 0:
        raise PreparationError("max_examples_per_split must be positive")
    if image_size < 0:
        raise PreparationError("image_size must be at least 0")
    if max_images <= 0:
        raise PreparationError("max_images must be positive")
    if max_image_pixels <= 0:
        raise PreparationError("max_image_pixels must be positive")
    if max_text_chars is not None and max_text_chars <= 0:
        raise PreparationError("max_text_chars must be positive")
    if tokenizer_local_files_only and not tokenizer_model:
        raise PreparationError(
            "tokenizer_local_files_only requires tokenizer_model"
        )
    if tokenizer_model and length_counter is None:
        length_counter = build_processor_length_counter(
            tokenizer_model,
            revision=tokenizer_revision,
            local_files_only=tokenizer_local_files_only,
        )
    if max_tokens is not None and max_tokens <= 0:
        raise PreparationError("max_tokens must be positive")
    if max_tokens is not None and length_counter is None:
        raise PreparationError(
            "max_tokens requires a length_counter or --tokenizer-model"
        )

    spec = SOURCES[scenario]
    output_root = _validate_output_root(Path(output_root))
    final_dir = output_root / spec.output_name
    _validate_output_target(final_dir)
    if final_dir.is_symlink():
        raise PreparationError(f"refusing symlink output directory: {final_dir}")
    if final_dir.exists() and not overwrite:
        raise PreparationError(
            f"output already exists: {final_dir}; pass --overwrite to "
            "replace it"
        )
    if final_dir.exists():
        _validate_owned_output(
            final_dir,
            scenario=scenario,
            output_name=spec.output_name,
        )

    loaded_internally = source_splits is None
    if source_splits is None:
        source_splits = _load_source_splits(spec)
    if loaded_internally and verify_source_counts:
        _verify_source_counts(spec, source_splits)

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.mkdtemp(
        dir=output_root,
        prefix=f".{spec.output_name}-",
    )
    work_dir = Path(temporary)

    try:
        manifest = _prepare_into_directory(
            scenario,
            spec,
            source_splits,
            work_dir=work_dir,
            final_dir=final_dir,
            seed=seed,
            validation_fraction=validation_fraction,
            max_examples_per_split=max_examples_per_split,
            image_size=image_size,
            max_images=max_images,
            max_image_pixels=max_image_pixels,
            max_text_chars=max_text_chars,
            length_counter=length_counter,
            tokenizer_model=tokenizer_model,
            tokenizer_revision=tokenizer_revision,
            tokenizer_local_files_only=tokenizer_local_files_only,
            max_tokens=max_tokens,
            drop_over_max_tokens=drop_over_max_tokens,
            source_counts_verified=loaded_internally and verify_source_counts,
        )
        manifest_path = work_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        backup_dir: Path | None = None
        if final_dir.is_symlink():
            raise PreparationError(f"refusing symlink output directory: {final_dir}")
        if final_dir.exists():
            _validate_owned_output(
                final_dir,
                scenario=scenario,
                output_name=spec.output_name,
            )
            backup_dir = Path(
                tempfile.mkdtemp(
                    dir=output_root,
                    prefix=f".{spec.output_name}-backup-",
                )
            )
            backup_dir.rmdir()
            final_dir.replace(backup_dir)
        try:
            work_dir.replace(final_dir)
        except Exception:
            if backup_dir is not None and not final_dir.exists():
                backup_dir.replace(final_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    return final_dir


def _prepare_into_directory(
    scenario: str,
    spec: SourceSpec,
    source_splits: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    work_dir: Path,
    final_dir: Path,
    seed: int,
    validation_fraction: float,
    max_examples_per_split: int | None,
    image_size: int,
    max_images: int,
    max_image_pixels: int,
    max_text_chars: int | None,
    length_counter: LengthCounter | None,
    tokenizer_model: str | None,
    tokenizer_revision: str,
    tokenizer_local_files_only: bool,
    max_tokens: int | None,
    drop_over_max_tokens: bool,
    source_counts_verified: bool,
) -> dict[str, Any]:
    output_splits = _output_splits(scenario, validation_fraction)
    stats = {split: SplitStats() for split in output_splits}
    seen_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    source_rows_seen = {split: 0 for split in spec.source_splits}
    filtered_rows = 0
    filtered_examples = {"over_max_tokens": 0}
    media_files = {split: 0 for split in output_splits}
    selected_ids: dict[str, list[str]] = {split: [] for split in output_splits}

    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(
                (work_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            )
            for split in output_splits
        }
        for source_split in spec.source_splits:
            if source_split not in source_splits:
                raise PreparationError(
                    f"missing source split {source_split!r} for {scenario}"
                )
            source_rows = _source_rows_for_processing(
                scenario,
                source_split,
                source_splits[source_split],
                order_by_id=max_examples_per_split is not None,
            )
            for source_index, row in source_rows:
                relevant_splits = _targets_for_source(
                    scenario,
                    source_split,
                    validation_fraction,
                )
                if max_examples_per_split is not None and all(
                    _split_is_full(stats[split], max_examples_per_split)
                    for split in relevant_splits
                ):
                    break
                source_rows_seen[source_split] += 1
                image_values = _source_images(scenario, row)
                if scenario == "docmatix" and len(image_values) < 2:
                    filtered_rows += 1
                    continue
                if len(image_values) > max_images:
                    raise PreparationError(
                        f"{scenario} source row {source_index} has "
                        f"{len(image_values)} images; max_images={max_images}"
                    )

                example_id = _source_example_id(
                    scenario,
                    source_split,
                    row,
                    len(image_values),
                )
                if example_id in seen_source_ids:
                    raise PreparationError(
                        f"duplicate source example id: {example_id}"
                    )
                seen_source_ids.add(example_id)
                target_split = _target_split(
                    scenario,
                    source_split,
                    example_id,
                    seed=seed,
                    validation_fraction=validation_fraction,
                )
                if _split_is_full(
                    stats[target_split],
                    max_examples_per_split,
                ):
                    continue

                runtime_paths, final_paths = _save_images(
                    image_values,
                    example_id=example_id,
                    split=target_split,
                    work_dir=work_dir,
                    final_dir=final_dir,
                    image_size=image_size,
                    max_image_pixels=max_image_pixels,
                )
                media_files[target_split] += len(runtime_paths)
                runtime_examples = _normalize_rows(
                    scenario,
                    row,
                    example_id=example_id,
                    image_paths=runtime_paths,
                )
                output_examples = _normalize_rows(
                    scenario,
                    row,
                    example_id=example_id,
                    image_paths=final_paths,
                )
                accepted_from_source = 0
                for runtime, output in zip(
                    runtime_examples,
                    output_examples,
                    strict=True,
                ):
                    if _split_is_full(
                        stats[target_split],
                        max_examples_per_split,
                    ):
                        break
                    output_id = _required_string(output, "id")
                    if output_id in seen_ids:
                        raise PreparationError(
                            f"duplicate output example id: {output_id}"
                        )
                    expected_images = _expected_image_range(scenario)
                    try:
                        metrics = preflight_messages(
                            runtime["messages"],
                            expected_images=expected_images,
                            max_images=max_images,
                            max_text_chars=max_text_chars,
                            length_counter=length_counter,
                            max_tokens=max_tokens,
                        )
                    except PreparationError as error:
                        if drop_over_max_tokens and str(error).startswith(
                            "token length "
                        ):
                            filtered_examples["over_max_tokens"] += 1
                            continue
                        raise
                    seen_ids.add(output_id)
                    preflight_messages(
                        output["messages"],
                        expected_images=expected_images,
                        max_images=max_images,
                        max_text_chars=max_text_chars,
                        check_paths=False,
                    )
                    writers[target_split].write(
                        json.dumps(
                            output,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    selected_ids[target_split].append(output_id)
                    stats[target_split].add(metrics)
                    accepted_from_source += 1
                if accepted_from_source == 0:
                    for runtime_path in runtime_paths:
                        Path(runtime_path).unlink(missing_ok=True)
                    media_files[target_split] -= len(runtime_paths)

    _ensure_required_outputs(stats, scenario, validation_fraction)
    split_manifests = {}
    for split, split_stats in stats.items():
        split_manifest = asdict(split_stats)
        split_manifest["jsonl_sha256"] = _file_sha256(
            work_dir / f"{split}.jsonl"
        )
        split_manifest["selected_ids_sha256"] = _selected_ids_sha256(
            selected_ids[split]
        )
        split_manifest["image_content_sha256"] = _image_content_sha256(
            work_dir / "images" / split
        )
        split_manifests[split] = split_manifest
    return {
        "format_version": 2,
        "scenario": scenario,
        "output_name": spec.output_name,
        "source": {
            "dataset_id": spec.dataset_id,
            "config_name": spec.config_name,
            "revision": spec.revision,
            "license": spec.license_name,
            "expected_rows": dict(spec.expected_rows),
            "rows_seen": source_rows_seen,
            "counts_verified": source_counts_verified,
        },
        "preparation": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "max_examples_per_split": max_examples_per_split,
            "image_size": image_size if scenario != "tulu" else None,
            "max_images": max_images if scenario != "tulu" else 0,
            "max_image_pixels": (
                max_image_pixels if scenario != "tulu" else None
            ),
            "max_text_chars": max_text_chars,
            "tokenizer_model": tokenizer_model,
            "tokenizer_revision": (
                tokenizer_revision if tokenizer_model else None
            ),
            "tokenizer_local_files_only": (
                tokenizer_local_files_only if tokenizer_model else False
            ),
            "max_tokens": max_tokens,
            "drop_over_max_tokens": drop_over_max_tokens,
            "filtered_rows": filtered_rows,
            "filtered_examples": filtered_examples,
            "media_files": media_files,
            "cosyn_explanation_policy": (
                {
                    "storage": "visible_assistant_content",
                    "reasoning_content": False,
                    "answer_prefix": "Answer: ",
                    "omit_empty_or_duplicate_explanation": True,
                }
                if scenario == "cosyn-circuit"
                else None
            ),
        },
        "splits": split_manifests,
    }


def _output_splits(
    scenario: str,
    validation_fraction: float,
) -> tuple[str, ...]:
    if scenario == "cosyn-circuit":
        return ("train", "validation")
    if scenario == "docmatix":
        return ("train", "validation", "test")
    if validation_fraction > 0:
        return ("train", "validation")
    return ("train",)


def _target_split(
    scenario: str,
    source_split: str,
    example_id: str,
    *,
    seed: int,
    validation_fraction: float,
) -> str:
    if scenario == "cosyn-circuit":
        return source_split
    if scenario == "docmatix" and source_split == "test":
        return "test"
    if deterministic_validation_assignment(
        example_id,
        seed=seed,
        validation_fraction=validation_fraction,
    ):
        return "validation"
    return "train"


def _targets_for_source(
    scenario: str,
    source_split: str,
    validation_fraction: float,
) -> tuple[str, ...]:
    if scenario == "cosyn-circuit":
        return (source_split,)
    if scenario == "docmatix" and source_split == "test":
        return ("test",)
    if validation_fraction > 0:
        return ("train", "validation")
    return ("train",)


def _source_rows_for_processing(
    scenario: str,
    source_split: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    order_by_id: bool,
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    indexed_rows = enumerate(rows)
    if not order_by_id:
        return indexed_rows
    if (
        scenario in {"tulu", "cosyn-circuit"}
        and hasattr(rows, "column_names")
        and "id" in rows.column_names  # type: ignore[attr-defined]
    ):
        source_ids = rows["id"]  # type: ignore[index]
        indices = sorted(range(len(source_ids)), key=source_ids.__getitem__)
        return ((index, rows[index]) for index in indices)  # type: ignore[index]
    return sorted(
        indexed_rows,
        key=lambda item: _source_example_id(
            scenario,
            source_split,
            item[1],
            len(_source_images(scenario, item[1])),
        ),
    )


def _source_images(
    scenario: str,
    row: Mapping[str, Any],
) -> list[Any]:
    if scenario == "tulu":
        return []
    if scenario == "cosyn-circuit":
        image = row.get("image")
        if image is None:
            raise PreparationError("CoSyn row has no image")
        return [image]
    images = row.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise PreparationError("Docmatix images must be a list")
    return list(images)


def _source_example_id(
    scenario: str,
    source_split: str,
    row: Mapping[str, Any],
    image_count: int,
) -> str:
    if scenario in {"tulu", "cosyn-circuit"}:
        return _required_string(row, "id")
    payload = json.dumps(
        {
            "image_count": image_count,
            "source_split": source_split,
            "texts": row.get("texts"),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"docmatix-{source_split}-{digest}"


def _normalize_rows(
    scenario: str,
    row: Mapping[str, Any],
    *,
    example_id: str,
    image_paths: Sequence[str],
) -> list[dict[str, Any]]:
    if scenario == "tulu":
        return [normalize_tulu(row)]
    if scenario == "cosyn-circuit":
        return normalize_cosyn(row, image_path=image_paths[0])
    return [
        normalize_docmatix(
            row,
            example_id=example_id,
            image_paths=image_paths,
        )
    ]


def _expected_image_range(scenario: str) -> tuple[int, int]:
    if scenario == "tulu":
        return (0, 0)
    if scenario == "cosyn-circuit":
        return (1, 1)
    return (2, DEFAULT_MAX_IMAGES)


def _save_images(
    images: Sequence[Any],
    *,
    example_id: str,
    split: str,
    work_dir: Path,
    final_dir: Path,
    image_size: int,
    max_image_pixels: int,
) -> tuple[list[str], list[str]]:
    runtime_paths = []
    final_paths = []
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", example_id).strip("._-")
    slug = (slug or "example")[:48]
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:12]
    for image_index, image in enumerate(images):
        filename = f"{slug}-{digest}-{image_index:02d}.png"
        relative = Path("images") / split / filename
        runtime_path = work_dir / relative
        final_path = final_dir / relative
        _write_png(
            image,
            runtime_path,
            image_size=image_size,
            max_image_pixels=max_image_pixels,
        )
        runtime_paths.append(str(runtime_path.resolve()))
        final_paths.append(str(final_path.resolve()))
    return runtime_paths, final_paths


def _write_png(
    value: Any,
    output_path: Path,
    *,
    image_size: int,
    max_image_pixels: int,
) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise PreparationError(
            "Pillow is required to prepare image datasets"
        ) from error

    if isinstance(value, Image.Image):
        _validate_image_dimensions(value, max_image_pixels)
        image = value.copy()
    elif isinstance(value, Mapping) and value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as loaded:
            _validate_image_dimensions(loaded, max_image_pixels)
            image = loaded.copy()
    elif isinstance(value, Mapping) and value.get("path"):
        with Image.open(value["path"]) as loaded:
            _validate_image_dimensions(loaded, max_image_pixels)
            image = loaded.copy()
    elif isinstance(value, (str, os.PathLike)):
        with Image.open(value) as loaded:
            _validate_image_dimensions(loaded, max_image_pixels)
            image = loaded.copy()
    else:
        raise PreparationError(
            f"unsupported image value: {type(value).__name__}"
        )

    try:
        image = image.convert("RGB")
        if image_size:
            image = ImageOps.pad(
                image,
                (image_size, image_size),
                method=Image.Resampling.BILINEAR,
                color=(0, 0, 0),
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=False)
    finally:
        image.close()


def _validate_image_dimensions(image: Any, max_image_pixels: int) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise PreparationError("image dimensions must be positive")
    if width * height > max_image_pixels:
        raise PreparationError(
            f"image has {width * height} pixels; "
            f"max_image_pixels={max_image_pixels}"
        )


def _split_is_full(
    stats: SplitStats,
    max_examples_per_split: int | None,
) -> bool:
    return (
        max_examples_per_split is not None
        and stats.examples >= max_examples_per_split
    )


def _ensure_required_outputs(
    stats: Mapping[str, SplitStats],
    scenario: str,
    validation_fraction: float,
) -> None:
    required = ["train"]
    if scenario == "cosyn-circuit" or validation_fraction > 0:
        required.append("validation")
    if scenario == "docmatix":
        required.append("test")
    empty = [split for split in required if stats[split].examples == 0]
    if empty:
        raise PreparationError(
            "preparation produced empty required splits: " + ", ".join(empty)
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_content_sha256(image_dir: Path) -> str | None:
    image_paths = sorted(path for path in image_dir.glob("*.png") if path.is_file())
    if not image_paths:
        return None

    digest = hashlib.sha256()
    for image_path in image_paths:
        digest.update(image_path.name.encode("utf-8"))
        digest.update(b"\0")
        with image_path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _selected_ids_sha256(example_ids: Sequence[str]) -> str:
    """Hash sorted JSON-encoded IDs so source row order is irrelevant."""

    digest = hashlib.sha256()
    for example_id in sorted(example_ids):
        digest.update(
            json.dumps(
                example_id,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    unsafe_roots = {
        Path(resolved.anchor),
        Path.home().resolve(),
        RECIPE_ROOT,
        REPOSITORY_ROOT,
    }
    if resolved in unsafe_roots:
        raise PreparationError(f"refusing unsafe output root: {resolved}")
    return resolved


def _validate_output_target(path: Path) -> None:
    if path == Path(path.anchor):
        raise PreparationError("refusing to use a filesystem root as output")
    if path.name not in {spec.output_name for spec in SOURCES.values()}:
        raise PreparationError(f"unexpected output directory name: {path.name}")


def _validate_owned_output(
    path: Path,
    *,
    scenario: str,
    output_name: str,
) -> None:
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(
            f"refusing to overwrite unowned output directory: {path}"
        ) from error
    if not isinstance(manifest, Mapping) or (
        manifest.get("scenario") != scenario
        or manifest.get("output_name") != output_name
    ):
        raise PreparationError(
            f"refusing to overwrite output with a mismatched manifest: {path}"
        )


def _load_source_splits(
    spec: SourceSpec,
) -> dict[str, Iterable[Mapping[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise PreparationError(
            "Install the recipe environment before downloading datasets"
        ) from error

    result = {}
    for split in spec.source_splits:
        kwargs: dict[str, Any] = {
            "path": spec.dataset_id,
            "split": split,
            "revision": spec.revision,
        }
        if spec.config_name is not None:
            kwargs["name"] = spec.config_name
        result[split] = load_dataset(**kwargs)
    return result


def _verify_source_counts(
    spec: SourceSpec,
    source_splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    for split, expected in spec.expected_rows.items():
        rows = source_splits.get(split)
        if rows is None:
            raise PreparationError(f"source did not provide split {split!r}")
        try:
            actual = len(rows)  # type: ignore[arg-type]
        except TypeError as error:
            raise PreparationError(
                f"cannot preflight the row count for split {split!r}"
            ) from error
        if actual != expected:
            raise PreparationError(
                f"source split {split!r} has {actual} rows; expected "
                f"{expected} "
                f"at revision {spec.revision}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        choices=(*SOURCES, "all"),
        help="dataset track to prepare",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="parent directory for prepared dataset directories",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="hash split for Tulu and Docmatix train rows",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        help="cap each output split for a smoke run",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="pad image inputs to this square size; 0 preserves size",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=DEFAULT_MAX_IMAGE_PIXELS,
    )
    parser.add_argument("--max-text-chars", type=int)
    parser.add_argument(
        "--tokenizer-model",
        help="optional processor id for exact multimodal token counts",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=DEFAULT_TOKENIZER_REVISION,
        help="processor revision; defaults to the pinned model revision",
    )
    parser.add_argument(
        "--tokenizer-local-files-only",
        action="store_true",
        help="do not access the network when loading the processor",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="fail if the optional processor count exceeds this value",
    )
    parser.add_argument(
        "--drop-over-max-tokens",
        action="store_true",
        help="record and skip examples above --max-tokens instead of failing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing prepared directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_tokens is not None and not args.tokenizer_model:
        raise PreparationError("--max-tokens requires --tokenizer-model")
    if args.tokenizer_local_files_only and not args.tokenizer_model:
        raise PreparationError(
            "--tokenizer-local-files-only requires --tokenizer-model"
        )
    length_counter = None
    if args.tokenizer_model:
        length_counter = build_processor_length_counter(
            args.tokenizer_model,
            revision=args.tokenizer_revision,
            local_files_only=args.tokenizer_local_files_only,
        )

    scenarios = tuple(SOURCES) if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        output = prepare_dataset(
            scenario,
            output_root=args.output_root,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            max_examples_per_split=args.max_examples_per_split,
            image_size=args.image_size,
            max_images=args.max_images,
            max_image_pixels=args.max_image_pixels,
            max_text_chars=args.max_text_chars,
            length_counter=length_counter,
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            tokenizer_local_files_only=args.tokenizer_local_files_only,
            max_tokens=args.max_tokens,
            drop_over_max_tokens=args.drop_over_max_tokens,
            overwrite=args.overwrite,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
