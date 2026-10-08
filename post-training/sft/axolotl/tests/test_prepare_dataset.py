# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
prepare_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_dataset
SPEC.loader.exec_module(prepare_dataset)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tulu_rows(count: int) -> list[dict]:
    return [
        {
            "id": f"tulu-{index:04d}",
            "prompt": f"Question {index}",
            "messages": [
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ],
            "constraints": ["format:test"],
        }
        for index in range(count)
    ]


def _cosyn_row(example_id: str, color: tuple[int, int, int]) -> dict:
    return {
        "id": example_id,
        "image": Image.new("RGB", (8, 4), color),
        "qa_pairs": {
            "question": ["What is shown?", "Which color is used?"],
            "explanation": [
                "The drawing contains a resistor.",
                "The main line is red.",
            ],
            "answer": ["A resistor", "Red"],
        },
        "metadata": {
            "figure_type": "Circuit",
            "persona": "Learner",
            "topic": "A small circuit",
        },
        "data": "{}",
        "code": "draw()",
    }


def _docmatix_row(index: int, image_count: int = 2) -> dict:
    images = []
    for image_index in range(image_count):
        images.append(
            Image.new(
                "RGB",
                (6, 4),
                (index % 255, image_index * 40, 10),
            )
        )
    return {
        "images": images,
        "texts": [
            {
                "user": f"Question for document {index}?",
                "assistant": f"Answer for document {index}.",
                "source": f"PDFA key: fixture-{index}",
            },
            {
                "user": "What is on the next page?",
                "assistant": "A second fixture section.",
                "source": f"PDFA key: fixture-{index}",
            },
        ],
    }


def test_source_revisions_are_pinned() -> None:
    for source in prepare_dataset.SOURCES.values():
        assert len(source.revision) == 40
        int(source.revision, 16)
        assert source.expected_rows


def test_all_training_configs_explicitly_enable_epoch_evaluation() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    config_names = (
        "muse-glimmer-30b-qlora.yaml",
        "muse-glimmer-30b-lora-fsdp2.yaml",
        "muse-glimmer-30b-full-fsdp2.yaml",
    )

    for config_name in config_names:
        top_level = {}
        for line in (config_dir / config_name).read_text().splitlines():
            if not line or line[0].isspace() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition(":")
            if separator:
                top_level[key] = value.split("#", 1)[0].strip()
        assert top_level.get("eval_strategy") == "epoch", config_name


def test_tulu_preserves_message_content_whitespace() -> None:
    source = {
        "id": "tulu-whitespace",
        "messages": [
            {"role": "user", "content": "  Keep leading spaces\n"},
            {"role": "assistant", "content": "\tKeep indentation\n\n"},
        ],
    }

    normalized = prepare_dataset.normalize_tulu(source)

    assert normalized["messages"] == source["messages"]


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_tulu_rejects_empty_message_content(content: str) -> None:
    source = {
        "id": "tulu-empty",
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Answer"},
        ],
    }

    with pytest.raises(prepare_dataset.PreparationError, match="must not be empty"):
        prepare_dataset.normalize_tulu(source)


def test_hash_split_is_stable_and_order_independent() -> None:
    ids = [f"sample-{index}" for index in range(200)]
    first = {
        item: prepare_dataset.deterministic_validation_assignment(
            item,
            seed=42,
            validation_fraction=0.2,
        )
        for item in ids
    }
    second = {
        item: prepare_dataset.deterministic_validation_assignment(
            item,
            seed=42,
            validation_fraction=0.2,
        )
        for item in reversed(ids)
    }
    assert first == second
    assert any(first.values())
    assert not all(first.values())


def test_selected_id_hash_is_order_independent() -> None:
    first = prepare_dataset._selected_ids_sha256(["gamma", "alpha", "beta"])
    second = prepare_dataset._selected_ids_sha256(["beta", "gamma", "alpha"])
    assert first == second


def test_capped_selection_is_source_order_independent(tmp_path: Path) -> None:
    rows = _tulu_rows(200)
    outputs = []
    for name, ordered_rows in (("forward", rows), ("reverse", list(reversed(rows)))):
        output = prepare_dataset.prepare_dataset(
            "tulu",
            output_root=tmp_path / name,
            source_splits={"train": ordered_rows},
            verify_source_counts=False,
            validation_fraction=0.2,
            max_examples_per_split=12,
        )
        outputs.append(output)

    for split in ("train", "validation"):
        first_rows = _read_jsonl(outputs[0] / f"{split}.jsonl")
        second_rows = _read_jsonl(outputs[1] / f"{split}.jsonl")
        assert first_rows == second_rows
        assert len(first_rows) == 12


def test_docmatix_id_does_not_depend_on_row_order() -> None:
    rows = [_docmatix_row(7, 3), _docmatix_row(8), _docmatix_row(9)]

    def ids_by_source(ordered_rows: list[dict]) -> dict[str, str]:
        return {
            row["texts"][0]["source"]: prepare_dataset._source_example_id(
                "docmatix",
                "train",
                row,
                len(row["images"]),
            )
            for row in ordered_rows
        }

    assert ids_by_source(rows) == ids_by_source(list(reversed(rows)))


def test_tulu_preparation_is_deterministic_without_network(
    tmp_path: Path,
) -> None:
    source = {"train": _tulu_rows(200)}
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = prepare_dataset.prepare_dataset(
        "tulu",
        output_root=first_root,
        source_splits=source,
        verify_source_counts=False,
        validation_fraction=0.2,
    )
    second = prepare_dataset.prepare_dataset(
        "tulu",
        output_root=second_root,
        source_splits=source,
        verify_source_counts=False,
        validation_fraction=0.2,
    )

    first_train = _read_jsonl(first / "train.jsonl")
    first_validation = _read_jsonl(first / "validation.jsonl")
    second_train = _read_jsonl(second / "train.jsonl")
    second_validation = _read_jsonl(second / "validation.jsonl")
    assert first_train == second_train
    assert first_validation == second_validation
    assert {row["id"] for row in first_train}.isdisjoint(
        row["id"] for row in first_validation
    )
    assert len(first_train) + len(first_validation) == 200
    assert set(first_train[0]) == {"id", "messages"}

    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["output_name"] == "tulu"
    for split in ("train", "validation"):
        split_path = first / f"{split}.jsonl"
        split_manifest = manifest["splits"][split]
        assert (
            split_manifest["jsonl_sha256"]
            == hashlib.sha256(split_path.read_bytes()).hexdigest()
        )
        rows = _read_jsonl(split_path)
        assert split_manifest["selected_ids_sha256"] == (
            prepare_dataset._selected_ids_sha256([row["id"] for row in rows])
        )
        assert len(split_manifest["selected_ids_sha256"]) == 64
        assert split_manifest["image_content_sha256"] is None


def test_cosyn_emits_one_dialogue_per_qa_and_reuses_image(
    tmp_path: Path,
) -> None:
    source = {
        "train": [_cosyn_row("circuit-train", (255, 0, 0))],
        "validation": [_cosyn_row("circuit-validation", (0, 255, 0))],
    }
    output = prepare_dataset.prepare_dataset(
        "cosyn-circuit",
        output_root=tmp_path,
        source_splits=source,
        verify_source_counts=False,
        image_size=16,
    )

    rows = _read_jsonl(output / "train.jsonl")
    assert [row["id"] for row in rows] == [
        "circuit-train-qa-00",
        "circuit-train-qa-01",
    ]
    assert all(
        [message["role"] for message in row["messages"]]
        == ["user", "assistant"]
        for row in rows
    )
    first_content = rows[0]["messages"][0]["content"]
    second_content = rows[1]["messages"][0]["content"]
    assert [part["type"] for part in first_content] == ["image", "text"]
    assert second_content[-1]["text"] == "Which color is used?"
    assert "Answer: A resistor" in rows[0]["messages"][1]["content"]
    assert "reasoning_content" not in rows[0]["messages"][1]

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preparation"]["cosyn_explanation_policy"] == {
        "storage": "visible_assistant_content",
        "reasoning_content": False,
        "answer_prefix": "Answer: ",
        "omit_empty_or_duplicate_explanation": True,
    }

    image_path = Path(first_content[0]["path"])
    assert second_content[0]["path"] == str(image_path)
    assert image_path.is_file()
    assert len(list((output / "images" / "train").glob("*.png"))) == 1
    with Image.open(image_path) as image:
        assert image.size == (16, 16)

    recorded_hash = manifest["splits"]["train"]["image_content_sha256"]
    assert recorded_hash == prepare_dataset._image_content_sha256(
        output / "images" / "train"
    )
    Image.new("RGB", (16, 16), (0, 0, 255)).save(image_path)
    assert recorded_hash != prepare_dataset._image_content_sha256(
        output / "images" / "train"
    )


def test_docmatix_keeps_every_image_inline_and_in_order(
    tmp_path: Path,
) -> None:
    train_rows = [_docmatix_row(index) for index in range(40)]
    test_row = _docmatix_row(240, image_count=3)
    output = prepare_dataset.prepare_dataset(
        "docmatix",
        output_root=tmp_path,
        source_splits={"train": train_rows, "test": [test_row]},
        verify_source_counts=False,
        validation_fraction=0.25,
        image_size=12,
    )

    train = _read_jsonl(output / "train.jsonl")
    validation = _read_jsonl(output / "validation.jsonl")
    assert train
    assert validation
    assert {row["id"] for row in train}.isdisjoint(
        row["id"] for row in validation
    )

    test = _read_jsonl(output / "test.jsonl")
    assert len(test) == 1
    content = test[0]["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image"]
    assert len(image_parts) == 3
    assert all("path" in part and "url" not in part for part in image_parts)
    assert content[-1] == {
        "type": "text",
        "text": "Question for document 240?",
    }

    colors = []
    for part in image_parts:
        with Image.open(part["path"]) as image:
            colors.append(image.getpixel((6, 6)))
    assert colors == [(240, 0, 10), (240, 40, 10), (240, 80, 10)]


def test_docmatix_filters_single_image_rows(tmp_path: Path) -> None:
    multi_rows = [_docmatix_row(index) for index in range(30)]
    rows = [_docmatix_row(999, image_count=1), *multi_rows]
    output = prepare_dataset.prepare_dataset(
        "docmatix",
        output_root=tmp_path,
        source_splits={
            "train": rows,
            "test": [_docmatix_row(500, image_count=2)],
        },
        verify_source_counts=False,
        validation_fraction=0.25,
        image_size=8,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preparation"]["filtered_rows"] == 1
    output_count = sum(
        manifest["splits"][name]["examples"] for name in ("train", "validation")
    )
    assert output_count == len(multi_rows)


def test_token_length_hook_can_reject_an_example(tmp_path: Path) -> None:
    with pytest.raises(
        prepare_dataset.PreparationError,
        match="token length 101 exceeds max_tokens=100",
    ):
        prepare_dataset.prepare_dataset(
            "tulu",
            output_root=tmp_path,
            source_splits={"train": _tulu_rows(1)},
            verify_source_counts=False,
            validation_fraction=0.0,
            length_counter=lambda messages: 101,
            max_tokens=100,
        )
    assert not (tmp_path / "tulu").exists()


def test_token_length_hook_can_drop_and_record_an_example(tmp_path: Path) -> None:
    source = _tulu_rows(2)

    def count(messages: list[dict]) -> int:
        return 101 if messages[0]["content"] == "Question 0" else 10

    output = prepare_dataset.prepare_dataset(
        "tulu",
        output_root=tmp_path,
        source_splits={"train": source},
        verify_source_counts=False,
        validation_fraction=0.0,
        length_counter=count,
        max_tokens=100,
        drop_over_max_tokens=True,
    )

    rows = _read_jsonl(output / "train.jsonl")
    assert [row["id"] for row in rows] == ["tulu-0001"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preparation"]["drop_over_max_tokens"] is True
    assert manifest["preparation"]["filtered_examples"] == {
        "over_max_tokens": 1
    }


def test_processor_loader_uses_revision_and_offline_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, messages: list[dict], **kwargs: object):
            assert messages
            assert kwargs["tokenize"] is True
            return {"input_ids": [[1, 2, 3]]}

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object):
            calls.append((model_id, kwargs))
            return FakeProcessor()

    fake_transformers = SimpleNamespace(AutoProcessor=FakeAutoProcessor)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    counter = prepare_dataset.build_processor_length_counter(
        "/models/muse-glimmer",
        revision="model-revision",
        local_files_only=True,
    )

    assert counter([{"role": "user", "content": "hello"}]) == 3
    assert calls == [
        (
            "/models/muse-glimmer",
            {
                "revision": "model-revision",
                "local_files_only": True,
            },
        )
    ]


def test_tokenizer_pin_is_recorded_in_manifest(tmp_path: Path) -> None:
    output = prepare_dataset.prepare_dataset(
        "tulu",
        output_root=tmp_path,
        source_splits={"train": _tulu_rows(1)},
        verify_source_counts=False,
        validation_fraction=0.0,
        length_counter=lambda messages: len(messages),
        tokenizer_model="/models/muse-glimmer",
        tokenizer_revision="model-revision",
        tokenizer_local_files_only=True,
    )
    preparation = json.loads((output / "manifest.json").read_text())[
        "preparation"
    ]
    assert preparation["tokenizer_model"] == "/models/muse-glimmer"
    assert preparation["tokenizer_revision"] == "model-revision"
    assert preparation["tokenizer_local_files_only"] is True


def test_repository_root_is_actual_checkout_root() -> None:
    checkout_root = Path(__file__).resolve().parents[4]
    assert prepare_dataset.REPOSITORY_ROOT == checkout_root
    assert (checkout_root / "post-training" / "sft" / "axolotl").is_dir()


@pytest.mark.parametrize(
    "unsafe_root",
    [
        Path("/"),
        Path.home(),
        prepare_dataset.RECIPE_ROOT,
        prepare_dataset.REPOSITORY_ROOT,
    ],
)
def test_unsafe_output_roots_are_refused(unsafe_root: Path) -> None:
    with pytest.raises(
        prepare_dataset.PreparationError,
        match="refusing unsafe output root",
    ):
        prepare_dataset.prepare_dataset(
            "tulu",
            output_root=unsafe_root,
            source_splits={"train": _tulu_rows(1)},
            verify_source_counts=False,
            validation_fraction=0.0,
            overwrite=True,
        )


def test_overwrite_refuses_an_unowned_directory(tmp_path: Path) -> None:
    target = tmp_path / "tulu"
    target.mkdir()
    marker = target / "user-file.txt"
    marker.write_text("keep me")

    with pytest.raises(
        prepare_dataset.PreparationError,
        match="refusing to overwrite unowned output directory",
    ):
        prepare_dataset.prepare_dataset(
            "tulu",
            output_root=tmp_path,
            source_splits={"train": _tulu_rows(1)},
            verify_source_counts=False,
            validation_fraction=0.0,
            overwrite=True,
        )
    assert marker.read_text() == "keep me"


def test_overwrite_refuses_a_mismatched_manifest(tmp_path: Path) -> None:
    target = tmp_path / "tulu"
    target.mkdir()
    manifest = target / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scenario": "docmatix",
                "output_name": "docmatix",
            }
        )
    )

    with pytest.raises(
        prepare_dataset.PreparationError,
        match="mismatched manifest",
    ):
        prepare_dataset.prepare_dataset(
            "tulu",
            output_root=tmp_path,
            source_splits={"train": _tulu_rows(1)},
            verify_source_counts=False,
            validation_fraction=0.0,
            overwrite=True,
        )
    assert manifest.is_file()


def test_overwrite_accepts_its_matching_manifest(tmp_path: Path) -> None:
    arguments = {
        "output_root": tmp_path,
        "source_splits": {"train": _tulu_rows(1)},
        "verify_source_counts": False,
        "validation_fraction": 0.0,
    }
    first = prepare_dataset.prepare_dataset("tulu", **arguments)
    first_manifest = json.loads((first / "manifest.json").read_text())

    second = prepare_dataset.prepare_dataset(
        "tulu",
        **arguments,
        overwrite=True,
    )
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest == second_manifest


def test_more_than_four_docmatix_images_fails_atomically(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        prepare_dataset.PreparationError,
        match="5 images",
    ):
        prepare_dataset.prepare_dataset(
            "docmatix",
            output_root=tmp_path,
            source_splits={
                "train": [_docmatix_row(1, image_count=5)],
                "test": [_docmatix_row(2, image_count=2)],
            },
            verify_source_counts=False,
            validation_fraction=0.0,
            image_size=8,
        )
    assert not (tmp_path / "docmatix").exists()


def test_cosyn_columnar_blank_answer_is_rejected() -> None:
    row = _cosyn_row("blank-answer", (1, 2, 3))
    row["qa_pairs"]["answer"][0] = "   "

    with pytest.raises(
        prepare_dataset.PreparationError,
        match="qa_pairs.answer must not contain empty strings",
    ):
        prepare_dataset.normalize_cosyn(row, image_path="unused.png")


def test_image_pixel_limit_is_checked(tmp_path: Path) -> None:
    with pytest.raises(
        prepare_dataset.PreparationError,
        match="image has 32 pixels; max_image_pixels=16",
    ):
        prepare_dataset.prepare_dataset(
            "cosyn-circuit",
            output_root=tmp_path,
            source_splits={
                "train": [_cosyn_row("large-train", (1, 2, 3))],
                "validation": [_cosyn_row("large-validation", (4, 5, 6))],
            },
            verify_source_counts=False,
            max_image_pixels=16,
        )
    assert not (tmp_path / "cosyn-circuit").exists()


def test_overwrite_refuses_symlink_output(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep me")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (output_root / "tulu").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        prepare_dataset.PreparationError,
        match="refusing symlink output directory",
    ):
        prepare_dataset.prepare_dataset(
            "tulu",
            output_root=output_root,
            source_splits={"train": _tulu_rows(1)},
            verify_source_counts=False,
            validation_fraction=0.0,
            overwrite=True,
        )
    assert marker.read_text() == "keep me"


def test_failed_overwrite_restores_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = {
        "output_root": tmp_path,
        "source_splits": {"train": _tulu_rows(1)},
        "verify_source_counts": False,
        "validation_fraction": 0.0,
    }
    target = prepare_dataset.prepare_dataset("tulu", **arguments)
    original_manifest = (target / "manifest.json").read_bytes()
    original_replace = Path.replace
    failed = False

    def fail_install_once(source: Path, destination: Path) -> Path:
        nonlocal failed
        if (
            not failed
            and destination == target
            and source.name.startswith(".tulu-")
            and not source.name.startswith(".tulu-backup-")
        ):
            failed = True
            raise OSError("simulated install failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_install_once)
    with pytest.raises(OSError, match="simulated install failure"):
        prepare_dataset.prepare_dataset(
            "tulu",
            **arguments,
            overwrite=True,
        )

    assert target.is_dir()
    assert (target / "manifest.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(".tulu-backup-*"))
