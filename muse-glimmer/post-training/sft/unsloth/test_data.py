# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for data building, prompt rendering, the loss mask and scoring.

The rendering and loss-mask tests download the Muse Glimmer processor and
config files (no weights). Run with ``pytest test_data.py``.
"""

from __future__ import annotations

import pytest
from PIL import Image

import data


def row(
    row_id: str, answers: list[str], explanation: str = "Because of the wiring."
) -> dict:
    return {
        "id": row_id,
        "image": Image.new("RGB", (800, 400), "white"),
        "qa_pairs": {
            "question": [f"Question {i}?" for i in range(len(answers))],
            "explanation": [explanation] * len(answers),
            "answer": answers,
        },
    }


# --- Building examples -------------------------------------------------------


def test_one_example_per_question_and_cap_stops_mid_diagram():
    rows = [row("a", ["1", "2", "3"]), row("b", ["4", "5"])]
    examples = data.build_examples(rows, limit=4)
    assert [ex["id"] for ex in examples] == ["a-qa-00", "a-qa-01", "a-qa-02", "b-qa-00"]
    assert [ex["image_id"] for ex in examples] == ["a", "a", "a", "b"]
    assert examples[0]["image"].size == (data.IMAGE_SIZE, data.IMAGE_SIZE)
    assert len(data.build_examples(rows)) == 5


def test_qa_pairs_accepts_both_layouts():
    columns = {"question": [" Q? "], "explanation": [""], "answer": [" A "]}
    records = [{"question": " Q? ", "explanation": None, "answer": " A "}]
    assert data.qa_pairs(columns) == data.qa_pairs(records) == [("Q?", "", "A")]


def test_target_is_explanation_then_answer():
    assert (
        data.target_text("Two transistors.", "Two") == "Two transistors.\n\nAnswer: Two"
    )
    assert data.target_text("", "Two") == "Two"
    assert data.target_text("Two", "Two") == "Two"


def test_conversation_format():
    example = data.build_examples([row("a", ["Two"])])[0]
    conversation = data.to_conversation(example)
    assert conversation["images"] == [example["image"]]
    system, user, assistant = conversation["messages"]
    assert system["content"] == [{"type": "text", "text": data.SYSTEM_PROMPT}]
    assert user["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Question 0?"},
    ]
    assert assistant["content"] == [{"type": "text", "text": example["target"]}]


# --- Rendering and loss mask -------------------------------------------------


@pytest.fixture(scope="module")
def train_module():
    return pytest.importorskip("train")  # imports Unsloth, which runs on CPU too


@pytest.fixture(scope="module")
def processor(train_module):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(train_module.RecipeArguments.tokenizer_name)


@pytest.fixture(scope="module")
def collator(train_module, processor):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    # unsloth.trainer.UnslothVisionDataCollator extends this class with video
    # checks only; unlike the subclass, it can be built without a GPU.
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    # The collator reads the model's config and dtype, so an empty model on the
    # meta device is enough: no weights are downloaded or allocated.
    config = AutoConfig.from_pretrained(train_module.RecipeArguments.tokenizer_name)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    return UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=1024,
        resize="max",
        train_on_responses_only=True,
        instruction_part=train_module.INSTRUCTION_PART,
        response_part=train_module.RESPONSE_PART,
    )


def test_prompt_has_low_reasoning_and_no_date(processor):
    example = data.build_examples([row("a", ["Two"])])[0]
    prompt = processor.apply_chat_template(
        data.prompt_messages(example), add_generation_prompt=True, tokenize=False
    )
    assert prompt.count("Reasoning strength") == 1
    assert "Reasoning strength: low." in prompt
    assert "Current date" not in prompt
    assert prompt.endswith("<|start|>assistant")


def test_loss_covers_only_the_recipient_answer_and_end_of_turn(collator, processor):
    examples = data.build_examples([row("a", ["Two", "R1, R2"])])
    batch = collator([data.to_conversation(ex) for ex in examples])
    for labels, example in zip(batch["labels"], examples):
        trained = processor.tokenizer.decode(labels[labels != -100])
        assert trained == f" to=user<|message|>{example['target']}<|eot|>"


def test_rows_that_would_be_truncated_are_rejected(train_module, collator):
    rows = [data.to_conversation(ex) for ex in data.build_examples([row("a", ["Two"])])]
    (length,) = train_module.sequence_lengths(collator, rows, max_length=1024)
    with pytest.raises(ValueError, match="would be truncated"):
        train_module.sequence_lengths(collator, rows, max_length=length)


def test_images_keep_their_letterboxed_size(collator, processor):
    example = data.build_examples([row("a", ["Two"])])[0]
    batch = collator([data.to_conversation(example)])
    reference = processor(
        images=[[example["image"]]], text=["<|patch|>"], return_tensors="pt"
    )
    assert batch["image_grid_thw"].tolist() == reference["image_grid_thw"].tolist()
    image_tokens = (batch["input_ids"] == processor.image_token_id).sum().item()
    assert image_tokens == reference["image_grid_thw"].prod().item() // 4


# --- Reading and scoring generations -----------------------------------------


def test_extract_user_message_skips_reasoning():
    direct = " to=user<|message|>It has two.\n\nAnswer: Two<|eot|>"
    reasoned = " to=self<|message|>Count: 3? no, 2.<|eom|><|start|>assistant to=user<|message|>Answer: 2<|eot|>"
    assert data.extract_user_message(direct) == "It has two.\n\nAnswer: Two"
    assert data.extract_user_message(reasoned) == "Answer: 2"
    assert data.extract_user_message(" to=self<|message|>Answer: 3<|eom|>") is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Explanation.\n\nAnswer: R1, R2", "R1, R2"),
        ("**Answer:** Two.", "Two"),
        ("Answer: first\nMore text\nanswer: second", "second"),
        ("The answer is two.", None),
        (None, None),
    ],
)
def test_extract_short_answer(message, expected):
    assert data.extract_short_answer(message) == expected


def test_normalization_and_metrics():
    assert data.normalize("The Two resistors!") == "2 resistors"
    assert data.exact_match("two", "2") == 1.0
    assert data.exact_match(None, "2") == 0.0
    assert data.token_f1("bias current", "Biasing current") == pytest.approx(0.5)
    assert data.token_f1("", "2") == 0.0
    assert data.contains_answer("C1, the filter capacitor", "C1") == 1.0
    assert data.contains_answer("R1 only", "R1 and R2") == 0.0
    assert data.contains_answer(None, "C1") == 0.0


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Two", True),
        ("12V", True),
        ("V_reg", True),
        ("R1, R2", True),
        ("Notch filter", False),
        ("Store electrical energy", False),
        ("Three outputs and two inputs", False),
    ],
)
def test_read_off_subset(answer, expected):
    assert data.is_read_off(answer) is expected


def test_score_counts_a_cap_hit_as_wrong():
    example = data.build_examples([row("a", ["Two"])])[0]
    generated = " to=user<|message|>Because of the wiring.\n\nAnswer: two<|eot|>"
    assert data.score(example, generated, False, 12)["exact_match"] == 1.0
    assert data.score(example, generated, False, 12)["response_f1"] == 1.0
    capped = data.score(example, generated, True, 1024)
    assert capped["exact_match"] == capped["answered"] == capped["response_f1"] == 0.0


def test_summary_and_bootstrap():
    def record(image_id: str, index: int, value: float, read_off: bool) -> dict:
        return {
            "id": f"{image_id}-{index}",
            "image_id": image_id,
            "read_off": read_off,
            "hit_cap": False,
            "generated_tokens": 10,
            **{key: value for key in data.METRICS},
        }

    base = [record(f"img{i}", j, 0.0, j == 0) for i in range(10) for j in range(3)]
    tuned = [record(f"img{i}", j, 1.0, j == 0) for i in range(10) for j in range(3)]
    summary = data.summarize(base)
    assert summary["num_questions"] == 30 and summary["num_images"] == 10
    assert summary["read_off"]["num_questions"] == 10
    assert data.paired_bootstrap(base, tuned, "exact_match") == {
        "delta": 1.0,
        "ci95": [1.0, 1.0],
    }
    assert data.paired_bootstrap(base, base, "exact_match") == {
        "delta": 0.0,
        "ci95": [0.0, 0.0],
    }
