# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for data building, prompt rendering, the loss mask and scoring.

The rendering test downloads the Muse Glimmer processor files (no weights).
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


def test_target_matches_the_axolotl_recipe():
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
    from datasets import Dataset

    assert Dataset.from_list([conversation]).num_rows == 1


# --- Rendering and loss mask -------------------------------------------------


@pytest.fixture(scope="module")
def processor():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B")


def test_prompt_has_low_reasoning_and_no_date(processor):
    example = data.build_examples([row("a", ["Two"])])[0]
    prompt = processor.apply_chat_template(
        data.prompt_messages(example), add_generation_prompt=True, tokenize=False
    )
    assert prompt.count("Reasoning strength") == 1
    assert "Reasoning strength: low." in prompt
    assert "Current date" not in prompt
    assert prompt.endswith("<|start|>assistant")


def test_loss_covers_only_the_answer_and_end_of_turn(processor):
    from trl.chat_template_utils import get_training_chat_template
    from trl.data_utils import prepare_multimodal_messages

    example = data.build_examples([row("a", ["Two"])])[0]
    conversation = data.to_conversation(example)
    messages = prepare_multimodal_messages(
        conversation["messages"], images=conversation["images"]
    )
    output = processor.apply_chat_template(
        [messages],
        chat_template=get_training_chat_template(
            processor, require_prefix_preservation=False
        ),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        processor_kwargs={"padding": True, "add_special_tokens": False},
    )
    ids = output["input_ids"][0]
    trained = processor.tokenizer.decode(ids[output["assistant_masks"][0].bool()])
    assert trained == f" to=user<|message|>{example['target']}<|eot|>"


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
