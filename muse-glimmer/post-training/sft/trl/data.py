# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CoSyn circuit data, prompts and answer scoring for the TRL SFT recipe.

Every example is one question about one circuit diagram. The assistant target
is the dataset's explanation followed by ``Answer: <short answer>``, exactly as
in the Axolotl recipe. Training and evaluation share one system prompt, so the
model is scored on the same prompt it was trained on.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter, defaultdict
from typing import Any

from PIL import Image, ImageOps

DATASET = "allenai/CoSyn-400K"
CONFIG = "circuit"
IMAGE_SIZE = 512

# A system message that names the reasoning strength replaces the template's
# default system block, which would otherwise insert today's date and
# "Reasoning strength: high.".
SYSTEM_PROMPT = (
    "You answer questions about circuit diagrams. Explain briefly, then give "
    'the final answer on its own last line as "Answer: <short answer>".\n\n'
    "Reasoning strength: low."
)

NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        )
        + (
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
        )
        + ("eighteen", "nineteen", "twenty")
    )
}


# ---------------------------------------------------------------------------
# Building examples
# ---------------------------------------------------------------------------


def qa_pairs(value: Any) -> list[tuple[str, str, str]]:
    """Return (question, explanation, answer) triples from a CoSyn row."""
    if isinstance(value, dict):
        triples = zip(
            value["question"], value["explanation"], value["answer"], strict=True
        )
    else:
        triples = (
            (item["question"], item["explanation"], item["answer"]) for item in value
        )
    return [(q.strip(), (e or "").strip(), a.strip()) for q, e, a in triples]


def target_text(explanation: str, answer: str) -> str:
    """The assistant target: the explanation, then the short answer."""
    if not explanation or explanation == answer:
        return answer
    return f"{explanation}\n\nAnswer: {answer}"


def letterbox(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize to a ``size`` x ``size`` square, padding with black."""
    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    )


def build_examples(
    rows, limit: int | None = None, image_size: int = IMAGE_SIZE
) -> list[dict]:
    """Emit one example per question, in row order, until ``limit`` examples.

    ``rows`` must already be sorted by ``id``. Like the Axolotl preparation, the
    cap can stop partway through one diagram's questions.
    """
    examples = []
    for row in rows:
        if limit is not None and len(examples) >= limit:
            break
        image = letterbox(row["image"], image_size)
        for index, (question, explanation, answer) in enumerate(
            qa_pairs(row["qa_pairs"])
        ):
            if limit is not None and len(examples) >= limit:
                break
            examples.append(
                {
                    "id": f"{row['id']}-qa-{index:02d}",
                    "image_id": row["id"],
                    "image": image,
                    "question": question,
                    "answer": answer,
                    "target": target_text(explanation, answer),
                }
            )
    return examples


def load_split(
    split: str, limit: int | None = None, image_size: int = IMAGE_SIZE
) -> list[dict]:
    """Load a CoSyn circuit split from the Hub, ordered by diagram ``id``."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET, CONFIG, split=split)
    ids = dataset["id"]
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return build_examples(
        (dataset[i] for i in order), limit=limit, image_size=image_size
    )


def prompt_messages(example: dict) -> list[dict]:
    """System and user turns. The image placeholder is filled from ``images``."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["question"]},
            ],
        },
    ]


def to_conversation(example: dict) -> dict:
    """A training row in TRL's conversational vision format.

    Every message uses a list of typed parts so the column has one Arrow type.
    """
    return {
        "images": [example["image"]],
        "messages": prompt_messages(example)
        + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["target"]}],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Reading and scoring generations
# ---------------------------------------------------------------------------

_USER_MESSAGE = re.compile(
    r"<\|start\|>assistant to=user<\|message\|>(.*?)(?=<\|eot\|>|<\|eom\|>|<\|start\|>|$)",
    re.DOTALL,
)
_ANSWER_LINE = re.compile(
    r"^[\s*_#>]*answer[\s*_]*:[\s*_]*(.+?)[\s*_]*$", re.IGNORECASE | re.MULTILINE
)


def extract_user_message(generated: str) -> str | None:
    """The last message addressed to the user, or None.

    ``generated`` is the decoded continuation of a prompt that ends with
    ``<|start|>assistant``, special tokens included. Reasoning written to the
    ``to=self`` channel is never scored.
    """
    matches = _USER_MESSAGE.findall("<|start|>assistant" + generated)
    return matches[-1].strip() if matches else None


def extract_short_answer(message: str | None) -> str | None:
    """The text after the last ``Answer:`` line, or None."""
    if not message:
        return None
    matches = _ANSWER_LINE.findall(message)
    return matches[-1].strip().rstrip(".").strip() if matches else None


def normalize(text: str) -> str:
    """SQuAD-style normalization, with number words written as digits."""
    text = text.lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    tokens = [NUMBER_WORDS.get(token, token) for token in text.split()]
    return " ".join(token for token in tokens if token not in {"a", "an", "the"})


def exact_match(prediction: str | None, reference: str) -> float:
    return float(
        prediction is not None and normalize(prediction) == normalize(reference)
    )


def token_f1(prediction: str | None, reference: str) -> float:
    if prediction is None:
        return 0.0
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str | None, reference: str) -> float:
    """1 if every reference word appears in the prediction, however long it is.

    This ignores phrasing and length, so it separates a correct but wordy
    answer from a wrong one.
    """
    if prediction is None:
        return 0.0
    words = set(normalize(prediction).split())
    return float(all(word in words for word in normalize(reference).split()))


def is_read_off(answer: str) -> bool:
    """Counts, values and component labels that can be read from the diagram."""
    normalized = normalize(answer)
    return len(normalized.split()) <= 3 and (
        any(ch.isdigit() for ch in normalized) or "_" in answer
    )


def score(example: dict, generated: str, hit_cap: bool, num_tokens: int) -> dict:
    """Score one generation. A response cut off by the token cap scores zero."""
    message = None if hit_cap else extract_user_message(generated)
    short = extract_short_answer(message)
    return {
        "id": example["id"],
        "image_id": example["image_id"],
        "question": example["question"],
        "answer": example["answer"],
        "read_off": is_read_off(example["answer"]),
        "generated": generated,
        "generated_tokens": num_tokens,
        "hit_cap": hit_cap,
        "short_answer": short,
        "answered": float(short is not None),
        "exact_match": exact_match(short, example["answer"]),
        "answer_f1": token_f1(short, example["answer"]),
        "contains_answer": contains_answer(short, example["answer"]),
        "response_f1": token_f1(message, example["target"]),
    }


METRICS = ("exact_match", "answer_f1", "contains_answer", "answered", "response_f1")


def summarize(records: list[dict]) -> dict:
    """Mean metrics over all questions and over the read-off subset."""

    def means(rows: list[dict]) -> dict:
        return (
            {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}
            if rows
            else {}
        )

    read_off = [row for row in records if row["read_off"]]
    return {
        "num_questions": len(records),
        "num_images": len({row["image_id"] for row in records}),
        "cap_hits": sum(row["hit_cap"] for row in records),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in records)
        / len(records),
        **means(records),
        "read_off": {"num_questions": len(read_off), **means(read_off)},
    }


def paired_bootstrap(
    base: list[dict], tuned: list[dict], key: str, samples: int = 10_000, seed: int = 0
) -> dict:
    """Mean tuned-minus-base difference with a 95% CI, resampling whole diagrams.

    Questions about one diagram are correlated, so diagrams, not questions, are
    the resampling unit.
    """
    tuned_by_id = {row["id"]: row for row in tuned}
    groups: dict[str, list[float]] = defaultdict(list)
    for row in base:
        groups[row["image_id"]].append(tuned_by_id[row["id"]][key] - row[key])
    clusters = [(sum(deltas), len(deltas)) for deltas in groups.values()]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        estimates.append(sum(s for s, _ in draw) / sum(n for _, n in draw))
    estimates.sort()
    total, count = sum(s for s, _ in clusters), sum(n for _, n in clusters)
    return {
        "delta": total / count,
        "ci95": [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]],
    }
