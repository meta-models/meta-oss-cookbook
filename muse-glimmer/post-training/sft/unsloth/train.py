# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QLoRA supervised fine-tuning of Muse Glimmer on CoSyn circuit questions with Unsloth.

On one GPU:

    python train.py --config config.yaml

The run scores the base model on the held-out test questions, trains, scores
the trained adapter on the same questions and writes both scores to
``<output_dir>/test_results.json``. With ``--evaluate_only`` it scores only the
base model, or a saved adapter given with ``--adapter_path``.
"""

from __future__ import annotations

# Unsloth patches Transformers, TRL and PEFT, so import it first.
from unsloth import FastModel  # isort: skip
from unsloth.trainer import UnslothVisionDataCollator  # isort: skip

import importlib.metadata
import json
import os
from dataclasses import dataclass

import torch
from trl import SFTConfig, SFTTrainer, TrlParser

import data

# Muse Glimmer's chat markers. The loss covers everything after RESPONSE_PART
# up to and including <|eot|>: the recipient " to=user<|message|>", then the
# answer. Starting the loss after "to=user<|message|>" instead teaches the answer
# but not the channel, and the trained model then writes it to private "to=self".
INSTRUCTION_PART = "<|start|>user<|message|>"
RESPONSE_PART = "<|start|>assistant"


@dataclass
class RecipeArguments:
    model_name_or_path: str = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
    """Unsloth's 4-bit (NF4) upload of meta-models/Muse-Glimmer-30B."""
    tokenizer_name: str = "meta-models/Muse-Glimmer-30B"
    """Processor and chat template. Unsloth's upload ships an older template."""
    train_size: int = 2048
    validation_size: int = 256
    test_size: int | None = None
    """Held-out test questions to score. None scores all 821."""
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = False
    """Also add LoRA adapters to the vision encoder."""
    max_new_tokens: int = 1024
    generation_batch_size: int = 64
    evaluate_only: bool = False
    adapter_path: str | None = None


@torch.inference_mode()
def score_test(model, processor, examples, recipe, name, output_dir) -> dict:
    """Greedy-decode every test question in batches and score it."""
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    records = []
    for start in range(0, len(examples), recipe.generation_batch_size):
        batch = examples[start : start + recipe.generation_batch_size]
        texts = [
            processor.apply_chat_template(
                data.prompt_messages(ex), add_generation_prompt=True, tokenize=False
            )
            for ex in batch
        ]
        inputs = processor(
            images=[[ex["image"]] for ex in batch],
            text=texts,
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=recipe.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        for ex, ids in zip(batch, output[:, inputs["input_ids"].shape[1] :].tolist()):
            stop = next((i for i, token in enumerate(ids) if token in eos_ids), None)
            kept = ids if stop is None else ids[: stop + 1]
            generated = processor.tokenizer.decode(kept, skip_special_tokens=False)
            records.append(data.score(ex, generated, stop is None, len(kept)))
        print(f"[test] {name}: {len(records)}/{len(examples)}", flush=True)
    summary = data.summarize(records)
    with open(os.path.join(output_dir, f"predictions_{name}.jsonl"), "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in records)
    print(f"[test] {name}: {json.dumps(summary)}", flush=True)
    return {"summary": summary, "records": records}


def compare(base: dict, tuned: dict) -> dict:
    """Tuned-minus-base differences with diagram-level bootstrap CIs."""

    def read_off(rows: list[dict]) -> list[dict]:
        return [row for row in rows if row["read_off"]]

    keys = ("exact_match", "answer_f1", "contains_answer")
    return {
        "all": {
            key: data.paired_bootstrap(base["records"], tuned["records"], key)
            for key in keys
        },
        "read_off": {
            key: data.paired_bootstrap(
                read_off(base["records"]), read_off(tuned["records"]), key
            )
            for key in keys
        },
    }


def sequence_lengths(collator, rows: list[dict], max_length: int) -> list[int]:
    """Tokens per row (image, prompt and answer). Stops if a row would be truncated.

    The collator truncates rows to max_length, which would cut image tokens or
    the answer and its <|eot|>.
    """
    lengths = [int(collator([row])["attention_mask"].sum()) for row in rows]
    too_long = sum(length >= max_length for length in lengths)
    if too_long:
        raise ValueError(
            f"{too_long} of {len(rows)} examples reach max_length={max_length} and "
            "would be truncated; raise max_length above the longest example"
        )
    return lengths


def training_summary(trainer: SFTTrainer, tokens_per_epoch: int) -> dict:
    """Token throughput and allocator peaks for the training loop.

    Tokens are every token in the training sequences (image, prompt and
    answer); with one example per micro-batch there is no padding.
    """
    history = trainer.state.log_history
    train_log = next(row for row in reversed(history) if "train_runtime" in row)
    eval_seconds = sum(row["eval_runtime"] for row in history if "eval_runtime" in row)
    train_seconds = train_log["train_runtime"] - eval_seconds
    num_tokens = tokens_per_epoch * trainer.state.epoch
    return {
        "gpus": 1,
        "gpu": torch.cuda.get_device_name(),
        "optimizer_steps": trainer.state.global_step,
        "train_tokens": num_tokens,
        "train_runtime_s": train_log["train_runtime"],
        "validation_runtime_s": eval_seconds,
        "seconds_per_update": train_seconds / trainer.state.global_step,
        "tokens_per_s": num_tokens / train_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def versions() -> dict:
    packages = ("unsloth", "unsloth_zoo", "trl", "transformers", "peft", "torch")
    return {name: importlib.metadata.version(name) for name in packages}


def main() -> None:
    recipe, args = TrlParser((RecipeArguments, SFTConfig)).parse_args_and_config()
    if recipe.adapter_path and not recipe.evaluate_only:
        raise ValueError(
            "--adapter_path scores a saved adapter; use it with --evaluate_only"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    test = data.load_split("validation", limit=recipe.test_size)

    if recipe.evaluate_only:
        # Given an adapter directory, Unsloth loads its base model and attaches it.
        model, processor = FastModel.from_pretrained(
            model_name=recipe.adapter_path or recipe.model_name_or_path,
            tokenizer_name=recipe.tokenizer_name,
            max_seq_length=args.max_length,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
        name = "adapter" if recipe.adapter_path else "base"
        result = score_test(model, processor, test, recipe, name, args.output_dir)
        with open(os.path.join(args.output_dir, f"test_results_{name}.json"), "w") as f:
            json.dump({name: result["summary"], "versions": versions()}, f, indent=2)
        return

    model, processor = FastModel.from_pretrained(
        model_name=recipe.model_name_or_path,
        tokenizer_name=recipe.tokenizer_name,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        random_state=args.seed,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=recipe.finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        bias="none",
        random_state=args.seed,
    )
    model.print_trainable_parameters()

    collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=args.max_length,
        resize="max",  # data.py already letterboxes every image to 512x512
        train_on_responses_only=True,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )
    train_rows = [
        data.to_conversation(ex) for ex in data.load_split("train", recipe.train_size)
    ]
    validation_rows = [
        data.to_conversation(ex)
        for ex in data.load_split("validation", recipe.validation_size)
    ]
    train_tokens = sum(sequence_lengths(collator, train_rows, args.max_length))
    sequence_lengths(collator, validation_rows, args.max_length)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=validation_rows,
        processing_class=processor.tokenizer,
        data_collator=collator,
    )

    # A disabled adapter is the unchanged quantized base model.
    FastModel.for_inference(model)
    with model.disable_adapter():
        base = score_test(model, processor, test, recipe, "base", args.output_dir)
    trainer.log(
        {f"test/base_{k}": v for k, v in base["summary"].items() if k in data.METRICS}
    )

    FastModel.for_training(model)
    torch.cuda.empty_cache()  # measure training memory without the test pass's cache
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    training = training_summary(trainer, train_tokens)
    trainer.save_model(args.output_dir)
    # The processor and chat template too, so the adapter directory reloads
    # with the official template even without tokenizer_name.
    processor.save_pretrained(args.output_dir)
    trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

    FastModel.for_inference(model)
    tuned = score_test(model, processor, test, recipe, "tuned", args.output_dir)
    trainer.log(
        {f"test/tuned_{k}": v for k, v in tuned["summary"].items() if k in data.METRICS}
    )

    results = {
        "base": base["summary"],
        "tuned": tuned["summary"],
        "tuned_minus_base": compare(base, tuned),
        "training": training,
        "seed": args.seed,
        "versions": versions(),
    }
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] {json.dumps(results)}", flush=True)


if __name__ == "__main__":
    main()
