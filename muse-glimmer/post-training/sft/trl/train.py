# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QLoRA supervised fine-tuning of Muse Glimmer on CoSyn circuit questions with TRL.

On one 8x H100 node:

    accelerate launch --num_processes 8 train.py --config config.yaml

The run scores the base model on all held-out test questions, trains, scores
the trained adapter on the same questions and writes both scores to
``<output_dir>/test_results.json``. With ``--evaluate_only`` it scores only the
base model, or a saved adapter given with ``--adapter_path``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, TrlParser

import data


@dataclass
class RecipeArguments:
    model_name_or_path: str = "meta-models/Muse-Glimmer-30B"
    train_size: int = 2048
    validation_size: int = 256
    test_size: int | None = None
    """Held-out test questions to score. None scores all 821."""
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    """Language-model projections only. The vision tower reuses the q/k/v names."""
    max_new_tokens: int = 1024
    generation_batch_size: int = 16
    evaluate_only: bool = False
    adapter_path: str | None = None


def quantization_config() -> BitsAndBytesConfig:
    """NF4 language model. The vision modules and output head stay in BF16."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
        llm_int8_skip_modules=[
            "model.vision_tower",
            "model.vision_adapter",
            "model.vision_projection",
            "lm_head",
        ],
    )


@torch.inference_mode()
def score_test(model, processor, examples, recipe, state, name, output_dir) -> dict:
    """Greedy-decode every test question, sharded across processes, and score it."""
    model.eval()
    eos_ids = set(model.generation_config.eos_token_id)
    records = []
    with state.split_between_processes(examples) as shard:
        for start in range(0, len(shard), recipe.generation_batch_size):
            batch = shard[start : start + recipe.generation_batch_size]
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
            for ex, ids in zip(
                batch, output[:, inputs["input_ids"].shape[1] :].tolist()
            ):
                stop = next(
                    (i for i, token in enumerate(ids) if token in eos_ids), None
                )
                kept = ids if stop is None else ids[: stop + 1]
                generated = processor.tokenizer.decode(kept, skip_special_tokens=False)
                records.append(data.score(ex, generated, stop is None, len(kept)))
    records = gather_object(records)
    summary = data.summarize(records)
    if state.is_main_process:
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


def training_summary(trainer: SFTTrainer, state: PartialState) -> dict:
    """Token throughput and per-GPU allocator peaks for the training loop."""
    history = trainer.state.log_history
    train_log = next(row for row in reversed(history) if "train_runtime" in row)
    num_tokens = max(row.get("num_tokens", 0) for row in history)
    eval_seconds = sum(row["eval_runtime"] for row in history if "eval_runtime" in row)
    train_seconds = train_log["train_runtime"] - eval_seconds
    peaks = gather_object(
        [
            {
                "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        ]
    )
    return {
        "gpus": state.num_processes,
        "optimizer_steps": trainer.state.global_step,
        "train_tokens": num_tokens,
        "train_runtime_s": train_log["train_runtime"],
        "validation_runtime_s": eval_seconds,
        "tokens_per_s": num_tokens / train_seconds,
        "tokens_per_s_per_gpu": num_tokens / train_seconds / state.num_processes,
        "peak_allocated_gib_per_gpu": max(p["allocated_gib"] for p in peaks),
        "peak_reserved_gib_per_gpu": max(p["reserved_gib"] for p in peaks),
    }


def main() -> None:
    recipe, args = TrlParser((RecipeArguments, SFTConfig)).parse_args_and_config()
    if recipe.adapter_path and not recipe.evaluate_only:
        raise ValueError(
            "--adapter_path scores a saved adapter; use it with --evaluate_only"
        )
    state = PartialState()
    os.makedirs(args.output_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(recipe.model_name_or_path)
    test = data.load_split("validation", limit=recipe.test_size)

    if recipe.evaluate_only:
        model = AutoModelForImageTextToText.from_pretrained(
            recipe.model_name_or_path,
            dtype=torch.bfloat16,
            quantization_config=quantization_config(),
            device_map={"": state.device},
        )
        name = "base"
        if recipe.adapter_path:
            # Keep the BF16 adapter dtype TRL trains QLoRA with; PEFT upcasts to FP32 by default.
            model = PeftModel.from_pretrained(
                model, recipe.adapter_path, autocast_adapter_dtype=False
            )
            name = "adapter"
        result = score_test(
            model, processor, test, recipe, state, name, args.output_dir
        )
        if state.is_main_process:
            with open(
                os.path.join(args.output_dir, f"test_results_{name}.json"), "w"
            ) as f:
                json.dump({name: result["summary"]}, f, indent=2)
        return

    def conversations(split: str, limit: int) -> Dataset:
        return Dataset.from_list(
            [data.to_conversation(ex) for ex in data.load_split(split, limit)]
        )

    trainer = SFTTrainer(
        model=recipe.model_name_or_path,
        args=args,
        train_dataset=conversations("train", recipe.train_size),
        eval_dataset=conversations("validation", recipe.validation_size),
        processing_class=processor,
        quantization_config=quantization_config(),
        peft_config=LoraConfig(
            r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=recipe.lora_target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    if state.is_main_process:
        trainer.model.print_trainable_parameters()

    # A disabled adapter is the unchanged quantized base model.
    with trainer.model.disable_adapter():
        base = score_test(
            trainer.model, processor, test, recipe, state, "base", args.output_dir
        )
    trainer.log(
        {f"test/base_{k}": v for k, v in base["summary"].items() if k in data.METRICS}
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    training = training_summary(trainer, state)
    trainer.save_model(args.output_dir)
    if state.is_main_process:
        trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

    tuned = score_test(
        trainer.model, processor, test, recipe, state, "tuned", args.output_dir
    )
    trainer.log(
        {f"test/tuned_{k}": v for k, v in tuned["summary"].items() if k in data.METRICS}
    )

    if state.is_main_process:
        results = {
            "base": base["summary"],
            "tuned": tuned["summary"],
            "tuned_minus_base": compare(base, tuned),
            "training": training,
            "seed": args.seed,
        }
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[results] {json.dumps(results)}", flush=True)


if __name__ == "__main__":
    main()
