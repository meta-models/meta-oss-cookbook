# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO post-training of Muse Glimmer on data-analyst tasks with TRL.

Start a vLLM server on two GPUs, then train on a third (one trainer process):

    CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 \
        vllm serve meta-models/Muse-Glimmer-30B --tensor-parallel-size 2 \
        --max-model-len 8192 --weight-transfer-config '{"backend": "nccl"}' \
        --logprobs-mode processed_logprobs --max-logprobs -1
    CUDA_VISIBLE_DEVICES=2 VLLM_HOST_IP=127.0.0.1 python train.py --config config.yaml

The run scores the held-out test split before and after training and writes
the scores to ``<output_dir>/test_results.json``. Pass ``--evaluate_only`` (and
optionally ``--adapter_path``) to score only the base model or a saved adapter; that
writes ``test_results_base.json`` or ``test_results_adapter.json`` instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# Evaluation batches mix short and long rollouts; avoid allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from peft import LoraConfig, PeftModel  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.integrations import WandbCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer, TrlParser  # noqa: E402

from env import DataAnalystEnv, accuracy, answered, make_dataset  # noqa: E402


@dataclass
class RecipeArguments:
    model_name_or_path: str = "meta-models/Muse-Glimmer-30B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    train_size: int = 4096
    validation_size: int = 96
    test_size: int = 256
    task_seed: int = 42
    evaluate_only: bool = False
    """Score the test split without training: the base model, or ``adapter_path``."""
    adapter_path: str | None = None
    """A saved adapter to score; requires ``evaluate_only``."""


def evaluate_test(trainer: GRPOTrainer, dataset, name: str) -> dict:
    """Score the test split without mixing it into the logged validation curve."""
    loggers = [
        cb for cb in trainer.callback_handler.callbacks if isinstance(cb, WandbCallback)
    ]
    for cb in loggers:
        trainer.remove_callback(cb)
    log_completions, trainer.log_completions = trainer.log_completions, False
    try:
        metrics = trainer.evaluate(dataset, metric_key_prefix="test")
    finally:
        trainer.log_completions = log_completions
        for cb in loggers:
            trainer.add_callback(cb)
    keys = {
        "accuracy": "eval_rewards/accuracy/mean",
        "answered": "eval_rewards/answered/mean",
        "reward": "eval_reward",
        "clipped_ratio": "eval_completions/clipped_ratio",
        "mean_completion_tokens": "eval_completions/mean_length",
        "tool_calls_per_rollout": "eval_tools/call_frequency",
    }
    result = {key: metrics.get(source) for key, source in keys.items()}
    result.update(
        num_tasks=len(dataset),
        samples_per_task=trainer.num_generations_eval,
        step=trainer.state.global_step,
        seed=trainer.args.seed,
    )
    if trainer.accelerator.is_main_process:
        print(f"[test] {name}: {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    recipe, args = TrlParser((RecipeArguments, GRPOConfig)).parse_args_and_config()
    if recipe.adapter_path and not recipe.evaluate_only:
        raise ValueError(
            "--adapter_path loads a frozen adapter; use it with --evaluate_only"
        )

    if recipe.adapter_path:
        base = AutoModelForImageTextToText.from_pretrained(
            recipe.model_name_or_path, dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, recipe.adapter_path)
        peft_config = None
    else:
        model = recipe.model_name_or_path
        peft_config = LoraConfig(
            r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=0.0,
            target_modules=recipe.lora_target_modules,
            exclude_modules=r".*vision.*",  # train the text decoder only
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=[accuracy, answered],
        train_dataset=make_dataset("train", recipe.train_size, recipe.task_seed),
        eval_dataset=make_dataset(
            "validation", recipe.validation_size, recipe.task_seed
        ),
        peft_config=peft_config,
        environment_factory=DataAnalystEnv,
    )
    test = make_dataset("test", recipe.test_size, recipe.task_seed)

    results_file = "test_results.json"
    if recipe.evaluate_only:
        name = "adapter" if recipe.adapter_path else "base"
        results = {name: evaluate_test(trainer, test, name)}
        results_file = (
            f"test_results_{name}.json"  # never overwrite a training run's scores
        )
    else:
        results = {"base": evaluate_test(trainer, test, "base")}
        trainer.train()
        trainer.save_model()
        trainer.save_state()
        results["trained"] = evaluate_test(trainer, test, "trained")

    if trainer.accelerator.is_main_process:
        peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        results["memory"] = {"trainer_peak_allocated_gib": peak_gib}
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, results_file), "w") as f:
            json.dump(results, f, indent=2)
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.run.summary.update(
                    {
                        f"test/{run}/{k}": v
                        for run, r in results.items()
                        for k, v in r.items()
                    }
                )


if __name__ == "__main__":
    main()
