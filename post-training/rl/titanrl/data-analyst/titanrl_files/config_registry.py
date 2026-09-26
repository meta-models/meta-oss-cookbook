# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Muse Glimmer data-analyst example.

Modeled on ``search_r1``'s Muse Glimmer config, swapping the multi-turn search env
for a one-tool ``bash`` code-execution env over a fully offline, deterministic
dataset -- no retrieval server or dataset download needed::

    python -m torchtitan.rl.train \\
        --module torchtitan.rl.examples.glimmer_data_analyst \\
        --config rl_grpo_muse_glimmer_30b_data_analyst
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_model_registry
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.glimmer_data_analyst.data import GlimmerDataAnalystDataset
from torchtitan.rl.examples.glimmer_data_analyst.env import GlimmerDataAnalystEnv
from torchtitan.rl.examples.glimmer_data_analyst.rubric import RewardCorrectArtifact
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.model.muse_glimmer.renderer import MuseGlimmerRendererConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer


def _data_analyst_rollouter_config() -> Rollouter.Config:
    return Rollouter.Config(
        train_dataset=GlimmerDataAnalystDataset.Config(split="train", seed=42),
        validation_dataset=GlimmerDataAnalystDataset.Config(split="validation", seed=42),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardCorrectArtifact.Config(weight=1.0)],
                # truncation_reward is deliberately left unset. Rubric
                # short-circuits truncated rollouts to a fixed reward without
                # running the reward fns. Early in training most rollouts here
                # DO truncate -- the model explores, computes the right number,
                # and runs out of budget before writing the file -- so a fixed
                # truncation reward erases the within-group variance GRPO needs
                # and the batch is discarded as untrainable. Unset, the reward
                # fn grades what the rollout achieved before it was cut off.
                truncation_reward=None,
            ),
            message_env=GlimmerDataAnalystEnv.Config(),
            token_env=TokenEnv.Config(
                # Sized so a worst-case rollout fits the trainer's 4096-token
                # context: prompt (~120 tok) + 6 turns x (384 generated + <=200
                # tok of tool output) ~= 3624 < 4096. An over-budget rollout ends
                # TRUNCATED_PROMPT_TOO_LONG; at 1024 tokens per turn that was 49%
                # of rollouts, many cut off after computing the answer but before
                # writing it.
                max_rollout_tokens=4096,
                max_num_turns=6,
                step_timeout_s=120.0,
            ),
        ),
    )


def rl_grpo_muse_glimmer_30b_data_analyst() -> Controller.Config:
    """GRPO/DAPO data-analyst recipe for Muse Glimmer 30B.

    8 GPUs: 6 trainer (FSDP=3 x TP=2) + 2 generator (TP=2) -- the same split as
    ``rl_grpo_muse_glimmer_30b_search_r1``.

    * **Generator TP <= 2.** Muse Glimmer has 2 KV heads, so attention cannot be
      tensor-split further.
    * **Full activation checkpointing is required.** Adam's m/v are allocated on
      the first ``optimizer.step()``, so per-GPU memory jumps by roughly 8
      bytes/param between step 1 and step 2; ``SelectiveAC`` OOMs at step 2.

    Unlike Search-R1, this needs no retrieval server or dataset download: the env
    generates its own csv/log tasks and grades them deterministically (see
    ``data.py`` / ``rubric.py``).
    """
    model_config = muse_glimmer_model_registry("30B", attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Muse-Glimmer-30B",
        async_loop=AsyncLoopConfig(
            num_training_steps=200,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            target_offpolicy_steps=1,
            validation=ValidationConfig(num_samples=64),
        ),
        compile=None,
        rollouter=_data_analyst_rollouter_config(),
        renderer=MuseGlimmerRendererConfig(),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                # The batcher requires the microbatch token budget to be a
                # multiple of max_context_length.
                num_tokens_per_microbatch_per_dp_rank=4096,
                max_context_length=4096,
            ),
            activation_checkpoint=FullAC.Config(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=3,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,  # <= 2 KV heads
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=384,
            ),
        ),
    )


def rl_grpo_muse_glimmer_30b_data_analyst_smoke() -> Controller.Config:
    """Small smoke variant: 10 steps, 4 prompts x 4 samples, no mid-run checkpoint.

    Run this first to confirm the loop works end to end on your hardware.
    """
    config = rl_grpo_muse_glimmer_30b_data_analyst()
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=10,
        num_prompts_per_train_step=4,
        num_samples_per_prompt=4,
        validation=ValidationConfig(num_samples=16),
    )
    # Past the run's step count: a mid-run DCP save of a 30B model is heavy I/O
    # that a smoke run does not need.
    config.trainer = dataclasses.replace(
        config.trainer,
        checkpointer=dataclasses.replace(config.trainer.checkpointer, interval=1000),
    )
    return config
