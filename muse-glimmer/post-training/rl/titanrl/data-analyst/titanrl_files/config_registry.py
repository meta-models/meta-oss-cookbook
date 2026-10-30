# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer data-analyst configurations for TorchTitan's RL stack.

Select the module containing the factory, rather than the example package::

    python -m torchtitan.rl.train \\
        --module torchtitan.rl.examples.glimmer_data_analyst.config_registry \\
        --config rl_grpo_muse_glimmer_30b_data_analyst_smoke
"""

from __future__ import annotations

import dataclasses

from renderers import CustomRendererConfig
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_glimmer import build_model_config
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.glimmer_data_analyst.data import GlimmerDataAnalystDataset
from torchtitan.rl.examples.glimmer_data_analyst.env import GlimmerDataAnalystEnv
from torchtitan.rl.examples.glimmer_data_analyst.rubric import RewardCorrectArtifact
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer.losses import DAPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_CONTEXT_LENGTH = 4096
_MAX_COMPLETION_TOKENS = 384
_SAMPLES_PER_PROMPT = 8


def _data_analyst_rollouter_config() -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=GlimmerDataAnalystDataset.Config(split="train", seed=42)
        ),
        validation_dataset=GlimmerDataAnalystDataset.Config(
            split="validation", seed=42
        ),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardCorrectArtifact.Config(weight=1.0)],
                # Grade progress even when a rollout exhausts its budget. A
                # fixed truncation score would erase the variance GRPO needs.
                truncation_reward=None,
            ),
            message_env=GlimmerDataAnalystEnv.Config(),
            token_env=TokenEnv.Config(
                # The limit applies to the prompt, not prompt + completion.
                # Reserve a full generation so every sample fits the trainer.
                max_rollout_tokens=_CONTEXT_LENGTH - _MAX_COMPLETION_TOKENS,
                max_num_turns=6,
                step_timeout_s=120.0,
            ),
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
    )


def rl_grpo_muse_glimmer_30b_data_analyst() -> Controller.Config:
    """GRPO/DAPO over offline CSV/log tasks, using all eight GPUs.

    Six trainer GPUs use FSDP=3 x TP=2; two generator GPUs use TP=2.
    Glimmer has two KV heads, so generator TP cannot be increased beyond two.
    Full activation rematerialization provides headroom for Adam's first
    optimizer-state allocation.
    """
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=3,
        tensor_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=2,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "30B",
        seq_len=_CONTEXT_LENGTH,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        # Glimmer's soft-capped head is not supported by the FP32-head converter.
        fp32_lm_head=False,
    )
    # Preserve eager named regions explicitly instead of inheriting model
    # defaults that can change as upstream adds compiler optimizations.
    trainer_model.local_compile = LocalCompileConfig(regions=[])
    generator_model.local_compile = LocalCompileConfig(regions=[])
    return Controller.Config(
        hf_assets_path="torchtitan/rl/example_checkpoint/Muse-Glimmer-30B",
        async_loop=AsyncLoopConfig(
            # Bound these easy templates instead of requesting many updates
            # after reward variance is exhausted. Final save/validation and
            # exit status still need checking; this is not an early-stop hook.
            # Longer training needs a harder task distribution, not a disabled
            # GRPO filter or an arbitrarily large update budget.
            num_training_steps=15,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=_SAMPLES_PER_PROMPT,
            target_offpolicy_steps=1,
            validation=ValidationConfig(steps=64),
        ),
        rollouter=_data_analyst_rollouter_config(),
        renderer=from_renderers(
            CustomRendererConfig(
                import_path="torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
            )
        ),
        # The upstream default keeps only each group's reward extremes. Keep
        # every sibling (and error) so plotted proportions describe the run.
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(
                k=_SAMPLES_PER_PROMPT, keep_errors=True
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2, decay_type="linear", min_lr_factor=1.0
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=_CONTEXT_LENGTH,
                max_context_length=_CONTEXT_LENGTH,
            ),
            activation_remat=FullActivationRematConfig(),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(trainer_model),
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="NONE")),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=_MAX_COMPLETION_TOKENS,
            ),
        ),
    )


def rl_grpo_muse_glimmer_30b_data_analyst_smoke() -> Controller.Config:
    """Ten optimizer steps, four prompts x four samples, on the same GPU split."""
    config = rl_grpo_muse_glimmer_30b_data_analyst()
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=10,
        num_prompts_per_train_step=4,
        num_samples_per_prompt=4,
        validation=ValidationConfig(steps=16),
    )
    # Avoid heavy mid-run checkpoint I/O, while preserving the final save.
    config.trainer = dataclasses.replace(
        config.trainer,
        checkpointer=dataclasses.replace(config.trainer.checkpointer, interval=1000),
    )
    return config
