# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the data-analyst example.

Modeled directly on ``search_r1``'s Muse Glimmer config, swapping the multi-turn
search env for a one-tool ``bash`` code-execution env with a fully offline,
deterministic dataset -- no retrieval server or external dataset download
needed::

    python -m torchtitan.experiments.rl.train \\
        --module glimmer_data_analyst \\
        --config rl_grpo_muse_glimmer_30b_data_analyst
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.glimmer_data_analyst.rollouter import (
    GlimmerDataAnalystRollouter,
)
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.muse_glimmer.renderer import (
    MuseGlimmerRendererConfig,
)
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_model_registry
from torchtitan.models.muse_glimmer.state_dict_adapter import (
    MuseGlimmerStateDictAdapter,
)


def rl_grpo_muse_glimmer_30b_data_analyst() -> Controller.Config:
    """GRPO/DAPO data-analyst recipe for Muse Glimmer 30B.

    8 GPUs: 6 trainer (FSDP=3 x TP=2) + 2 generator (TP=2) -- the same split
    ``rl_grpo_muse_glimmer_30b_search_r1`` uses, verified end to end on this
    hardware.

    * **Generator TP <= 2.** Muse Glimmer has 2 KV heads, so attention cannot be
      tensor-split further.
    * **Full activation checkpointing is required.** Adam's m/v are allocated on
      the first ``optimizer.step()``, so per-GPU memory jumps by roughly 8
      bytes/param between step 1 and step 2; the default ``SelectiveAC`` OOMs at
      step 2.
    * **Pin TorchTitan to a commit before #4535.** That change (2026-09-13) made
      the fused gate-up projection the default and registered a state-dict save
      hook that materializes a full unfused copy of the projection. RL calls
      ``state_dict()`` on every weight sync, and at 30B the extra allocation is
      enough to exhaust a 95 GiB card -- it breaks upstream's OWN Muse Glimmer
      RL config, not just this one. Verified: the stock
      ``rl_grpo_muse_glimmer_30b_search_r1`` OOMs in ``_split_w13_on_save`` on
      current main, and runs clean at ``8108e201a`` (the commit before #4535),
      where the trainer peaks at ~88 GB instead of ~97 GB.

    Unlike Search-R1, this recipe needs no retrieval server or external dataset
    download: the data-analyst env generates its own csv/log tasks and grades them
    deterministically (see ``data.py`` / ``rubric.py``).
    """
    model_spec = muse_glimmer_model_registry("30B", attn_backend="varlen")
    model_spec = dataclasses.replace(
        model_spec, state_dict_adapter=MuseGlimmerStateDictAdapter
    )

    return Controller.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Muse-Glimmer-30B",
        async_loop=AsyncLoopConfig(
            num_training_steps=200,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            target_offpolicy_steps=1,
            validation=ValidationConfig(num_samples=64),
        ),
        compile=CompileConfig(enable=False),
        rollouter=GlimmerDataAnalystRollouter.Config(),
        renderer=MuseGlimmerRendererConfig(),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(
                # Context must cover the rollouter's max_rollout_tokens (4096):
                # at 1536 roughly half of all rollouts died as
                # "truncated_prompt_too_long", several after already computing
                # the right answer. The batcher requires the microbatch token
                # budget to be a multiple of max_context_length, so these move
                # together.
                num_tokens_per_microbatch_per_dp_rank=4096,
                max_context_length=4096,
                # NOTE: enable_cpu_offload=True would free the ~40 GB/GPU of
                # Adam state, but this RL stack initializes only a NCCL process
                # group, so FSDP's CPU offload path dies with "No backend type
                # associated with device type cpu". Left off until the stack
                # initializes a gloo PG for CPU tensors.
            ),
            ac_config=FullAC.Config(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=3,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
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
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,  # <= 2 KV heads
            ),
            cudagraph=VLLMCudagraphConfig(enable=False),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=384,
            ),
        ),
    )


def rl_grpo_muse_glimmer_30b_data_analyst_smoke() -> Controller.Config:
    """Tiny smoke variant: a handful of steps, small batch, no mid-run checkpoint
    save. Use this first to confirm the loop runs end to end before the full
    ``rl_grpo_muse_glimmer_30b_data_analyst`` run, and as the cookbook's "run it on
    your own machine" entry point.
    """
    config = rl_grpo_muse_glimmer_30b_data_analyst()
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=10,
        num_prompts_per_train_step=4,
        num_samples_per_prompt=4,
        validation=ValidationConfig(num_samples=16),
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        # Past the smoke run's step count: a mid-run DCP save of a 56 GB checkpoint
        # is heavy I/O that has destabilized this class of host before (see
        # W0_RESULTS.md); the smoke run only needs the initial HF load to work.
        checkpoint=dataclasses.replace(config.trainer.checkpoint, interval=1000),
    )
    return config
