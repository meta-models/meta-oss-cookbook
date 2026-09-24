# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.glimmer_data_analyst.data import (
    GlimmerDataAnalystDataset,
)
from torchtitan.experiments.rl.examples.glimmer_data_analyst.env import (
    GlimmerDataAnalystEnv,
)
from torchtitan.experiments.rl.examples.glimmer_data_analyst.rubric import (
    RewardCorrectArtifact,
)
from torchtitan.experiments.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.experiments.rl.rubrics import Rubric


class GlimmerDataAnalystWorker(RolloutWorker):
    """The data-analyst env + rubric. Pure config; all methods inherited."""

    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardCorrectArtifact.Config(weight=1.0)],
                # truncation_reward is deliberately NOT set. The framework's
                # Rubric short-circuits truncated rollouts to a fixed reward and
                # never runs the reward fns -- which would force every truncated
                # rollout to an identical score. In this env most early-training
                # rollouts DO truncate (the model explores, computes the right
                # number, and runs out of turn/token budget before writing the
                # file), so short-circuiting them erases all within-group reward
                # variance and GRPO discards the batch as untrainable. Leaving it
                # None lets RewardCorrectArtifact grade what the rollout actually
                # achieved before it was cut off.
                truncation_reward=None,
            )
        )
        message_env: GlimmerDataAnalystEnv.Config = field(
            default_factory=GlimmerDataAnalystEnv.Config
        )
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                # Budgeted so a WORST-CASE rollout still fits inside the
                # trainer's 4096-token context: prompt (~120 tok) + 6 turns x
                # (384 generated + <=200 tok of tool output) ~= 3624 < 4096.
                # Getting this wrong is not a soft failure -- an over-budget
                # rollout ends as TRUNCATED_PROMPT_TOO_LONG, and an earlier
                # 1024-token-per-turn setting made that 49% of all rollouts,
                # many cut off AFTER computing the right answer but BEFORE
                # writing it out.
                max_rollout_tokens=4096,
                max_num_turns=6,
                step_timeout_s=120.0,
            )
        )


class GlimmerDataAnalystRollouter(Rollouter):
    """Data-analyst rollouter: the synthetic csv/log dataset, the one-tool ``bash``
    env, and the artifact-correctness rubric wired together. All behavior is
    inherited from ``Rollouter``; this only supplies the default configs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: GlimmerDataAnalystDataset.Config = field(
            default_factory=lambda: GlimmerDataAnalystDataset.Config(
                split="train", seed=42
            )
        )
        validation_dataset: GlimmerDataAnalystDataset.Config = field(
            default_factory=lambda: GlimmerDataAnalystDataset.Config(
                split="validation", seed=42
            )
        )
        worker: GlimmerDataAnalystWorker.Config = field(
            default_factory=GlimmerDataAnalystWorker.Config
        )
