# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic, graded artifact+correctness reward for the data-analyst rollout.

Reads the ``env_rewards`` the env attaches to each step (see
``GlimmerDataAnalystEnv._grade``) rather than touching the filesystem itself, so
grading is correct regardless of where or when the rubric runs relative to the env.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.rl.rollout.types import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


def _last_graded_env_rewards(rollout: Rollout) -> dict[str, float]:
    """The env_rewards from the last turn that has any -- the env grades on every
    step, so this is the most recent (and most complete) file state observed."""
    for turn in reversed(rollout.turns):
        if turn.env_rewards:
            return turn.env_rewards
    return {}


class RewardCorrectArtifact(RewardFn):
    """Graded artifact-correctness reward.

    The full score requires the required output file to hold the right number.
    The intermediate credits below it exist for a specific reason: with a bare
    0/1 correctness signal, every rollout in a GRPO group scores identically
    until the model can solve the task outright, so the group has zero reward
    variance, the batch is dropped as untrainable, and the policy never gets a
    gradient to climb out on. The intermediate signals separate the behaviors
    this recipe is trying to shape -- running a command successfully at all,
    producing the named artifact, making that artifact well-formed -- so a
    partially-competent rollout outscores a flailing one.

    Correctness still dominates by construction: every partial credit combined
    stays below ``score``, so writing an empty or wrong-valued file can never
    beat getting the answer right.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        score: float = 1.0
        """Reward for the required file holding the correct value."""

        file_parses_credit: float = 0.25
        """Credit for a well-formed ``{"answer": <number>}`` file whose value is
        wrong: the agentic mechanics all worked, only the analysis missed."""

        file_exists_credit: float = 0.15
        """Credit for producing the named output file at all, even malformed --
        directly counter to the "never writes the artifact" failure mode."""

        ran_ok_credit: float = 0.05
        """Credit for getting at least one command to execute successfully --
        separates a rollout that ran code from one that only emitted broken code."""

        repeated_failing_command_penalty: float = 0.05
        """Deducted per repeat of a command that already failed (the thrash loop:
        re-sending identical broken code instead of trying something else).
        Capped so it can never drive a correct rollout below an incorrect one."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._score = config.score
        self._file_parses_credit = config.file_parses_credit
        self._file_exists_credit = config.file_exists_credit
        self._ran_ok_credit = config.ran_ok_credit
        self._repeated_failing_command_penalty = (
            config.repeated_failing_command_penalty
        )

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        env_rewards = _last_graded_env_rewards(rollout)
        if not env_rewards:
            return 0.0

        if env_rewards.get("correct", 0.0) >= 1.0:
            return self._score

        # Not correct: accumulate the partial credits the rollout did earn.
        reward = 0.0
        if env_rewards.get("ran_ok", 0.0) >= 1.0:
            reward += self._ran_ok_credit
        if env_rewards.get("file_exists", 0.0) >= 1.0:
            reward += self._file_exists_credit
        if env_rewards.get("file_parses", 0.0) >= 1.0:
            reward += self._file_parses_credit

        # Anti-thrash, bounded so partial credit can still be positive.
        penalty = self._repeated_failing_command_penalty * env_rewards.get(
            "repeated_failing_commands", 0.0
        )
        return max(0.0, reward - penalty)


__all__ = ["RewardCorrectArtifact"]
