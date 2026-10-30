# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
import shlex
import sys

import numpy as np
import pytest
from renderers.base import ParsedToolCall
from torchtitan.rl.examples.glimmer_data_analyst.data import (
    GlimmerDataAnalystDataset,
)
from torchtitan.rl.examples.glimmer_data_analyst.env import (
    MAX_OUTPUT_CHARS,
    GlimmerDataAnalystEnv,
)
from torchtitan.rl.examples.glimmer_data_analyst.rubric import (
    RewardCorrectArtifact,
)
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.types import (
    Rollout,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)


def test_dataset_generates_verifiable_samples() -> None:
    dataset = GlimmerDataAnalystDataset.Config(split="train", seed=7).build()
    samples = [next(dataset) for _ in range(40)]
    assert len(samples) == 40
    for sample in samples:
        assert sample.input_content
        assert sample.prompt
        assert sample.golden_value == sample.golden_value  # not NaN
        assert abs(sample.golden_value) < float("inf")


def test_train_and_validation_never_collide() -> None:
    train = GlimmerDataAnalystDataset.Config(split="train", seed=7).build()
    validation = GlimmerDataAnalystDataset.Config(split="validation", seed=7).build()
    train_ids = {next(train).task_id for _ in range(200)}
    validation_ids = {next(validation).task_id for _ in range(200)}
    assert train_ids.isdisjoint(validation_ids)


def test_dataset_is_deterministic_for_a_given_seed() -> None:
    a = GlimmerDataAnalystDataset.Config(split="train", seed=123).build()
    b = GlimmerDataAnalystDataset.Config(split="train", seed=123).build()
    samples_a = [next(a) for _ in range(10)]
    samples_b = [next(b) for _ in range(10)]
    assert [s.task_id for s in samples_a] == [s.task_id for s in samples_b]
    assert [s.golden_value for s in samples_a] == [s.golden_value for s in samples_b]


def _turn_with_env_rewards(env_rewards: dict[str, float]) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_prefix_len=0,
        prompt_delta_token_ids=np.asarray([1, 2, 3], dtype=np.int32),
        completion_token_ids=np.asarray([4, 5], dtype=np.int32),
        completion_logprobs=np.asarray([0.0, 0.0], dtype=np.float32),
        env_rewards=env_rewards,
    )


def _rollout_with_turns(*turns: RolloutTurn) -> Rollout:
    return Rollout(
        group_id=0, rollout_id=0, status=RolloutStatus.COMPLETED, turns=list(turns)
    )


@pytest.mark.parametrize(
    ("env_rewards", "expected"),
    [
        # Correct answer wins outright, regardless of what else happened.
        ({"correct": 1.0, "file_parses": 1.0, "file_exists": 1.0, "ran_ok": 1.0}, 1.0),
        # Well-formed file, wrong number: ran_ok + file_exists + file_parses.
        (
            {"correct": 0.0, "file_parses": 1.0, "file_exists": 1.0, "ran_ok": 1.0},
            0.45,
        ),
        # File written but malformed: ran_ok + file_exists.
        (
            {"correct": 0.0, "file_parses": 0.0, "file_exists": 1.0, "ran_ok": 1.0},
            0.20,
        ),
        # Ran a command successfully but never wrote the artifact.
        (
            {"correct": 0.0, "file_parses": 0.0, "file_exists": 0.0, "ran_ok": 1.0},
            0.05,
        ),
        # Emitted only broken code: nothing earned.
        (
            {"correct": 0.0, "file_parses": 0.0, "file_exists": 0.0, "ran_ok": 0.0},
            0.0,
        ),
        ({}, 0.0),
    ],
)
def test_reward_correct_artifact(
    env_rewards: dict[str, float], expected: float
) -> None:
    reward_fn = RewardCorrectArtifact.Config().build()
    rollout = _rollout_with_turns(_turn_with_env_rewards(env_rewards))
    score = asyncio.run(reward_fn(rollout, env_input=None))
    assert score == pytest.approx(expected)


def test_reward_is_graded_so_a_group_has_variance() -> None:
    """The point of the intermediate credits: rollouts that all FAIL the task
    must still score differently, or GRPO sees no variance and drops the batch."""
    reward_fn = RewardCorrectArtifact.Config().build()
    tiers = [
        {"correct": 0.0, "file_parses": 1.0, "file_exists": 1.0, "ran_ok": 1.0},
        {"correct": 0.0, "file_parses": 0.0, "file_exists": 1.0, "ran_ok": 1.0},
        {"correct": 0.0, "file_parses": 0.0, "file_exists": 0.0, "ran_ok": 1.0},
        {"correct": 0.0, "file_parses": 0.0, "file_exists": 0.0, "ran_ok": 0.0},
    ]
    scores = [
        asyncio.run(reward_fn(_rollout_with_turns(_turn_with_env_rewards(t)), None))
        for t in tiers
    ]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores), "each tier must score distinctly"


def test_partial_credit_never_outscores_correct() -> None:
    """Anti-reward-hacking: the best possible wrong rollout must lose to a
    correct one, so the model can't farm partial credit instead of solving."""
    reward_fn = RewardCorrectArtifact.Config().build()
    best_wrong = asyncio.run(
        reward_fn(
            _rollout_with_turns(
                _turn_with_env_rewards(
                    {
                        "correct": 0.0,
                        "file_parses": 1.0,
                        "file_exists": 1.0,
                        "ran_ok": 1.0,
                    }
                )
            ),
            None,
        )
    )
    correct = asyncio.run(
        reward_fn(_rollout_with_turns(_turn_with_env_rewards({"correct": 1.0})), None)
    )
    assert best_wrong < correct


def test_thrash_penalty_reduces_reward_but_not_below_zero() -> None:
    reward_fn = RewardCorrectArtifact.Config().build()
    base = {"correct": 0.0, "file_parses": 0.0, "file_exists": 0.0, "ran_ok": 1.0}
    thrashed = {**base, "repeated_failing_commands": 10.0}
    base_score = asyncio.run(
        reward_fn(_rollout_with_turns(_turn_with_env_rewards(base)), None)
    )
    thrashed_score = asyncio.run(
        reward_fn(_rollout_with_turns(_turn_with_env_rewards(thrashed)), None)
    )
    assert thrashed_score < base_score
    assert thrashed_score >= 0.0


def test_reward_reads_the_last_graded_turn() -> None:
    # A later turn with no signal must not erase an earlier turn's correct grade.
    reward_fn = RewardCorrectArtifact.Config().build()
    rollout = _rollout_with_turns(
        _turn_with_env_rewards({"file_exists": 1.0, "correct": 1.0}),
        _turn_with_env_rewards({}),
    )
    score = asyncio.run(reward_fn(rollout, env_input=None))
    assert score == pytest.approx(1.0)


def test_empty_rollout_scores_zero() -> None:
    reward_fn = RewardCorrectArtifact.Config().build()
    rollout = _rollout_with_turns()
    score = asyncio.run(reward_fn(rollout, env_input=None))
    assert score == pytest.approx(0.0)


def _completion_message(tool_calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _bash_call(arguments: dict) -> ParsedToolCall:
    return ParsedToolCall(raw="", name="bash", arguments=arguments, id="bash-call")


def test_env_grades_a_correct_bash_workflow() -> None:
    dataset = GlimmerDataAnalystDataset.Config(split="train", seed=1).build()
    sample = next(dataset)

    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    asyncio.run(env.init())

    write_answer_cmd = (
        f"python3 -c \"import json; json.dump({{'answer': {sample.golden_value}}}, "
        f"open('answer.json', 'w'))\""
    )
    step_output = asyncio.run(
        env.step(_completion_message([_bash_call({"command": write_answer_cmd})]))
    )
    assert step_output.env_rewards["correct"] == 1.0
    assert step_output.env_rewards["file_exists"] == 1.0

    final_output = asyncio.run(env.step(_completion_message(None)))
    assert final_output.done is True
    assert final_output.env_rewards["correct"] == 1.0
    asyncio.run(env.close())


def test_env_grades_a_missing_output_file_as_incorrect() -> None:
    dataset = GlimmerDataAnalystDataset.Config(split="train", seed=1).build()
    sample = next(dataset)

    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    asyncio.run(env.init())
    step_output = asyncio.run(env.step(_completion_message(None)))
    assert step_output.done is True
    assert step_output.env_rewards["correct"] == 0.0
    assert step_output.env_rewards["file_exists"] == 0.0
    asyncio.run(env.close())


def test_env_close_removes_the_workspace() -> None:
    dataset = GlimmerDataAnalystDataset.Config(split="train", seed=1).build()
    sample = next(dataset)

    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    asyncio.run(env.init())
    workdir = env._workdir  # test-only reach-in to assert cleanup
    assert workdir.exists()
    asyncio.run(env.close())
    assert not workdir.exists()


def test_loader_replays_unacknowledged_samples() -> None:
    config = IterableRLDataLoader.Config(
        dataset=GlimmerDataAnalystDataset.Config(seed=42)
    )
    loader = config.build()
    first = [next(loader) for _ in range(3)]
    loader.acknowledge([first[0][0]])
    state = loader.state_dict()
    expected_next = next(loader)
    resumed = config.build()
    resumed.load_state_dict(state)
    assert [next(resumed), next(resumed)] == first[1:]
    resumed.acknowledge([first[1][0], first[2][0]])
    assert next(resumed) == expected_next


def test_first_failure_after_success_is_not_a_repeated_failure() -> None:
    sample = next(GlimmerDataAnalystDataset.Config(seed=42).build())
    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    try:
        env._note_command("test -f transient.txt", succeeded=True)
        env._note_command("test -f transient.txt", succeeded=False)
        assert env._repeated_failing_commands == 0
        env._note_command("test -f transient.txt", succeeded=False)
        assert env._repeated_failing_commands == 1
    finally:
        asyncio.run(env.close())


def test_commands_use_active_python_and_strip_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GLIMMER_TEST_SECRET", "not-for-the-child")
    sample = next(GlimmerDataAnalystDataset.Config(seed=42).build())
    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    try:
        code = (
            "import json,os,sys; print(json.dumps({"
            "'prefix':sys.prefix,'home':os.environ['HOME'],"
            "'secret':os.environ.get('GLIMMER_TEST_SECRET')}))"
        )
        command = f"python3 -c {shlex.quote(code)}"
        result = json.loads(asyncio.run(env._run_command(command)))
        assert result["prefix"] == sys.prefix
        assert result["home"] == str(env._workdir)
        assert result["secret"] is None
    finally:
        asyncio.run(env.close())


def test_combined_tool_output_is_bounded() -> None:
    sample = next(GlimmerDataAnalystDataset.Config(seed=42).build())
    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    try:
        output = asyncio.run(
            env._run_command(
                "python3 -c \"import sys; print('x'*1200); "
                "print('y'*1200,file=sys.stderr)\""
            )
        )
        assert len(output) < MAX_OUTPUT_CHARS + 100
        assert output.count("[truncated") == 1
    finally:
        asyncio.run(env.close())


def test_failure_diagnostic_survives_large_stdout() -> None:
    sample = next(GlimmerDataAnalystDataset.Config(seed=42).build())
    env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
    try:
        output = asyncio.run(
            env._run_command(
                "python3 -c \"import sys; print('x'*2000); "
                "print('recoverable error',file=sys.stderr); sys.exit(1)\""
            )
        )
        assert "recoverable error" in output
        assert "[exit code: 1]" in output
    finally:
        asyncio.run(env.close())


def test_command_does_not_block_the_sibling_event_loop() -> None:
    async def run() -> None:
        sample = next(GlimmerDataAnalystDataset.Config(seed=42).build())
        env = GlimmerDataAnalystEnv.Config().build(env_input=sample)
        command = "printf ready > ready; while [ ! -f release ]; do sleep 0.01; done"
        task = asyncio.create_task(
            env.step(_completion_message([_bash_call({"command": command})]))
        )
        try:

            async def wait_until_ready() -> None:
                while not (env._workdir / "ready").exists():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_until_ready(), timeout=10.0)
            assert not task.done()
            (env._workdir / "release").write_text("release")
            output = await task
            assert output.env_rewards["executed_tool_calls"] == 1.0
        finally:
            await env.close()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
