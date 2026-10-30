# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CPU subprocess regressions; every file is in an env-owned workspace."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest
from renderers.base import ParsedToolCall
from torchtitan.rl.examples.glimmer_data_analyst.data import GlimmerDataAnalystSample
from torchtitan.rl.examples.glimmer_data_analyst.env import (
    MAX_OUTPUT_CHARS,
    MAX_RETAINED_STREAM_BYTES,
    GlimmerDataAnalystEnv,
)


def _make_env(**config_values: float) -> GlimmerDataAnalystEnv:
    sample = GlimmerDataAnalystSample(
        task_id="process_lifecycle",
        prompt='Write {"answer": 42} to answer.json.',
        input_filename="data.csv",
        input_content="value\n42\n",
        golden_value=42.0,
    )
    return GlimmerDataAnalystEnv.Config(**config_values).build(env_input=sample)


def _python_command(code: str) -> str:
    return f"exec python3 -c {shlex.quote(code)}"


def _message(*commands: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            ParsedToolCall(
                raw="", name="bash", arguments={"command": command}, id=str(i)
            )
            for i, command in enumerate(commands)
        ],
    }


async def _wait_for_file(path: Path) -> str:
    async def read_when_ready() -> str:
        while True:
            try:
                contents = path.read_text()
            except FileNotFoundError:
                contents = ""
            if contents:
                return contents
            await asyncio.sleep(0.01)

    # A readiness handshake, not a sleep asserting elapsed command duration.
    return await asyncio.wait_for(read_when_ready(), timeout=10.0)


def _blocked_group_command() -> str:
    child_code = (
        "import os,signal; from pathlib import Path; "
        "Path('child-ready').write_text(str(os.getpid())); signal.pause()"
    )
    return _python_command(
        "import json,os,signal,subprocess,sys; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "Path('ready').write_text(json.dumps({'parent':os.getpid(),'child':child.pid})); "
        "signal.pause()"
    )


def _blocked_command() -> str:
    return _python_command(
        "import os,signal; from pathlib import Path; "
        "Path('ready').write_text(str(os.getpid())); signal.pause()"
    )


def _process_running(pid: int) -> bool:
    # These IDs come only from this test's children. A killed grandchild can
    # briefly be a zombie until its adopter reaps it; it is no longer running.
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in {"Z", "X"}


def _assert_direct_child_reaped(process: asyncio.subprocess.Process) -> None:
    assert process.returncode == -signal.SIGKILL
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


def test_close_kills_group_and_reaps_command_before_removing_workspace() -> None:
    async def run() -> None:
        env = _make_env(command_timeout_s=30.0)
        task = asyncio.create_task(env._run_command(_blocked_group_command()))
        workdir = env._workdir
        try:
            pids = json.loads(await _wait_for_file(workdir / "ready"))
            await _wait_for_file(workdir / "child-ready")
            process = next(iter(env._processes))
            assert process.pid == pids["parent"]
            assert os.getpgid(pids["child"]) == process.pid
            await env.close()
            with pytest.raises(asyncio.CancelledError):
                await task
            _assert_direct_child_reaped(process)
            assert not _process_running(pids["child"])
            assert not env._processes
            assert not env._command_tasks
            assert not workdir.exists()
        finally:
            await env.close()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_external_cancellation_propagates_only_after_group_shutdown() -> None:
    async def run() -> None:
        env = _make_env(command_timeout_s=30.0)
        task = asyncio.create_task(env.step(_message(_blocked_group_command())))
        try:
            pids = json.loads(await _wait_for_file(env._workdir / "ready"))
            await _wait_for_file(env._workdir / "child-ready")
            process = next(iter(env._processes))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            _assert_direct_child_reaped(process)
            assert not _process_running(pids["child"])
            assert not env._processes
            assert not env._command_tasks
            assert env._grade()["executed_tool_calls"] == 1.0
            assert env._workdir.exists()
            assert not env._any_command_succeeded
        finally:
            await env.close()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_command_timeout_preserves_prior_artifact_and_messages() -> None:
    async def run() -> None:
        env = _make_env(command_timeout_s=5.0)
        try:
            output = await asyncio.wait_for(
                env.step(
                    _message(
                        "printf '{\"answer\":42}' > answer.json; printf artifact-written",
                        _blocked_command(),
                    )
                ),
                timeout=20.0,
            )
            assert output.done is False
            assert len(output.env_messages) == 2
            assert output.env_messages[0]["content"] == "artifact-written"
            assert "command timed out" in output.env_messages[1]["content"]
            assert output.env_rewards["correct"] == 1.0
            assert output.env_rewards["ran_ok"] == 1.0
            assert output.env_rewards["executed_tool_calls"] == 2.0
            assert output.env_rewards["command_timeouts"] == 1.0
            assert output.env_rewards["turn_budget_exhausted"] == 0.0
            assert not env._processes
            assert not _process_running(int((env._workdir / "ready").read_text()))
        finally:
            await env.close()

    asyncio.run(run())


def test_turn_budget_returns_prior_progress_and_does_not_start_remaining_calls() -> (
    None
):
    async def run() -> None:
        env = _make_env(command_timeout_s=30.0, turn_timeout_s=5.0)
        try:
            output = await asyncio.wait_for(
                env.step(
                    _message(
                        "printf '{\"answer\":42}' > answer.json; printf artifact-written",
                        _blocked_command(),
                        "printf should-not-run > unexecuted.txt",
                    )
                ),
                timeout=20.0,
            )
            assert output.done is True
            assert len(output.env_messages) == 2
            assert output.env_messages[0]["content"] == "artifact-written"
            assert "[error] turn budget exhausted" in output.env_messages[1]["content"]
            assert output.env_rewards["correct"] == 1.0
            assert output.env_rewards["executed_tool_calls"] == 2.0
            assert output.env_rewards["command_timeouts"] == 1.0
            assert output.env_rewards["turn_budget_exhausted"] == 1.0
            assert not (env._workdir / "unexecuted.txt").exists()
            assert not env._processes
            assert not _process_running(int((env._workdir / "ready").read_text()))
        finally:
            await env.close()

    asyncio.run(run())


def test_output_retains_only_fixed_prefixes_and_preserves_failure_diagnostics() -> None:
    async def run() -> None:
        env = _make_env(command_timeout_s=20.0)
        num_bytes = 64 * MAX_RETAINED_STREAM_BYTES
        try:
            output = await env._run_command(
                _python_command(
                    "import sys; "
                    f"sys.stdout.buffer.write(b'x'*{num_bytes}); "
                    f"sys.stderr.buffer.write(b'failure diagnostic\\n'+b'y'*{num_bytes}); "
                    "sys.exit(7)"
                )
            )
            assert len(env._last_stdout.prefix) == MAX_RETAINED_STREAM_BYTES
            assert len(env._last_stderr.prefix) == MAX_RETAINED_STREAM_BYTES
            assert env._last_stdout.num_bytes == num_bytes
            assert env._last_stderr.num_bytes > num_bytes
            body = output.split("\n... [truncated", 1)[0]
            assert len(body) <= MAX_OUTPUT_CHARS
            assert output.startswith("[stderr]\nfailure diagnostic")
            assert "[discarded " in output
            assert "[exit code: 7]" in output
            assert len(output) < MAX_OUTPUT_CHARS + 200
            assert not env._processes
        finally:
            await env.close()

    asyncio.run(run())


def test_waiting_command_allows_a_real_sibling_command_to_finish() -> None:
    async def run() -> None:
        env = _make_env(command_timeout_s=30.0)
        sibling = _make_env(command_timeout_s=30.0)
        command = _python_command(
            "import time; from pathlib import Path; "
            "Path('ready').write_text('ready'); "
            "\nwhile not Path('release').exists(): time.sleep(0.01)\n"
            "print('released')"
        )
        task = asyncio.create_task(env._run_command(command))
        try:
            await _wait_for_file(env._workdir / "ready")
            result = await asyncio.wait_for(
                sibling._run_command("printf sibling-finished"), timeout=10.0
            )
            assert result == "sibling-finished"
            assert not task.done()
            (env._workdir / "release").write_text("release")
            assert (await task).strip() == "released"
            assert env._grade()["executed_tool_calls"] == 1.0
            assert sibling._grade()["executed_tool_calls"] == 1.0
        finally:
            await asyncio.gather(env.close(), sibling.close())
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "arguments",
    [{"command": ""}, {"command": 42}, {}, '{"command":'],
    ids=["empty", "wrong-type", "missing", "invalid-json"],
)
def test_malformed_repeats_fail_without_counting_subprocess_starts(arguments) -> None:
    async def run() -> None:
        env = _make_env()
        call = ParsedToolCall(raw="", name="bash", arguments=arguments, id="malformed")
        message = {"role": "assistant", "tool_calls": [call]}
        try:
            first = await env.step(message)
            second = await env.step(message)
            assert "[error]" in first.env_messages[0]["content"]
            assert second.env_rewards["repeated_failing_commands"] == 1.0
            assert second.env_rewards["executed_tool_calls"] == 0.0
            assert second.env_rewards["ran_ok"] == 0.0
            success = await env.step(_message("printf valid"))
            assert success.env_rewards["executed_tool_calls"] == 1.0
            assert success.env_rewards["ran_ok"] == 1.0
            third = await env.step(message)
            assert third.env_rewards["repeated_failing_commands"] == 2.0
            assert third.env_rewards["executed_tool_calls"] == 1.0
        finally:
            await env.close()

    asyncio.run(run())


def test_sequential_calls_use_active_python_and_cumulative_execution_counts() -> None:
    async def run() -> None:
        env = _make_env()
        try:
            output = await env.step(
                _message(
                    "printf first > sequence.txt",
                    _python_command(
                        "import json,os,sys; from pathlib import Path; "
                        "print(json.dumps({'sequence':Path('sequence.txt').read_text(), "
                        "'prefix':sys.prefix, 'home':os.environ['HOME'], "
                        "'tmpdir':os.environ['TMPDIR']}))"
                    ),
                )
            )
            details = json.loads(output.env_messages[1]["content"])
            assert details["sequence"] == "first"
            assert details["prefix"] == sys.prefix
            assert details["home"] == str(env._workdir)
            assert details["tmpdir"] == str(env._workdir)
            assert output.env_rewards["executed_tool_calls"] == 2.0
            final = await env.step({"role": "assistant", "content": "done"})
            assert final.done is True
            assert final.env_rewards["executed_tool_calls"] == 2.0
        finally:
            await env.close()

    asyncio.run(run())


def test_failed_commands_only_bookkeeping_uses_real_exit_codes() -> None:
    async def run() -> None:
        env = _make_env()
        command = "test -f transient.txt"
        try:
            await env._run_command("printf present > transient.txt")
            assert await env._run_command(command) == "[no output]"
            await env._run_command("rm transient.txt")
            assert "[exit code: 1]" in await env._run_command(command)
            assert env._grade()["repeated_failing_commands"] == 0.0
            assert "[exit code: 1]" in await env._run_command(command)
            assert env._grade()["repeated_failing_commands"] == 1.0
            assert env._grade()["executed_tool_calls"] == 5.0
        finally:
            await env.close()

    asyncio.run(run())


def test_close_is_idempotent_and_rejects_new_command_starts() -> None:
    async def run() -> None:
        env = _make_env()
        workdir = env._workdir
        await env.close()
        await env.close()
        assert not workdir.exists()
        with pytest.raises(RuntimeError, match="closed"):
            await env._run_command("printf should-not-run")
        assert env._grade()["executed_tool_calls"] == 0.0

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command_timeout_s", 0.0),
        ("command_timeout_s", float("inf")),
        ("turn_timeout_s", 0.0),
        ("turn_timeout_s", 120.0),
        ("turn_timeout_s", float("nan")),
    ],
)
def test_timeout_config_requires_finite_limits_with_outer_cleanup_margin(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        GlimmerDataAnalystEnv.Config(**{field: value})
