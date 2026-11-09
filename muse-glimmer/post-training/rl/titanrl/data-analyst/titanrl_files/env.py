# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A one-tool bash environment for deterministic CSV/log analysis.

Commands execute in a private workspace with a stripped environment. Process
cleanup covers the command's process group, not descendants that escape it
with setsid. This is not filesystem or container isolation; use appropriately
isolated infrastructure without host secrets.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from renderers import Message, ToolSpec
from renderers.base import ParsedToolCall
from torchtitan.rl.examples.glimmer_data_analyst.data import (
    OUTPUT_FILENAME,
    GlimmerDataAnalystSample,
)
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

MAX_OUTPUT_CHARS = 800
"""Combined model-facing body limit, excluding truncation/exit annotations.

Characters are not a token bound. The config separately reserves completion
headroom in the token budget.
"""
MAX_RETAINED_STREAM_BYTES = 4096
"""Retain only this byte prefix per stream while draining the entire pipe."""
_DRAIN_SHUTDOWN_TIMEOUT_S = 1.0
_MAX_TURN_TIMEOUT_S = 110.0

# Keep the active venv Python, but not credentials, shell startup variables,
# or the real user's HOME/TMPDIR. Do not resolve sys.executable's venv symlink.
_SANDBOX_BASE_ENV = {
    "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/local/bin",
    "LANG": "C.UTF-8",
}

BASH_TOOL: ToolSpec = {
    "name": "bash",
    "description": (
        "Run a shell command in your task workspace and return its stdout/stderr. "
        "Your input data file is already there; write your output file there too."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run (via bash -c).",
            },
        },
        "required": ["command"],
    },
}


def _command_from_tool_call(tool_call: ParsedToolCall) -> str:
    """Malformed arguments and unsupported tool names are failed empty calls."""
    if tool_call.name != "bash":
        return ""
    arguments = tool_call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    if isinstance(arguments, dict):
        command = arguments.get("command", "")
        return command if isinstance(command, str) else ""
    return ""


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} retained chars]"


@dataclass(slots=True)
class _CapturedStream:
    prefix: bytearray = field(default_factory=bytearray)
    num_bytes: int = 0

    async def drain(self, reader: asyncio.StreamReader) -> None:
        while chunk := await reader.read(MAX_RETAINED_STREAM_BYTES):
            self.num_bytes += len(chunk)
            remaining = MAX_RETAINED_STREAM_BYTES - len(self.prefix)
            if remaining:
                self.prefix.extend(chunk[:remaining])
            # A continuously ready pipe must not monopolize sibling rollouts.
            await asyncio.sleep(0)


async def _await_shutdown(task: asyncio.Task[None]) -> None:
    """Finish owned cleanup despite repeated cancellation, then propagate it."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


class GlimmerDataAnalystEnv(MessageEnv):
    """Execute sequential bash calls; kill/reap commands before removing scratch.

    Grades travel in env_rewards, so rubric evaluation is independent of the
    workspace lifetime. A budget-ended turn retains earlier calls and grades,
    but carries an explicit turn_budget_exhausted signal rather than implying
    that analysis succeeded.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        command_timeout_s: float = 60.0
        """Wall-clock limit for one shell command, including spawn and pipe drains."""
        turn_timeout_s: float = 90.0
        """Total turn budget. At most 110s leaves cleanup margin below the recipe's
        120s TokenEnv timeout; lower it if the outer timeout is reduced."""

        def __post_init__(self) -> None:
            if not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0:
                raise ValueError("command_timeout_s must be finite and positive")
            if (
                not math.isfinite(self.turn_timeout_s)
                or not 0 < self.turn_timeout_s <= _MAX_TURN_TIMEOUT_S
            ):
                raise ValueError("turn_timeout_s must be finite and in (0, 110]")

    def __init__(self, config: Config, *, env_input: GlimmerDataAnalystSample) -> None:
        self._output_filename = OUTPUT_FILENAME
        self._golden_value = env_input.golden_value
        self._tolerance = env_input.tolerance
        self._command_timeout_s = config.command_timeout_s
        self._turn_timeout_s = config.turn_timeout_s
        self._prompt = env_input.prompt
        self._any_command_succeeded = False
        self._failed_commands: set[str] = set()
        self._repeated_failing_commands = 0
        self._executed_tool_calls = 0
        self._command_timeouts = 0
        self._turn_budget_exhausted = False
        self._closed = False
        self._command_tasks: set[asyncio.Task[str]] = set()
        self._processes: set[asyncio.subprocess.Process] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._last_stdout = _CapturedStream()
        self._last_stderr = _CapturedStream()
        self._workdir = Path(tempfile.mkdtemp(prefix="glimmer_da_"))
        (self._workdir / env_input.input_filename).write_text(env_input.input_content)

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": self._prompt}],
            tools=[BASH_TOOL],
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        if self._closed:
            raise RuntimeError("environment is closed")
        tool_calls: list[ParsedToolCall] = completion_message.get("tool_calls") or []
        if not tool_calls or self._turn_budget_exhausted:
            return MessageEnvStepOutput(done=True, env_rewards=self._grade())

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._turn_timeout_s
        env_messages: list[Message] = []
        for tool_call in tool_calls:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self._turn_budget_exhausted = True
                env_messages.append(
                    {
                        "role": "tool",
                        "content": "[error] turn budget exhausted; remaining calls were not executed",
                    }
                )
                break
            output = await self._run_command(
                _command_from_tool_call(tool_call),
                timeout_s=min(self._command_timeout_s, remaining),
                turn_budget_limited=remaining <= self._command_timeout_s,
            )
            env_messages.append({"role": "tool", "content": output})
            if self._turn_budget_exhausted:
                break
        return MessageEnvStepOutput(
            done=self._turn_budget_exhausted,
            env_messages=env_messages,
            env_rewards=self._grade(),
        )

    async def close(self) -> None:
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await _await_shutdown(self._close_task)

    async def _close(self) -> None:
        tasks = tuple(self._command_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._command_tasks.difference_update(tasks)
        # Each command task finishes its process cleanup in finally.
        if self._processes:
            raise RuntimeError("command processes remain after shutdown")
        shutil.rmtree(self._workdir, ignore_errors=True)

    async def _run_command(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
        turn_budget_limited: bool = False,
    ) -> str:
        if self._closed:
            raise RuntimeError("environment is closed")
        if not command.strip():
            self._note_command("", succeeded=False)
            return "[error] empty or malformed command"
        task = asyncio.create_task(
            self._execute_command(
                command,
                timeout_s=self._command_timeout_s if timeout_s is None else timeout_s,
                turn_budget_limited=turn_budget_limited,
            )
        )
        self._command_tasks.add(task)
        task.add_done_callback(self._command_tasks.discard)
        # Cancellation propagates into the owned task, whose finally kills/reaps.
        return await task

    def _register_process(self, process: asyncio.subprocess.Process) -> None:
        self._processes.add(process)
        self._executed_tool_calls += 1

    async def _execute_command(
        self, command: str, *, timeout_s: float, turn_budget_limited: bool
    ) -> str:
        env = dict(_SANDBOX_BASE_ENV)
        env["HOME"] = str(self._workdir)
        env["TMPDIR"] = str(self._workdir)
        stdout, stderr = _CapturedStream(), _CapturedStream()
        self._last_stdout, self._last_stderr = stdout, stderr
        spawn = None
        process = None
        completion = None
        annotation = ""
        try:
            try:
                async with asyncio.timeout(timeout_s):
                    # Shield spawning: cancellation must still obtain the process
                    # handle before cleanup, rather than orphaning a new child.
                    spawn = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            "bash",
                            "-c",
                            command,
                            cwd=self._workdir,
                            env=env,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            limit=2 * MAX_RETAINED_STREAM_BYTES,
                            start_new_session=True,
                        )
                    )
                    process = await asyncio.shield(spawn)
                    self._register_process(process)
                    assert process.stdout is not None and process.stderr is not None
                    completion = asyncio.gather(
                        stdout.drain(process.stdout),
                        stderr.drain(process.stderr),
                        process.wait(),
                        return_exceptions=True,
                    )
                    results = await asyncio.shield(completion)
                    if any(isinstance(result, BaseException) for result in results):
                        raise RuntimeError("failed to drain command output")
            except TimeoutError:
                self._command_timeouts += 1
                self._note_command(command, succeeded=False)
                if turn_budget_limited:
                    self._turn_budget_exhausted = True
                    annotation = "[error] turn budget exhausted; remaining calls were not executed"
                else:
                    annotation = f"[error] command timed out after {timeout_s:g}s"
            except asyncio.CancelledError:
                self._note_command(command, succeeded=False)
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                self._note_command(command, succeeded=False)
                annotation = f"[error] failed to run command: {str(exc)[:200]}"
            else:
                assert process is not None
                self._note_command(command, succeeded=process.returncode == 0)
        finally:
            if spawn is not None:
                cleanup = asyncio.create_task(
                    self._shutdown_process(spawn, process, completion)
                )
                await _await_shutdown(cleanup)
        return self._format_output(stdout, stderr, process, annotation)

    async def _shutdown_process(
        self,
        spawn: asyncio.Task[asyncio.subprocess.Process],
        process: asyncio.subprocess.Process | None,
        completion: asyncio.Future | None,
    ) -> None:
        if process is None:
            try:
                process = await spawn
            except (OSError, ValueError):
                return
            # Spawn completed after timeout/cancellation. It still counts as a
            # real dispatch and must be registered, killed, and reaped.
            self._register_process(process)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if completion is not None:
            # Give killed group members time to close their pipes. An escaped
            # descendant must not keep our read transports alive indefinitely.
            try:
                await asyncio.wait_for(
                    asyncio.shield(completion), timeout=_DRAIN_SHUTDOWN_TIMEOUT_S
                )
            except TimeoutError:
                completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)
        # The child watcher sets returncode when the direct child is reaped,
        # even when Process.wait is still waiting on inherited pipe handles.
        while process.returncode is None:
            await asyncio.sleep(0.01)
        # Process exposes no public close for its stdout/stderr transports.
        # Close only this owned transport after reap, including on cancelled
        # spawn, so inherited pipes cannot keep Process.wait pending.
        process._transport.close()
        await process.wait()
        self._processes.discard(process)

    @staticmethod
    def _format_output(
        stdout: _CapturedStream,
        stderr: _CapturedStream,
        process: asyncio.subprocess.Process | None,
        annotation: str,
    ) -> str:
        out = stdout.prefix.decode("utf-8", errors="replace")
        err = stderr.prefix.decode("utf-8", errors="replace")
        err = f"[stderr]\n{err}" if err else ""
        failed = bool(annotation) or (process is not None and process.returncode != 0)
        streams = (err, out) if failed else (out, err)
        parts = [_truncate("\n".join(stream for stream in streams if stream))]
        discarded = (
            stdout.num_bytes
            - len(stdout.prefix)
            + stderr.num_bytes
            - len(stderr.prefix)
        )
        if discarded:
            parts.append(f"[discarded {discarded} output bytes]")
        if annotation:
            parts.append(annotation)
        if process is not None and process.returncode:
            parts.append(f"[exit code: {process.returncode}]")
        return "\n".join(part for part in parts if part) or "[no output]"

    def _note_command(self, command: str, *, succeeded: bool) -> None:
        if succeeded:
            self._any_command_succeeded = True
        else:
            if command in self._failed_commands:
                self._repeated_failing_commands += 1
            self._failed_commands.add(command)

    def _grade(self) -> dict[str, float]:
        """Accumulate progress/execution signals and check the current answer file."""
        signals = {
            "ran_ok": 1.0 if self._any_command_succeeded else 0.0,
            "repeated_failing_commands": float(self._repeated_failing_commands),
            "executed_tool_calls": float(self._executed_tool_calls),
            "command_timeouts": float(self._command_timeouts),
            "turn_budget_exhausted": float(self._turn_budget_exhausted),
            "file_exists": 0.0,
            "file_parses": 0.0,
            "correct": 0.0,
        }
        output_path = self._workdir / self._output_filename
        if not output_path.exists():
            return signals
        signals["file_exists"] = 1.0
        try:
            numeric = float(json.loads(output_path.read_text())["answer"])
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return signals
        signals["file_parses"] = 1.0
        if abs(numeric - self._golden_value) <= self._tolerance:
            signals["correct"] = 1.0
        return signals
