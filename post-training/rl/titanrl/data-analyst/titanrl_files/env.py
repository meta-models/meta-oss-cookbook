# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A one-tool (``bash``) code-execution env for the data-analyst recipe.

Mirrors the real PinchBench failure mode this recipe targets: the model has to
construct and run a (possibly multi-line) Python invocation through a SHELL command
string, exactly like the terminal tool in the agent scaffold PinchBench measured
against. That is where the original bug lived: literal ``\\n`` inside a heredoc
instead of real newlines, causing a ``SyntaxError`` the model then repeated instead
of recovering from. A ``run_python(code=...)`` tool would sidestep that exact bug by
never routing through a shell; ``bash`` keeps the recipe honest about what it fixes.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from renderers import Message, ToolSpec
from renderers.base import ParsedToolCall

from torchtitan.rl.rollout.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.rl.examples.glimmer_data_analyst.data import (
    OUTPUT_FILENAME,
    GlimmerDataAnalystSample,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 800
"""Cap on stdout/stderr fed back to the model (~200 tokens). Deliberately tight:
every tool reply is re-sent as part of the prompt on every subsequent turn, so a
verbose reply is paid for again each turn. Sized together with the rollout token
budget in ``config_registry.py`` so a full-length rollout cannot outgrow it."""

# Minimal env for the sandboxed subprocess: a bare PATH and a HOME/TMPDIR pointed at
# the task workspace, not the real user's, so `~`-relative paths and stray `.cache`
# writes land in the scratch dir rather than the host account. This is a lightweight
# subprocess sandbox, not full container isolation -- see the recipe README.
_SANDBOX_BASE_ENV = {"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C.UTF-8"}

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
                "description": "The shell command to run (via `bash -c`).",
            },
        },
        "required": ["command"],
    },
}


def _command_from_tool_call(tool_call: ParsedToolCall) -> str:
    """Pull the ``command`` argument out of a renderer-parsed ``bash`` tool call."""
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
    dropped = len(text) - limit
    return text[:limit] + f"\n... [truncated {dropped} chars]"


class GlimmerDataAnalystEnv(MessageEnv):
    """Single-tool (``bash``) code-execution env for one data-analysis task.

    Each instance owns a private scratch directory (created in ``__init__``, removed
    in ``close``) holding the task's input data file. The model runs shell commands
    there via the ``bash`` tool until it either writes the required output file and
    stops calling tools, or runs out of turns/tokens.

    Correctness is graded on every step (``_grade``) and reported through
    ``MessageEnvStepOutput.env_rewards`` -- the rubric reads the last graded turn's
    ``env_rewards`` (see ``rubric.py``) rather than touching the filesystem itself,
    so grading is correct regardless of where or when the rubric runs relative to
    this env's lifetime.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        command_timeout_s: float = 60.0
        """Wall-clock timeout for one ``bash`` command."""

    def __init__(self, config: Config, *, env_input: GlimmerDataAnalystSample) -> None:
        self._output_filename = OUTPUT_FILENAME
        self._golden_value = env_input.golden_value
        self._tolerance = env_input.tolerance
        self._command_timeout_s = config.command_timeout_s
        self._prompt = env_input.prompt

        # Progress signals accumulated across the rollout (see _grade).
        self._any_command_succeeded = False
        self._command_history: list[str] = []
        self._repeated_failing_commands = 0

        self._workdir = Path(tempfile.mkdtemp(prefix="glimmer_da_"))
        (self._workdir / env_input.input_filename).write_text(env_input.input_content)

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": self._prompt}],
            tools=[BASH_TOOL],
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        tool_calls: list[ParsedToolCall] = completion_message.get("tool_calls") or []
        if not tool_calls:
            # No tool call -> the assistant considers itself done; grade now.
            return MessageEnvStepOutput(done=True, env_rewards=self._grade())

        env_messages: list[Message] = [
            {"role": "tool", "content": self._run_command(_command_from_tool_call(tc))}
            for tc in tool_calls
        ]
        # Grade after every command too, not only on the final turn: a model that
        # already succeeded and then keeps exploring should not lose credit for not
        # stopping on the exact turn it wrote the file. The rubric reads whichever
        # graded turn is last, whether the rollout ends via "done" or via running
        # out of the turn budget.
        return MessageEnvStepOutput(env_messages=env_messages, env_rewards=self._grade())

    async def close(self) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)

    def _run_command(self, command: str) -> str:
        if not command.strip():
            return "[error] empty command"
        env = dict(_SANDBOX_BASE_ENV)
        env["HOME"] = str(self._workdir)
        env["TMPDIR"] = str(self._workdir)
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self._workdir,
                capture_output=True,
                text=True,
                timeout=self._command_timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            self._note_command(command, succeeded=False)
            return f"[error] command timed out after {self._command_timeout_s:.0f}s"
        except Exception as exc:  # a malformed command must not crash the rollout
            self._note_command(command, succeeded=False)
            return f"[error] failed to run command: {exc}"

        self._note_command(command, succeeded=result.returncode == 0)

        parts = []
        if result.stdout:
            parts.append(_truncate(result.stdout))
        if result.stderr:
            parts.append(f"[stderr]\n{_truncate(result.stderr)}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(parts) if parts else "[no output]"

    def _note_command(self, command: str, *, succeeded: bool) -> None:
        """Track the two behaviors this recipe exists to shape: getting a command
        to run at all, and NOT repeating one that already failed (the thrash loop
        PinchBench observed -- re-sending identical broken code until timeout)."""
        if succeeded:
            self._any_command_succeeded = True
        elif command in self._command_history:
            self._repeated_failing_commands += 1
        self._command_history.append(command)

    def _grade(self) -> dict[str, float]:
        """Graded progress signals for the rubric, from weakest to strongest.

        Deliberately graded rather than a bare correct/not: with a single 0/1
        signal, every rollout in a GRPO group scores identically until the model
        can solve the task outright, the group has no reward variance, and the
        batch is discarded as untrainable -- so the policy never gets a gradient
        to climb. These intermediate signals distinguish "ran code successfully"
        from "kept erroring" and "wrote an artifact" from "wrote nothing", which
        are exactly the behaviors this recipe targets.
        """
        signals = {
            "ran_ok": 1.0 if self._any_command_succeeded else 0.0,
            "repeated_failing_commands": float(self._repeated_failing_commands),
            "file_exists": 0.0,
            "file_parses": 0.0,
            "correct": 0.0,
        }

        output_path = self._workdir / self._output_filename
        if not output_path.exists():
            return signals
        signals["file_exists"] = 1.0

        try:
            value = json.loads(output_path.read_text())["answer"]
            numeric = float(value)
        except Exception:
            return signals
        signals["file_parses"] = 1.0

        if abs(numeric - self._golden_value) <= self._tolerance:
            signals["correct"] = 1.0
        return signals
