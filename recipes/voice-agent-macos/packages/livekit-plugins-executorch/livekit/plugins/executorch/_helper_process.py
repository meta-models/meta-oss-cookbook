from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from .log import logger

_DEFAULT_MAX_HEADER_BYTES = 64 * 1024
_DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class HelperProcessError(RuntimeError):
    """Raised when a native helper exits or violates the framed protocol."""


class HelperProtocolError(HelperProcessError):
    """Raised when a helper sends invalid framing or message data."""


class HelperProcess:
    """Async lifecycle and JSON-plus-binary framing for one native helper."""

    def __init__(
        self,
        executable: str,
        argv: Sequence[str] = (),
        *,
        name: str,
        ready_timeout: float = 120.0,
        shutdown_timeout: float = 2.0,
        terminate_timeout: float = 2.0,
        max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        if not executable:
            raise ValueError("helper executable must be non-empty")
        if max_header_bytes <= 0 or max_payload_bytes <= 0:
            raise ValueError("helper framing limits must be positive")

        self._executable = executable
        self._argv = tuple(argv)
        self._name = name
        self._ready_timeout = ready_timeout
        self._shutdown_timeout = shutdown_timeout
        self._terminate_timeout = terminate_timeout
        self._max_header_bytes = max_header_bytes
        self._max_payload_bytes = max_payload_bytes
        self._process: asyncio.subprocess.Process | None = None
        self._write_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending_read: asyncio.Task[tuple[dict[str, Any], bytes | None]] | None = None
        self._ready_message: dict[str, Any] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    async def start(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self.running:
                if self._ready_message is None:
                    raise HelperProcessError(f"{self._name} helper has no cached ready message")
                return dict(self._ready_message)

            await self._close_locked(graceful=False)
            self._stderr_tail.clear()
            try:
                process = await asyncio.create_subprocess_exec(
                    self._executable,
                    *self._argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=self._max_header_bytes + 1,
                )
            except OSError as exc:
                raise HelperProcessError(
                    f"failed to start {self._name} helper {self._executable!r}: {exc}"
                ) from exc

            self._process = process
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(process), name=f"{self._name}-stderr"
            )
            try:
                message, payload = await asyncio.wait_for(
                    self.read_message(), timeout=self._ready_timeout
                )
            except (TimeoutError, HelperProcessError) as exc:
                await self._close_locked(graceful=False)
                context = self._format_stderr_context()
                if isinstance(exc, asyncio.TimeoutError):
                    raise HelperProcessError(
                        f"{self._name} helper did not become ready within "
                        f"{self._ready_timeout:.1f}s{context}"
                    ) from None
                raise HelperProcessError(f"{exc}{context}") from exc

            if payload is not None or message.get("type") != "ready" or message.get("version") != 1:
                await self._close_locked(graceful=False)
                raise HelperProtocolError(
                    f"{self._name} helper sent invalid ready message: {message!r}"
                    f"{self._format_stderr_context()}"
                )
            self._ready_message = dict(message)
            return dict(message)

    async def write_message(
        self, message: Mapping[str, Any], payload: bytes | bytearray | memoryview | None = None
    ) -> None:
        process = self._require_running()
        if process.stdin is None:
            raise HelperProcessError(f"{self._name} helper stdin is unavailable")

        payload_bytes = bytes(payload) if payload is not None else b""
        if len(payload_bytes) > self._max_payload_bytes:
            raise HelperProtocolError(
                f"{self._name} helper payload exceeds {self._max_payload_bytes} bytes"
            )
        try:
            header = json.dumps(dict(message), separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError) as exc:
            raise HelperProtocolError(f"helper message is not valid JSON: {exc}") from exc
        if b"\n" in header or len(header) > self._max_header_bytes:
            raise HelperProtocolError("helper message header exceeds framing limits")

        async with self._write_lock:
            try:
                process.stdin.write(header + b"\n")
                if payload_bytes:
                    process.stdin.write(payload_bytes)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise self._process_exit_error("failed to write helper request") from exc

    async def read_message(self) -> tuple[dict[str, Any], bytes | None]:
        if self._pending_read is None:
            self._pending_read = asyncio.create_task(
                self._read_message_impl(), name=f"{self._name}-read"
            )
        task = self._pending_read
        try:
            return await asyncio.shield(task)
        finally:
            if task.done() and self._pending_read is task:
                self._pending_read = None

    async def restart(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            await self._close_locked(graceful=False)
        return await self.start()

    async def aclose(self, *, graceful: bool = True) -> None:
        async with self._lifecycle_lock:
            await self._close_locked(graceful=graceful)

    async def _read_message_impl(self) -> tuple[dict[str, Any], bytes | None]:
        process = self._require_running()
        if process.stdout is None:
            raise HelperProcessError(f"{self._name} helper stdout is unavailable")
        try:
            line = await process.stdout.readline()
        except ValueError as exc:
            raise HelperProtocolError(
                f"{self._name} helper header exceeds {self._max_header_bytes} bytes"
            ) from exc
        if not line:
            raise self._process_exit_error("unexpected EOF from helper")
        if not line.endswith(b"\n") or len(line) - 1 > self._max_header_bytes:
            raise HelperProtocolError(
                f"{self._name} helper header exceeds {self._max_header_bytes} bytes"
            )
        try:
            parsed = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HelperProtocolError(f"{self._name} helper sent malformed JSON header") from exc
        if not isinstance(parsed, dict):
            raise HelperProtocolError(f"{self._name} helper header must be a JSON object")

        payload_size = parsed.get("payload_byte_count", 0)
        if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size < 0:
            raise HelperProtocolError("helper payload_byte_count must be a non-negative integer")
        if payload_size > self._max_payload_bytes:
            raise HelperProtocolError(
                f"{self._name} helper payload exceeds {self._max_payload_bytes} bytes"
            )
        if payload_size == 0:
            return parsed, None
        try:
            payload = await process.stdout.readexactly(payload_size)
        except asyncio.IncompleteReadError as exc:
            raise HelperProcessError(
                f"unexpected EOF reading {self._name} helper payload: "
                f"expected {payload_size}, received {len(exc.partial)}"
            ) from exc
        return parsed, payload

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while line := await process.stderr.readline():
            text = line.decode(errors="replace").rstrip()
            self._stderr_tail.append(text)
            logger.debug("%s helper: %s", self._name, text)

    async def _close_locked(self, *, graceful: bool) -> None:
        process = self._process
        if process is None:
            return

        if graceful and process.returncode is None:
            with contextlib.suppress(HelperProcessError, HelperProtocolError):
                await self.write_message({"type": "shutdown", "version": 1})
        if process.returncode is None and graceful:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout)
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=self._terminate_timeout)
            except TimeoutError:
                process.kill()
                await process.wait()

        if process.stdin is not None:
            process.stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()

        if self._pending_read is not None:
            self._pending_read.cancel()
            with contextlib.suppress(asyncio.CancelledError, HelperProcessError):
                await self._pending_read
            self._pending_read = None
        if self._stderr_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
        self._process = None
        self._ready_message = None

    def _require_running(self) -> asyncio.subprocess.Process:
        if not self.running or self._process is None:
            raise self._process_exit_error("helper is not running")
        return self._process

    def _process_exit_error(self, message: str) -> HelperProcessError:
        returncode = self._process.returncode if self._process is not None else None
        suffix = f" (exit code {returncode})" if returncode is not None else ""
        return HelperProcessError(f"{self._name} {message}{suffix}{self._format_stderr_context()}")

    def _format_stderr_context(self) -> str:
        if not self._stderr_tail:
            return ""
        return "; recent stderr: " + " | ".join(self._stderr_tail)
