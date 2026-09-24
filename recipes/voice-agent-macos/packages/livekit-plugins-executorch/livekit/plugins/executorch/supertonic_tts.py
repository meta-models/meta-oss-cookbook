from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import tempfile
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

_SAMPLE_RATE = 44100
_NUM_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2
_MAX_JSON_LINE_BYTES = 64 * 1024
_MAX_STDERR_LINES = 20


class _ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class _SupertonicOptions:
    runner_path: str
    pte_path: str
    asset_dir: str
    voice_style_path: str
    language: str
    speed: float
    seed: int


class SupertonicTTS(tts.TTS):
    """Batch TTS backed by one persistent Supertonic JSONL server process."""

    def __init__(
        self,
        *,
        runner_path: str | Path,
        pte_path: str | Path,
        asset_dir: str | Path,
        voice_style_path: str | Path,
        language: str = "en",
        speed: float = 1.05,
        seed: int = 42,
        ready_timeout: float = 120.0,
        shutdown_timeout: float = 2.0,
        terminate_timeout: float = 2.0,
    ) -> None:
        runner = _required_file(runner_path, "runner_path", executable=True)
        pte = _required_file(pte_path, "pte_path")
        assets = _required_directory(asset_dir, "asset_dir")
        voice_style = _required_file(voice_style_path, "voice_style_path")
        if not language.strip():
            raise ValueError("language must be non-empty")
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, value in (
            ("ready_timeout", ready_timeout),
            ("shutdown_timeout", shutdown_timeout),
            ("terminate_timeout", terminate_timeout),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._opts = _SupertonicOptions(
            runner_path=str(runner),
            pte_path=str(pte),
            asset_dir=str(assets),
            voice_style_path=str(voice_style),
            language=language.strip(),
            speed=speed,
            seed=seed,
        )
        self._ready_timeout = ready_timeout
        self._shutdown_timeout = shutdown_timeout
        self._terminate_timeout = terminate_timeout
        self._synthesis_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_MAX_STDERR_LINES)
        self._request_active = False
        self._next_request_id = 1
        self._closed = False

    @property
    def model(self) -> str:
        return self._opts.pte_path

    @property
    def provider(self) -> str:
        return "ExecuTorch Supertonic"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        self._closed = True
        async with self._lifecycle_lock:
            process = self._process
            if process is None:
                return
            if self._request_active:
                await self._stop_locked(process, graceful=False)
                return
            try:
                await self._write_json(process, {"type": "shutdown"})
                response = await asyncio.wait_for(
                    self._read_json(process), timeout=self._shutdown_timeout
                )
                if response != {"type": "stopped"}:
                    raise _ProtocolError(f"Supertonic sent invalid shutdown response: {response!r}")
                if not await _wait_for_exit(process, self._shutdown_timeout):
                    raise TimeoutError
            except (TimeoutError, BrokenPipeError, ConnectionResetError, _ProtocolError):
                await self._stop_locked(process, graceful=False)
            else:
                await self._clear_process_locked(process)

    def _command(self) -> tuple[str, ...]:
        return (
            self._opts.runner_path,
            "--server_jsonl=true",
            f"--pte={self._opts.pte_path}",
            f"--asset_dir={self._opts.asset_dir}",
            f"--voice_style={self._opts.voice_style_path}",
            f"--language={self._opts.language}",
            f"--speed={self._opts.speed}",
            f"--seed={self._opts.seed}",
        )

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        async with self._lifecycle_lock:
            if self._closed:
                raise APIConnectionError("Supertonic TTS is closed", retryable=False)
            if self.running and self._process is not None:
                return self._process
            if self._process is not None:
                await self._stop_locked(self._process, graceful=False)

            self._stderr_tail.clear()
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=_MAX_JSON_LINE_BYTES + 1,
                )
            except OSError as exc:
                raise APIConnectionError(f"failed to start Supertonic runner: {exc}") from exc
            self._process = process
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(process), name="supertonic-stderr"
            )
            try:
                ready = await asyncio.wait_for(
                    self._read_json(process), timeout=self._ready_timeout
                )
                _validate_ready(ready)
            except asyncio.CancelledError:
                await asyncio.shield(self._stop_locked(process, graceful=False))
                raise
            except TimeoutError:
                await self._stop_locked(process, graceful=False)
                raise APITimeoutError(
                    self._with_stderr(
                        f"Supertonic did not become ready within {self._ready_timeout:.1f}s"
                    )
                ) from None
            except (OSError, _ProtocolError) as exc:
                await self._stop_locked(process, graceful=False)
                raise APIConnectionError(self._with_stderr(str(exc))) from exc
            if self._closed:
                await self._stop_locked(process, graceful=False)
                raise APIConnectionError("Supertonic TTS is closed", retryable=False)
            return process

    async def _request(self, text: str, output_path: Path, timeout: float) -> None:
        process = await self._ensure_started()
        request_id = self._next_request_id
        self._next_request_id += 1
        request = {
            "type": "synthesize",
            "id": request_id,
            "text": text,
            "output": str(output_path),
        }
        async with self._lifecycle_lock:
            if process is not self._process or process.returncode is not None or self._closed:
                raise APIConnectionError("Supertonic runner is unavailable")
            self._request_active = True
            try:
                await self._write_json(process, request)
            except asyncio.CancelledError:
                self._request_active = False
                await asyncio.shield(self._stop_locked(process, graceful=False))
                raise
            except _ProtocolError as exc:
                self._request_active = False
                raise APIError(str(exc), retryable=False) from exc
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                self._request_active = False
                await self._stop_locked(process, graceful=False)
                raise APIConnectionError(
                    self._with_stderr("Supertonic request write failed")
                ) from exc

        try:
            response = await asyncio.wait_for(self._read_json(process), timeout=timeout)
            _validate_response(response, request_id, output_path)
        except TimeoutError:
            await asyncio.shield(self._stop(process))
            raise APITimeoutError(self._with_stderr("Supertonic synthesis timed out")) from None
        except asyncio.CancelledError:
            await asyncio.shield(self._stop(process))
            raise
        except _ProtocolError as exc:
            await self._stop(process)
            raise APIError(self._with_stderr(str(exc)), retryable=False) from exc
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            await self._stop(process)
            raise APIConnectionError(self._with_stderr("Supertonic runner failed")) from exc
        finally:
            async with self._lifecycle_lock:
                self._request_active = False

        if response["type"] == "error":
            raise APIError(str(response["message"]), body=response, retryable=False)

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        async with self._lifecycle_lock:
            await self._stop_locked(process, graceful=False)

    async def _stop_locked(self, process: asyncio.subprocess.Process, *, graceful: bool) -> None:
        if process.returncode is None and graceful:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                await self._write_json(process, {"type": "shutdown"})
            await _wait_for_exit(process, self._shutdown_timeout)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            if not await _wait_for_exit(process, self._terminate_timeout):
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                if not await _wait_for_exit(process, self._terminate_timeout):
                    raise RuntimeError("Supertonic runner did not exit after SIGKILL")
        await self._clear_process_locked(process)

    async def _clear_process_locked(self, process: asyncio.subprocess.Process) -> None:
        stderr_task, self._stderr_task = self._stderr_task, None
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
        if process.stdin is not None:
            process.stdin.close()
        if self._process is process:
            self._process = None
        await asyncio.sleep(0)

    async def _write_json(
        self, process: asyncio.subprocess.Process, message: dict[str, object]
    ) -> None:
        if process.stdin is None:
            raise OSError("Supertonic stdin is unavailable")
        encoded = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_JSON_LINE_BYTES:
            raise _ProtocolError("Supertonic request exceeds the JSONL size limit")
        process.stdin.write(encoded + b"\n")
        await process.stdin.drain()

    async def _read_json(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        if process.stdout is None:
            raise OSError("Supertonic stdout is unavailable")
        try:
            line = await process.stdout.readline()
        except ValueError as exc:
            raise _ProtocolError("Supertonic response exceeds the JSONL size limit") from exc
        if not line:
            returncode = await process.wait()
            raise _ProtocolError(f"Supertonic runner exited unexpectedly with code {returncode}")
        if not line.endswith(b"\n") or len(line) - 1 > _MAX_JSON_LINE_BYTES:
            raise _ProtocolError("Supertonic response exceeds the JSONL size limit")
        try:
            response = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite number {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise _ProtocolError("Supertonic sent invalid JSON") from exc
        if not isinstance(response, dict):
            raise _ProtocolError("Supertonic response must be a JSON object")
        return response

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while line := await process.stderr.readline():
            self._stderr_tail.append(line.decode(errors="replace").rstrip())

    def _with_stderr(self, message: str) -> str:
        if not self._stderr_tail:
            return message
        return f"{message}; recent stderr: {' | '.join(self._stderr_tail)}"


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: SupertonicTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: SupertonicTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if not self._input_text:
            raise APIError("Supertonic synthesis text must be non-empty", retryable=False)
        async with self._tts._synthesis_lock:
            if self._tts._closed:
                raise APIConnectionError("Supertonic TTS is closed", retryable=False)
            with tempfile.TemporaryDirectory(prefix="livekit-supertonic-") as temporary:
                output_path = Path(temporary) / "speech.wav"
                await self._tts._request(self._input_text, output_path, self._conn_options.timeout)
                try:
                    payload = await asyncio.to_thread(_read_pcm_wav, output_path)
                except (OSError, EOFError, wave.Error, ValueError) as exc:
                    raise APIError(
                        f"Supertonic produced invalid audio: {exc}", retryable=False
                    ) from exc

                output_emitter.initialize(
                    request_id=utils.shortuuid("supertonic_"),
                    sample_rate=_SAMPLE_RATE,
                    num_channels=_NUM_CHANNELS,
                    mime_type="audio/pcm",
                    frame_size_ms=50,
                )
                output_emitter.push(payload)
                output_emitter.flush()


async def _wait_for_exit(process: asyncio.subprocess.Process, timeout: float) -> bool:
    if process.returncode is not None:
        return True
    wait_task = asyncio.create_task(process.wait())
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
        return True
    except TimeoutError:
        return process.returncode is not None
    finally:
        if not wait_task.done():
            wait_task.cancel()


def _validate_ready(response: dict[str, Any]) -> None:
    expected = {
        "type",
        "protocol_version",
        "sample_rate",
        "load_seconds",
        "warmup_seconds",
    }
    if set(response) != expected or response.get("type") != "ready":
        raise _ProtocolError(f"Supertonic sent invalid ready response: {response!r}")
    if response.get("protocol_version") != 1:
        raise _ProtocolError("Supertonic uses an unsupported protocol version")
    if response.get("sample_rate") != _SAMPLE_RATE:
        raise _ProtocolError(f"Supertonic must use {_SAMPLE_RATE} Hz")
    for field in ("load_seconds", "warmup_seconds"):
        value = response.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _ProtocolError(f"Supertonic ready field {field!r} must be numeric")
        if not math.isfinite(value) or value < 0.0:
            raise _ProtocolError(
                f"Supertonic ready field {field!r} must be finite and non-negative"
            )


def _validate_response(response: dict[str, Any], request_id: int, output_path: Path) -> None:
    response_type = response.get("type")
    if response_type == "error":
        if set(response) != {"type", "id", "message"}:
            raise _ProtocolError("Supertonic error response has unexpected fields")
        if response.get("id") != request_id or not isinstance(response.get("message"), str):
            raise _ProtocolError("Supertonic error response is invalid")
        return
    expected = {
        "type",
        "id",
        "output",
        "samples",
        "audio_seconds",
        "synthesis_seconds",
        "rtf",
    }
    if response_type != "result" or set(response) != expected:
        raise _ProtocolError(f"Supertonic sent invalid synthesis response: {response!r}")
    if response.get("id") != request_id:
        raise _ProtocolError("Supertonic response id does not match the request")
    if response.get("output") != str(output_path):
        raise _ProtocolError("Supertonic response output does not match the request")
    samples = response.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise _ProtocolError("Supertonic response samples must be a positive integer")
    for field in ("audio_seconds", "synthesis_seconds", "rtf"):
        value = response.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _ProtocolError(f"Supertonic response field {field!r} must be numeric")
        if not math.isfinite(value) or value < 0.0:
            raise _ProtocolError(
                f"Supertonic response field {field!r} must be finite and non-negative"
            )


def _required_file(value: str | Path, name: str, *, executable: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name} must point to a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{name} must point to an executable file: {path}")
    return path


def _required_directory(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{name} must point to a directory: {path}")
    return path


def _read_pcm_wav(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError("output WAV is missing")
    with wave.open(str(path), "rb") as output:
        channels = output.getnchannels()
        sample_rate = output.getframerate()
        sample_width = output.getsampwidth()
        compression = output.getcomptype()
        frame_count = output.getnframes()
        payload = output.readframes(frame_count)
    if compression != "NONE":
        raise ValueError("output WAV must be uncompressed PCM")
    if channels != _NUM_CHANNELS:
        raise ValueError("output WAV must be mono")
    if sample_rate != _SAMPLE_RATE:
        raise ValueError(f"output WAV must use {_SAMPLE_RATE} Hz")
    if sample_width != _SAMPLE_WIDTH_BYTES:
        raise ValueError("output WAV must use signed PCM16 samples")
    expected_bytes = frame_count * channels * sample_width
    if frame_count <= 0 or not payload:
        raise ValueError("output WAV contains no audio")
    if len(payload) != expected_bytes:
        raise ValueError("output WAV is truncated")
    return payload
