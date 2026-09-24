from __future__ import annotations

import asyncio
import contextlib
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from livekit import rtc

from ._helper_process import HelperProcess, HelperProcessError, HelperProtocolError
from .log import logger

_SAMPLE_RATE = 16000


class STT(stt.STT):
    """Batch Parakeet STT backed by a persistent ExecuTorch helper."""

    def __init__(
        self,
        *,
        helper_path: str | Path,
        model_path: str | Path,
        tokenizer_path: str | Path,
        delegate_data_path: str | Path | None = None,
        language: str = "en",
        ready_timeout: float = 120.0,
        _helper: HelperProcess | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self._model_path = str(model_path)
        self._language = LanguageCode(language)
        argv = [f"--model_path={model_path}", f"--tokenizer_path={tokenizer_path}"]
        if delegate_data_path is not None:
            argv.append(f"--data_path={delegate_data_path}")
        self._helper = _helper or HelperProcess(
            str(helper_path), argv, name="parakeet", ready_timeout=ready_timeout
        )
        self._recognize_lock = asyncio.Lock()
        self._prewarm_task: asyncio.Task[dict[str, Any]] | None = None

    @property
    def model(self) -> str:
        return self._model_path

    @property
    def provider(self) -> str:
        return "ExecuTorch"

    def prewarm(self) -> None:
        if self._prewarm_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._prewarm_task = loop.create_task(self._helper.start())

    async def start(self) -> None:
        """Start the Parakeet helper and wait for its ready message."""
        await self._ensure_ready()

    async def _ensure_ready(self) -> None:
        if self._prewarm_task is not None:
            task, self._prewarm_task = self._prewarm_task, None
            await task
        else:
            await self._helper.start()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        async with self._recognize_lock:
            request_id = utils.shortuuid("parakeet_")
            payload = _audio_buffer_to_f32le(buffer)
            message = {
                "type": "transcribe",
                "version": 1,
                "request_id": request_id,
                "audio": {
                    "encoding": "f32le",
                    "sample_rate": _SAMPLE_RATE,
                    "channel_count": 1,
                    "payload_byte_count": len(payload),
                },
                "enable_runtime_profile": False,
            }
            try:
                await self._ensure_ready()
                response = await asyncio.wait_for(
                    self._request(message, payload, request_id), timeout=conn_options.timeout
                )
            except asyncio.CancelledError:
                await asyncio.shield(self._helper.aclose(graceful=False))
                raise
            except TimeoutError:
                await self._helper.aclose(graceful=False)
                raise APITimeoutError("Parakeet transcription timed out") from None
            except HelperProtocolError as exc:
                await self._helper.aclose(graceful=False)
                raise APIError(str(exc), retryable=False) from exc
            except HelperProcessError as exc:
                await self._helper.aclose(graceful=False)
                raise APIConnectionError("Parakeet helper failed") from exc

            transcript_language = LanguageCode(language) if is_given(language) else self._language
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(
                        language=transcript_language,
                        text=response["text"],
                        metadata={
                            "provider": self.provider,
                            "model": self.model,
                            "runtime": "parakeet_helper",
                        },
                    )
                ],
            )

    async def _request(
        self, message: dict[str, Any], payload: bytes, request_id: str
    ) -> dict[str, Any]:
        await self._helper.write_message(message, payload)
        while True:
            response, response_payload = await self._helper.read_message()
            if response_payload is not None:
                raise HelperProtocolError("Parakeet response must not contain a binary payload")
            if response.get("version") != 1:
                raise HelperProtocolError("Parakeet response has unsupported protocol version")
            if response.get("request_id") != request_id:
                raise HelperProtocolError("Parakeet response request_id does not match")
            response_type = response.get("type")
            if response_type == "status":
                logger.debug("Parakeet status: %s", response.get("message", response.get("phase")))
                continue
            if response_type == "result":
                _required_string(response, "text")
                return response
            if response_type == "error":
                details = response.get("details")
                error_message = str(response.get("message", "Parakeet transcription failed"))
                if details:
                    error_message = f"{error_message}: {details}"
                raise APIError(error_message, body=response, retryable=False)
            raise HelperProtocolError(f"unexpected Parakeet response type: {response_type!r}")

    async def aclose(self) -> None:
        if self._prewarm_task is not None:
            if not self._prewarm_task.done():
                self._prewarm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, HelperProcessError):
                await self._prewarm_task
            self._prewarm_task = None
        await self._helper.aclose()


def _required_string(message: dict[str, Any], key: str) -> str:
    value = message.get(key)
    if not isinstance(value, str):
        raise HelperProtocolError(f"Parakeet response field {key!r} must be a string")
    return value


def _audio_buffer_to_f32le(buffer: utils.AudioBuffer) -> bytes:
    frame = rtc.combine_audio_frames(buffer)
    if frame.samples_per_channel == 0:
        return b""

    mono_samples = _downmix_s16(frame)
    mono_frame = rtc.AudioFrame(
        data=_s16le_bytes(mono_samples),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=len(mono_samples),
    )
    if mono_frame.sample_rate != _SAMPLE_RATE:
        resampler = rtc.AudioResampler(
            input_rate=mono_frame.sample_rate,
            output_rate=_SAMPLE_RATE,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        frames = [*resampler.push(mono_frame), *resampler.flush()]
        mono_frame = rtc.combine_audio_frames(frames)

    samples = array("h")
    samples.frombytes(mono_frame.data.tobytes())
    if sys.byteorder != "little":
        samples.byteswap()
    floats = array("f", (sample / 32768.0 for sample in samples))
    if sys.byteorder != "little":
        floats.byteswap()
    return floats.tobytes()


def _downmix_s16(frame: rtc.AudioFrame) -> Sequence[int]:
    samples = array("h")
    samples.frombytes(frame.data.tobytes())
    if sys.byteorder != "little":
        samples.byteswap()
    if frame.num_channels == 1:
        return samples
    channels = frame.num_channels
    return [
        sum(samples[index : index + channels]) // channels
        for index in range(0, len(samples), channels)
    ]


def _s16le_bytes(samples: Sequence[int]) -> bytes:
    output = array("h", samples)
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()
