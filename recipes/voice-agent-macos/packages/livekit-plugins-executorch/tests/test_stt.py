from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path

import pytest
from livekit.agents import APIConnectOptions, APIError, stt

from livekit import rtc
from livekit.plugins.executorch import STT
from livekit.plugins.executorch._helper_process import HelperProcess
from livekit.plugins.executorch.stt import _audio_buffer_to_f32le

pytestmark = pytest.mark.unit

_FAKE_HELPER = Path(__file__).with_name("fake_helper.py")


def provider(*, mode: str = "stt") -> STT:
    helper = HelperProcess(
        sys.executable,
        [str(_FAKE_HELPER)],
        name="fake-parakeet",
        ready_timeout=1.0,
        shutdown_timeout=0.05,
        terminate_timeout=0.05,
    )
    return STT(
        helper_path="unused",
        model_path="parakeet.pte",
        tokenizer_path="tokenizer.model",
        _helper=helper,
    )


def frame(samples: tuple[int, ...], *, sample_rate: int, channels: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=struct.pack(f"<{len(samples)}h", *samples),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=len(samples) // channels,
    )


def test_audio_conversion_downmixes_and_scales_s16() -> None:
    audio = frame((-32768, -32768, 16384, 16384, 32767, 32767), sample_rate=16000, channels=2)
    converted = struct.unpack("<3f", _audio_buffer_to_f32le(audio))
    assert converted == pytest.approx((-1.0, 0.5, 32767 / 32768))


def test_audio_conversion_resamples_to_16khz() -> None:
    audio = frame(tuple([1000] * 480), sample_rate=48000, channels=1)
    converted = _audio_buffer_to_f32le(audio)
    assert len(converted) // 4 == pytest.approx(160, abs=2)


async def test_recognize_builds_final_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stt")
    parakeet = provider()
    event = await parakeet.recognize(
        frame((-32768, 0, 16384, 32767), sample_rate=16000, channels=1),
        language="en-US",
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )
    assert event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id.startswith("parakeet_")
    assert event.alternatives[0].text == "-1.000,0.000,0.500,1.000"
    assert str(event.alternatives[0].language) == "en-US"
    assert event.alternatives[0].metadata == {
        "provider": "ExecuTorch",
        "model": "parakeet.pte",
        "runtime": "parakeet_helper",
    }
    await parakeet.aclose()


async def test_malformed_result_is_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stt_bad_result")
    parakeet = provider()
    with pytest.raises(APIError, match="field 'text' must be a string") as exc_info:
        await parakeet.recognize(
            frame((0,), sample_rate=16000, channels=1),
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        )
    assert not exc_info.value.retryable
    assert not parakeet._helper.running
    await parakeet.aclose()


async def test_helper_error_is_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stt_error")
    parakeet = provider()
    with pytest.raises(APIError, match="bad audio: fake failure") as exc_info:
        await parakeet.recognize(
            frame((0,), sample_rate=16000, channels=1),
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        )
    assert not exc_info.value.retryable
    await parakeet.aclose()


async def test_recognition_calls_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stt")
    parakeet = provider()
    audio = frame((0,), sample_rate=16000, channels=1)
    events = await asyncio.gather(
        parakeet.recognize(audio, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)),
        parakeet.recognize(audio, conn_options=APIConnectOptions(max_retry=0, timeout=1.0)),
    )
    assert len({event.request_id for event in events}) == 2
    assert all(event.alternatives[0].text == "0.000" for event in events)
    await parakeet.aclose()


async def test_cancellation_closes_uncancellable_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stt_slow")
    parakeet = provider()
    task = asyncio.create_task(
        parakeet.recognize(
            frame((0,), sample_rate=16000, channels=1),
            conn_options=APIConnectOptions(max_retry=0, timeout=60.0),
        )
    )
    await asyncio.sleep(0.05)
    assert parakeet._helper._process is not None
    old_pid = parakeet._helper._process.pid
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not parakeet._helper.running

    monkeypatch.setenv("FAKE_HELPER_MODE", "stt")
    event = await parakeet.recognize(
        frame((0,), sample_rate=16000, channels=1),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )
    assert parakeet._helper._process is not None
    assert parakeet._helper._process.pid != old_pid
    assert event.alternatives[0].text == "0.000"
    await parakeet.aclose()
