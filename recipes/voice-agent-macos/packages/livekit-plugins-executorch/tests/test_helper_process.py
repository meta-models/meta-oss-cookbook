from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from livekit.plugins.executorch._helper_process import (
    HelperProcess,
    HelperProcessError,
    HelperProtocolError,
)

pytestmark = pytest.mark.unit

_FAKE_HELPER = Path(__file__).with_name("fake_helper.py")


def helper(*, ready_timeout: float = 1.0, max_payload_bytes: int = 1024) -> HelperProcess:
    return HelperProcess(
        sys.executable,
        [str(_FAKE_HELPER)],
        name="fake",
        ready_timeout=ready_timeout,
        shutdown_timeout=0.05,
        terminate_timeout=0.05,
        max_payload_bytes=max_payload_bytes,
    )


async def test_ready_write_read_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stt")
    process = helper()
    ready = await process.start()
    assert ready == {"type": "ready", "version": 1}

    payload = b"\x00\x00\x00\x00"
    await process.write_message(
        {
            "type": "transcribe",
            "version": 1,
            "request_id": "request-1",
            "audio": {
                "encoding": "f32le",
                "sample_rate": 16000,
                "channel_count": 1,
                "payload_byte_count": len(payload),
            },
        },
        payload,
    )
    status, status_payload = await process.read_message()
    result, result_payload = await process.read_message()
    assert status["type"] == "status"
    assert status_payload is None
    assert result["text"] == "0.000"
    assert result_payload is None

    await process.aclose()
    assert not process.running


async def test_startup_timeout_reports_recent_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "stderr_crash")
    process = helper()
    with pytest.raises(HelperProcessError, match="model load exploded"):
        await process.start()
    assert not process.running


async def test_startup_timeout_terminates_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "timeout")
    process = helper(ready_timeout=0.02)
    with pytest.raises(HelperProcessError, match="did not become ready"):
        await process.start()
    assert not process.running


async def test_shutdown_escalates_for_unresponsive_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "ignore_shutdown")
    process = helper()
    await process.start()
    await asyncio.wait_for(process.aclose(), timeout=1.0)
    assert not process.running


async def test_malformed_header_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "malformed_ready")
    process = helper()
    with pytest.raises(HelperProcessError, match="malformed JSON"):
        await process.start()


async def test_oversized_payload_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "oversized")
    process = helper(max_payload_bytes=32)
    await process.start()
    with pytest.raises(HelperProtocolError, match="payload exceeds"):
        await process.read_message()
    await process.aclose(graceful=False)


async def test_unexpected_eof_includes_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_HELPER_MODE", "eof")
    process = helper()
    await process.start()
    payload = b"\x00\x00\x00\x00"
    await process.write_message(
        {
            "type": "transcribe",
            "version": 1,
            "request_id": "request-1",
            "audio": {
                "encoding": "f32le",
                "sample_rate": 16000,
                "channel_count": 1,
                "payload_byte_count": len(payload),
            },
        },
        payload,
    )
    with pytest.raises(HelperProcessError, match="unexpected EOF"):
        await asyncio.wait_for(process.read_message(), 1.0)
    await process.aclose(graceful=False)
