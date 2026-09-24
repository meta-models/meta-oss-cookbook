from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from livekit.agents import APIConnectOptions, APIError, APITimeoutError

from livekit.plugins.executorch import SupertonicTTS

pytestmark = pytest.mark.unit

_FAKE_RUNNER = Path(__file__).with_name("fake_supertonic_runner.py")


@pytest.fixture(autouse=True)
def executable_runner() -> None:
    _FAKE_RUNNER.chmod(0o755)


def provider(
    tmp_path: Path,
    *,
    ready_timeout: float = 1.0,
    shutdown_timeout: float = 0.05,
    terminate_timeout: float = 0.05,
) -> SupertonicTTS:
    pte = tmp_path / "supertonic.pte"
    voice = tmp_path / "F1.json"
    assets = tmp_path / "assets"
    pte.write_bytes(b"pte")
    voice.write_text("{}", encoding="utf-8")
    assets.mkdir(exist_ok=True)
    return SupertonicTTS(
        runner_path=_FAKE_RUNNER,
        pte_path=pte,
        asset_dir=assets,
        voice_style_path=voice,
        language="en",
        speed=1.05,
        seed=42,
        ready_timeout=ready_timeout,
        shutdown_timeout=shutdown_timeout,
        terminate_timeout=terminate_timeout,
    )


async def collect(supertonic: SupertonicTTS, text: str, *, timeout: float = 1.0):
    stream = supertonic.synthesize(
        text,
        conn_options=APIConnectOptions(max_retry=0, timeout=timeout),
    )
    return [event async for event in stream]


async def test_reuses_one_server_and_sends_text_only_over_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv_capture = tmp_path / "argv.txt"
    request_capture = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_SUPERTONIC_ARGV_CAPTURE", str(argv_capture))
    monkeypatch.setenv("FAKE_SUPERTONIC_REQUEST_CAPTURE", str(request_capture))
    text = "hello; touch /tmp/should-not-run && $(false)"
    supertonic = provider(tmp_path)

    first = await collect(supertonic, text)
    assert supertonic._process is not None
    pid = supertonic._process.pid
    second = await collect(supertonic, "second request")

    assert first and second
    assert supertonic._process is not None and supertonic._process.pid == pid
    argv = argv_capture.read_text(encoding="utf-8").splitlines()
    assert "--server_jsonl=true" in argv
    assert all(not argument.startswith("--text") for argument in argv)
    assert text not in "\n".join(argv)
    requests = [json.loads(line) for line in request_capture.read_text().splitlines()]
    assert [request["text"] for request in requests] == [text, "second request"]
    assert [request["id"] for request in requests] == [1, 2]
    await supertonic.aclose()
    assert not supertonic.running


async def test_oversized_request_does_not_poison_server(tmp_path: Path) -> None:
    supertonic = provider(tmp_path)

    with pytest.raises(APIError, match="JSONL size limit") as exc_info:
        await collect(supertonic, "x" * (64 * 1024))

    assert not exc_info.value.retryable
    assert not supertonic._request_active
    assert await collect(supertonic, "small request")
    await supertonic.aclose()


async def test_emits_44100_hz_pcm_without_wav_header(tmp_path: Path) -> None:
    supertonic = provider(tmp_path)
    events = await collect(supertonic, "hello")
    payload = b"".join(event.frame.data.tobytes() for event in events)

    assert payload
    assert not payload.startswith(b"RIFF")
    assert events[0].request_id.startswith("supertonic_")
    assert events[0].frame.sample_rate == 44100
    assert events[0].frame.num_channels == 1
    assert events[-1].is_final
    await supertonic.aclose()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing", "missing"),
        ("malformed", "invalid audio"),
        ("stereo", "mono"),
        ("wrong_rate", "44100 Hz"),
        ("wrong_width", "PCM16"),
    ],
)
async def test_rejects_invalid_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", mode)
    supertonic = provider(tmp_path)
    with pytest.raises(APIError, match=message) as exc_info:
        await collect(supertonic, "hello")
    assert not exc_info.value.retryable
    await supertonic.aclose()


async def test_runner_error_includes_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "error")
    supertonic = provider(tmp_path)
    with pytest.raises(APIError, match="voice style is invalid") as exc_info:
        await collect(supertonic, "hello")
    assert not exc_info.value.retryable
    await supertonic.aclose()


async def test_ready_timeout_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "ready_timeout")
    supertonic = provider(tmp_path, ready_timeout=0.02)
    with pytest.raises(APITimeoutError):
        await collect(supertonic, "hello")
    assert not supertonic.running
    await supertonic.aclose()


async def test_cancellation_during_startup_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "ready_timeout")
    supertonic = provider(tmp_path, ready_timeout=60.0)
    stream = supertonic.synthesize(
        "hello", conn_options=APIConnectOptions(max_retry=0, timeout=60.0)
    )

    await asyncio.sleep(0.05)
    await stream.aclose()

    assert not supertonic.running
    assert supertonic._process is None
    await supertonic.aclose()


async def test_synthesis_timeout_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "ignore_terminate")
    supertonic = provider(tmp_path, terminate_timeout=0.01)
    with pytest.raises(APITimeoutError):
        await collect(supertonic, "hello", timeout=0.02)
    assert not supertonic.running
    await supertonic.aclose()


async def test_stream_cancellation_terminates_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "sleep")
    supertonic = provider(tmp_path)
    stream = supertonic.synthesize(
        "hello", conn_options=APIConnectOptions(max_retry=0, timeout=60.0)
    )

    await asyncio.sleep(0.05)
    await stream.aclose()

    assert not supertonic.running
    await supertonic.aclose()


async def test_aclose_terminates_active_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "sleep")
    supertonic = provider(tmp_path)
    stream = supertonic.synthesize(
        "hello", conn_options=APIConnectOptions(max_retry=0, timeout=60.0)
    )
    task = asyncio.create_task(anext(stream))

    await asyncio.sleep(0.05)
    await supertonic.aclose()
    with pytest.raises(APIError):
        await task
    await stream.aclose()
    assert not supertonic.running


async def test_synthesis_calls_are_serialized_and_share_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request_capture = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_SUPERTONIC_REQUEST_CAPTURE", str(request_capture))
    supertonic = provider(tmp_path)

    results = await asyncio.gather(collect(supertonic, "one"), collect(supertonic, "two"))

    assert all(result for result in results)
    assert supertonic.running
    assert len(request_capture.read_text().splitlines()) == 2
    await supertonic.aclose()


async def test_shutdown_escalates_for_unresponsive_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "ignore_shutdown")
    supertonic = provider(tmp_path)
    await collect(supertonic, "hello")
    await asyncio.wait_for(supertonic.aclose(), timeout=1.0)
    assert not supertonic.running


async def test_protocol_mismatch_terminates_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_SUPERTONIC_MODE", "wrong_id")
    supertonic = provider(tmp_path)
    with pytest.raises(APIError, match="response id"):
        await collect(supertonic, "hello")
    assert not supertonic.running
    await supertonic.aclose()


def test_constructor_validates_paths_and_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runner_path"):
        SupertonicTTS(
            runner_path=tmp_path / "missing",
            pte_path=tmp_path / "missing.pte",
            asset_dir=tmp_path,
            voice_style_path=tmp_path / "missing.json",
        )

    runner = tmp_path / "runner"
    pte = tmp_path / "model.pte"
    voice = tmp_path / "voice.json"
    for path in (runner, pte, voice):
        path.write_bytes(b"x")
    runner.chmod(0o755)
    with pytest.raises(ValueError, match="speed"):
        SupertonicTTS(
            runner_path=runner,
            pte_path=pte,
            asset_dir=tmp_path,
            voice_style_path=voice,
            speed=0.0,
        )


def test_runner_can_be_python_interpreter_for_fake_protocol(tmp_path: Path) -> None:
    assert Path(sys.executable).is_file()
