from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from muse_glimmer_worker import agent


class Closeable:
    def __init__(self) -> None:
        self.calls = 0

    async def aclose(self) -> None:
        self.calls += 1


class FailingParakeet(Closeable):
    async def start(self) -> None:
        raise RuntimeError("startup failed")


class FakeContext:
    def __init__(self) -> None:
        self.room = SimpleNamespace(name="room")
        self.proc = SimpleNamespace(userdata={agent._VAD_KEY: object()})
        self.log_context_fields = {}
        self.shutdown_callback = None

    def add_shutdown_callback(self, callback) -> None:
        self.shutdown_callback = callback


async def test_startup_failure_and_shutdown_callback_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parakeet = FailingParakeet()
    model = Closeable()
    supertonic = Closeable()
    providers = SimpleNamespace(
        parakeet=parakeet,
        llm=model,
        supertonic=supertonic,
        session_stt=object(),
        session_tts=object(),
    )
    monkeypatch.setattr(agent.GlimmerConfig, "from_env", lambda: SimpleNamespace(agent_name="a"))
    monkeypatch.setattr(agent, "create_providers", lambda *args, **kwargs: providers)
    context = FakeContext()

    with pytest.raises(RuntimeError, match="startup failed"):
        await agent.entrypoint(context)
    assert context.shutdown_callback is not None
    await context.shutdown_callback()

    assert (model.calls, parakeet.calls, supertonic.calls) == (1, 1, 1)


async def test_normal_shutdown_callback_cleanup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StartedParakeet(Closeable):
        async def start(self) -> None:
            return None

    class FakeSession:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, **kwargs) -> None:
            pass

        def on(self, event_name):
            return lambda callback: callback

        async def start(self, **kwargs) -> None:
            return None

    parakeet = StartedParakeet()
    model = Closeable()
    supertonic = Closeable()
    providers = SimpleNamespace(
        parakeet=parakeet,
        llm=model,
        supertonic=supertonic,
        session_stt=object(),
        session_tts=object(),
    )
    config = SimpleNamespace(agent_name="a", instructions="Answer briefly.")
    monkeypatch.setattr(agent.GlimmerConfig, "from_env", lambda: config)
    monkeypatch.setattr(agent, "create_providers", lambda *args, **kwargs: providers)
    monkeypatch.setattr(agent, "AgentSession", FakeSession)
    context = FakeContext()

    await agent.entrypoint(context)
    assert context.shutdown_callback is not None
    await context.shutdown_callback()
    await context.shutdown_callback()

    assert (model.calls, parakeet.calls, supertonic.calls) == (1, 1, 1)


def test_worker_is_loopback_only_with_neutral_agent_name() -> None:
    assert agent.server._host == "127.0.0.1"
    assert agent.server._agent_name == "assistant"


def test_onnx_runtime_telemetry_is_disabled_before_agent_import() -> None:
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
