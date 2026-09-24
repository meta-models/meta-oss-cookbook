from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from muse_glimmer_worker import config as config_module


def _set_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "local-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "local-secret")
    for name in (
        "PARAKEET_HELPER_PATH",
        "PARAKEET_MODEL_PATH",
        "PARAKEET_TOKENIZER_PATH",
        "SUPERTONIC_RUNNER_PATH",
        "SUPERTONIC_PTE_PATH",
        "SUPERTONIC_VOICE_STYLE_PATH",
    ):
        path = tmp_path / name.lower()
        path.write_bytes(b"test")
        if name.endswith(("HELPER_PATH", "RUNNER_PATH")):
            path.chmod(0o755)
        monkeypatch.setenv(name, str(path))
    assets = tmp_path / "supertonic-assets"
    assets.mkdir()
    monkeypatch.setenv("SUPERTONIC_ASSET_DIR", str(assets))


def test_config_accepts_only_fixed_local_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required(monkeypatch, tmp_path)
    config = config_module.GlimmerConfig.from_env()
    assert config.livekit_url == "ws://127.0.0.1:7880"
    assert config.muse_glimmer_base_url == "http://127.0.0.1:8000/v1"
    assert config.muse_glimmer_max_tokens == 256

    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    with pytest.raises(ValueError, match="must be exactly"):
        config_module.GlimmerConfig.from_env()
    monkeypatch.setenv("LIVEKIT_URL", config_module.LIVEKIT_URL)
    monkeypatch.setenv("MUSE_GLIMMER_BASE_URL", "https://example.com/v1")
    with pytest.raises(ValueError, match="must be exactly"):
        config_module.GlimmerConfig.from_env()


def test_config_requires_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required(monkeypatch, tmp_path)
    monkeypatch.delenv("LIVEKIT_API_SECRET")
    with pytest.raises(ValueError, match="LIVEKIT_API_SECRET"):
        config_module.GlimmerConfig.from_env()


def test_llm_uses_reasoning_strength_low_without_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required(monkeypatch, tmp_path)
    captured: dict[str, Any] = {}

    class FakeSTT:
        def __init__(self, **kwargs: object) -> None:
            captured["stt"] = kwargs

    class FakeTTS:
        def __init__(self, **kwargs: object) -> None:
            captured["tts"] = kwargs

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured["llm"] = kwargs

    monkeypatch.setattr(config_module.executorch, "STT", FakeSTT)
    monkeypatch.setattr(config_module.executorch, "SupertonicTTS", FakeTTS)
    monkeypatch.setattr(config_module.openai, "LLM", FakeLLM)
    config_module.create_local_providers(config_module.GlimmerConfig.from_env())

    llm_options = captured["llm"]
    assert llm_options["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "low"}}
    assert "reasoning_effort" not in llm_options
    assert "reasoning_effort" not in repr(llm_options)


def test_environment_cannot_override_reasoning_strength(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_required(monkeypatch, tmp_path)
    monkeypatch.setenv("MUSE_GLIMMER_REASONING_STRENGTH", "high")
    assert config_module.REASONING_STRENGTH == "low"
    assert "MUSE_GLIMMER_REASONING_STRENGTH" in os.environ
