from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from livekit.agents import llm, stt, tts, vad
from livekit.plugins import executorch, openai

LIVEKIT_URL = "ws://127.0.0.1:7880"
MUSE_GLIMMER_BASE_URL = "http://127.0.0.1:8000/v1"
REASONING_STRENGTH = "low"


@dataclass(frozen=True, slots=True)
class GlimmerConfig:
    agent_name: str
    instructions: str
    language: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    parakeet_helper_path: Path
    parakeet_model_path: Path
    parakeet_tokenizer_path: Path
    parakeet_delegate_data_path: Path | None
    muse_glimmer_base_url: str
    muse_glimmer_model_id: str
    muse_glimmer_api_key: str
    muse_glimmer_temperature: float
    muse_glimmer_max_tokens: int
    supertonic_runner_path: Path
    supertonic_pte_path: Path
    supertonic_asset_dir: Path
    supertonic_voice_style_path: Path
    supertonic_speed: float
    supertonic_seed: int

    @classmethod
    def from_env(cls) -> GlimmerConfig:
        return cls(
            agent_name=_env("GLIMMER_AGENT_NAME", "assistant"),
            instructions=_env(
                "GLIMMER_AGENT_INSTRUCTIONS",
                "You are Glimmer, a concise and friendly voice assistant. "
                "Answer naturally for speech. Do not use markdown, emoji, or long lists.",
            ),
            language=_env("GLIMMER_LANGUAGE", "en"),
            livekit_url=_exact_env("LIVEKIT_URL", LIVEKIT_URL),
            livekit_api_key=_required_env("LIVEKIT_API_KEY"),
            livekit_api_secret=_required_env("LIVEKIT_API_SECRET"),
            parakeet_helper_path=_required_file("PARAKEET_HELPER_PATH", executable=True),
            parakeet_model_path=_required_file("PARAKEET_MODEL_PATH"),
            parakeet_tokenizer_path=_required_file("PARAKEET_TOKENIZER_PATH"),
            parakeet_delegate_data_path=_optional_file("PARAKEET_DELEGATE_DATA_PATH"),
            muse_glimmer_base_url=_exact_env("MUSE_GLIMMER_BASE_URL", MUSE_GLIMMER_BASE_URL),
            muse_glimmer_model_id=_env(
                "MUSE_GLIMMER_MODEL_ID",
                "muse-glimmer-k-quant-17G-128K-text-dflash-metal",
            ),
            muse_glimmer_api_key=_env("MUSE_GLIMMER_API_KEY", "local"),
            muse_glimmer_temperature=_finite_float("MUSE_GLIMMER_TEMPERATURE", 0.0),
            muse_glimmer_max_tokens=_positive_int("MUSE_GLIMMER_MAX_TOKENS", 256),
            supertonic_runner_path=_required_file("SUPERTONIC_RUNNER_PATH", executable=True),
            supertonic_pte_path=_required_file("SUPERTONIC_PTE_PATH"),
            supertonic_asset_dir=_required_directory("SUPERTONIC_ASSET_DIR"),
            supertonic_voice_style_path=_required_file("SUPERTONIC_VOICE_STYLE_PATH"),
            supertonic_speed=_positive_float("SUPERTONIC_SPEED", 1.05),
            supertonic_seed=_non_negative_int("SUPERTONIC_SEED", 42),
        )


@dataclass(frozen=True, slots=True)
class LocalProviders:
    parakeet: executorch.STT
    llm: llm.LLM
    supertonic: executorch.SupertonicTTS


@dataclass(frozen=True, slots=True)
class Providers:
    parakeet: executorch.STT
    session_stt: stt.STT
    llm: llm.LLM
    supertonic: executorch.SupertonicTTS
    session_tts: tts.TTS


def create_local_providers(config: GlimmerConfig) -> LocalProviders:
    parakeet = executorch.STT(
        helper_path=config.parakeet_helper_path,
        model_path=config.parakeet_model_path,
        tokenizer_path=config.parakeet_tokenizer_path,
        delegate_data_path=config.parakeet_delegate_data_path,
        language=config.language,
    )
    supertonic = executorch.SupertonicTTS(
        runner_path=config.supertonic_runner_path,
        pte_path=config.supertonic_pte_path,
        asset_dir=config.supertonic_asset_dir,
        voice_style_path=config.supertonic_voice_style_path,
        language=config.language,
        speed=config.supertonic_speed,
        seed=config.supertonic_seed,
    )
    muse_glimmer = openai.LLM(
        model=config.muse_glimmer_model_id,
        api_key=config.muse_glimmer_api_key,
        base_url=config.muse_glimmer_base_url,
        temperature=config.muse_glimmer_temperature,
        max_completion_tokens=config.muse_glimmer_max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "reasoning_strength": REASONING_STRENGTH,
            },
        },
    )
    return LocalProviders(parakeet=parakeet, llm=muse_glimmer, supertonic=supertonic)


def create_providers(config: GlimmerConfig, *, voice_activity_detector: vad.VAD) -> Providers:
    local = create_local_providers(config)
    return Providers(
        parakeet=local.parakeet,
        session_stt=stt.StreamAdapter(stt=local.parakeet, vad=voice_activity_detector),
        llm=local.llm,
        supertonic=local.supertonic,
        session_tts=local.supertonic,
    )


def _env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set and non-empty")
    return value


def _exact_env(name: str, expected: str) -> str:
    value = _env(name, expected)
    if value != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _required_file(name: str, *, executable: bool = False) -> Path:
    value = _optional_env(name)
    if value is None:
        raise ValueError(f"{name} must be set")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{name} must point to a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{name} must point to an executable file: {path}")
    return path


def _required_directory(name: str) -> Path:
    value = _optional_env(name)
    if value is None:
        raise ValueError(f"{name} must be set")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{name} must point to a directory: {path}")
    return path


def _optional_file(name: str) -> Path | None:
    if _optional_env(name) is None:
        return None
    return _required_file(name)


def _finite_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_float(name: str, default: float) -> float:
    value = _finite_float(name, default)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(name: str, default: int) -> int:
    return _bounded_int(name, default, minimum=1)


def _non_negative_int(name: str, default: int) -> int:
    return _bounded_int(name, default, minimum=0)


def _bounded_int(name: str, default: int, *, minimum: int) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        constraint = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {constraint}")
    return value
