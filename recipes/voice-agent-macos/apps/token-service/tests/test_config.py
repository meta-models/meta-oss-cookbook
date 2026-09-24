from __future__ import annotations

from pathlib import Path

import pytest
from muse_glimmer_token_service.config import ALLOWED_WEB_ORIGINS, Settings
from pydantic import SecretStr, ValidationError

_SECRET = "test-secret-with-at-least-thirty-two-characters"


def _values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "LIVEKIT_API_KEY": "test-key",
        "LIVEKIT_API_SECRET": _SECRET,
    }
    values.update(overrides)
    return values


def test_settings_read_credentials_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "  environment-key  ")
    monkeypatch.setenv("LIVEKIT_API_SECRET", f"  {_SECRET}  ")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.livekit_api_key.get_secret_value() == "environment-key"
    assert settings.livekit_api_secret.get_secret_value() == _SECRET
    assert settings.livekit_url == "ws://127.0.0.1:7880"
    assert settings.allowed_web_origins == ("http://127.0.0.1:5173",)


def test_dotenv_file_is_never_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    (tmp_path / ".env").write_text(
        f"LIVEKIT_API_KEY=dotenv-key\nLIVEKIT_API_SECRET={_SECRET}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


@pytest.mark.parametrize("field", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"])
def test_blank_credentials_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**_values(**{field: "   "}))  # type: ignore[arg-type]


def test_credentials_are_redacted_from_settings_representation() -> None:
    settings = Settings(
        **_values(
            LIVEKIT_API_KEY=SecretStr("test-key"),
            LIVEKIT_API_SECRET=SecretStr(_SECRET),
        )
    )

    assert "test-key" not in repr(settings)
    assert _SECRET not in repr(settings)


@pytest.mark.parametrize(
    "url",
    [
        "ws://localhost:7880",
        "ws://127.0.0.1:7880/",
        "ws://127.0.0.1:7880/path",
        "ws://127.0.0.1:7880?query=yes",
        "wss://127.0.0.1:7880",
        "wss://example.livekit.cloud",
    ],
)
def test_livekit_url_variations_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError, match="must be exactly"):
        Settings(**_values(LIVEKIT_URL=url))


@pytest.mark.parametrize("ttl", [60, 3600])
def test_token_ttl_boundaries_are_allowed(ttl: int) -> None:
    assert Settings(**_values(TOKEN_TTL_SECONDS=ttl)).token_ttl_seconds == ttl


@pytest.mark.parametrize("ttl", [59, 3601])
def test_token_ttl_outside_bounds_is_rejected(ttl: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**_values(TOKEN_TTL_SECONDS=ttl))


def test_allowed_origins_cannot_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://attacker.example")

    settings = Settings(**_values())

    assert settings.allowed_web_origins == ALLOWED_WEB_ORIGINS
