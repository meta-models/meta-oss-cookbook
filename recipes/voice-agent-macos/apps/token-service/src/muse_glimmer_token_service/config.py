from __future__ import annotations

from typing import ClassVar, Final

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVEKIT_SERVER_URL: Final = "ws://127.0.0.1:7880"
ALLOWED_WEB_ORIGINS: Final = ("http://127.0.0.1:5173",)


class Settings(BaseSettings):
    """Environment-only credentials and bounded token lifetime."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=True,
        extra="ignore",
        frozen=True,
    )

    livekit_api_key: SecretStr = Field(validation_alias="LIVEKIT_API_KEY", min_length=1)
    livekit_api_secret: SecretStr = Field(validation_alias="LIVEKIT_API_SECRET", min_length=1)
    livekit_url: str = Field(
        default=LIVEKIT_SERVER_URL,
        validation_alias="LIVEKIT_URL",
    )
    token_ttl_seconds: int = Field(
        default=600,
        validation_alias="TOKEN_TTL_SECONDS",
        ge=60,
        le=3600,
    )
    allowed_web_origins: ClassVar[tuple[str, ...]] = ALLOWED_WEB_ORIGINS

    @field_validator("livekit_api_key", "livekit_api_secret", mode="before")
    @classmethod
    def strip_credential(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("LiveKit credentials must be non-empty")
        return value

    @field_validator("livekit_url")
    @classmethod
    def require_local_livekit_url(cls, value: str) -> str:
        if value != LIVEKIT_SERVER_URL:
            raise ValueError(f"LIVEKIT_URL must be exactly {LIVEKIT_SERVER_URL}")
        return value


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
