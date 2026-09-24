from __future__ import annotations

from ipaddress import ip_address
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings, get_settings
from .tokens import issue_connection


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"pragma", b"no-cache"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-content-type-options", b"nosniff"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class LoopbackClientMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not _is_loopback_client(scope.get("client")):
            response = JSONResponse({"detail": "Loopback clients only"}, status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _is_loopback_client(client: Any) -> bool:
    if not isinstance(client, tuple) or not client:
        return False
    try:
        return ip_address(client[0]).is_loopback
    except ValueError:
        return False


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class ConnectionResponse(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    server_url: str
    participant_token: str
    room_name: str
    participant_identity: str


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="Muse Glimmer local token service",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_web_origins),
        allow_credentials=False,
        allow_methods=["POST"],
        allow_headers=["Accept"],
    )
    app.add_middleware(LoopbackClientMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/token", response_model=ConnectionResponse, response_model_by_alias=True)
    async def token(origin: str | None = Header(default=None)) -> ConnectionResponse:
        if origin is not None and origin not in resolved_settings.allowed_web_origins:
            raise HTTPException(status_code=403, detail="Origin is not allowed")
        details = issue_connection(resolved_settings)
        return ConnectionResponse(
            server_url=details.server_url,
            participant_token=details.participant_token,
            room_name=details.room_name,
            participant_identity=details.participant_identity,
        )

    return app
