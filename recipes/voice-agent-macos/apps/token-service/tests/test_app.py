from __future__ import annotations

import time

import jwt
import pytest
from fastapi.testclient import TestClient
from muse_glimmer_token_service.app import create_app
from muse_glimmer_token_service.config import ALLOWED_WEB_ORIGINS, Settings
from muse_glimmer_token_service.tokens import AGENT_NAME
from pydantic import SecretStr

_API_KEY = "test-key"
_SECRET = "test-secret-with-at-least-thirty-two-characters"


def _settings() -> Settings:
    return Settings(
        LIVEKIT_API_KEY=SecretStr(_API_KEY),
        LIVEKIT_API_SECRET=SecretStr(_SECRET),
        TOKEN_TTL_SECONDS=600,
    )


def _client(*, host: str = "127.0.0.1", client_host: str = "127.0.0.1") -> TestClient:
    return TestClient(
        create_app(_settings()),
        base_url=f"http://{host}",
        client=(client_host, 50000),
    )


def test_token_has_restricted_grants_fixed_dispatch_and_exact_response() -> None:
    with _client() as client:
        response = client.post("/api/token", headers={"Origin": ALLOWED_WEB_ORIGINS[0]})

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["access-control-allow-origin"] == ALLOWED_WEB_ORIGINS[0]
    body = response.json()
    assert set(body) == {
        "participantIdentity",
        "participantToken",
        "roomName",
        "serverUrl",
    }
    assert body["serverUrl"] == "ws://127.0.0.1:7880"
    assert body["roomName"].startswith("r_")
    assert body["participantIdentity"].startswith("p_")

    assert jwt.get_unverified_header(body["participantToken"])["alg"] == "HS256"
    claims = jwt.decode(
        body["participantToken"],
        _SECRET,
        algorithms=["HS256"],
        options={"verify_aud": False},
    )
    assert claims["iss"] == _API_KEY
    assert claims["sub"] == body["participantIdentity"]
    assert 590 <= claims["exp"] - int(time.time()) <= 600
    assert claims["video"] == {
        "canPublish": True,
        "canPublishData": False,
        "canPublishSources": ["microphone"],
        "canSubscribe": True,
        "room": body["roomName"],
        "roomJoin": True,
    }
    assert claims["roomConfig"] == {"agents": [{"agentName": AGENT_NAME}]}
    assert "admin" not in claims
    assert "recorder" not in claims


def test_each_request_gets_random_room_and_participant() -> None:
    with _client() as client:
        first = client.post("/api/token", headers={"Origin": ALLOWED_WEB_ORIGINS[0]}).json()
        second = client.post("/api/token", headers={"Origin": ALLOWED_WEB_ORIGINS[0]}).json()

    assert first["roomName"] != second["roomName"]
    assert first["participantIdentity"] != second["participantIdentity"]


def test_client_cannot_choose_room_or_agent() -> None:
    with _client() as client:
        response = client.post(
            "/api/token",
            headers={"Origin": ALLOWED_WEB_ORIGINS[0]},
            json={"roomName": "attacker-room", "agentName": "other-agent"},
        )

    assert response.status_code == 200
    assert response.json()["roomName"] != "attacker-room"


@pytest.mark.parametrize("origin", ALLOWED_WEB_ORIGINS)
def test_exact_web_origins_are_allowed(origin: str) -> None:
    with _client() as client:
        response = client.post("/api/token", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173/",
        "http://127.0.0.1:5174",
        "https://127.0.0.1:5173",
        "https://unapproved.example",
    ],
)
def test_inexact_origin_is_rejected(origin: str) -> None:
    headers = {"Origin": origin}
    with _client() as client:
        response = client.post("/api/token", headers=headers)

    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers
    assert _API_KEY not in response.text
    assert _SECRET not in response.text


def test_missing_origin_is_allowed_for_loopback_native_clients() -> None:
    with _client() as client:
        response = client.post("/api/token")

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_only_expected_origin_and_method() -> None:
    with _client() as client:
        allowed = client.options(
            "/api/token",
            headers={
                "Origin": ALLOWED_WEB_ORIGINS[0],
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Accept",
            },
        )
        disallowed = client.options(
            "/api/token",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ALLOWED_WEB_ORIGINS[0]
    assert disallowed.status_code == 400
    assert "access-control-allow-origin" not in disallowed.headers


def test_untrusted_host_is_rejected() -> None:
    with _client(host="attacker.example") as client:
        response = client.get("/healthz")

    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"


def test_non_loopback_client_is_rejected() -> None:
    with _client(client_host="203.0.113.10") as client:
        response = client.get("/healthz")

    assert response.status_code == 403
    assert response.json() == {"detail": "Loopback clients only"}
    assert response.headers["cache-control"] == "no-store"


def test_health_and_disabled_documentation_expose_no_configuration() -> None:
    with _client() as client:
        health = client.get("/healthz")
        docs = [client.get(path) for path in ("/docs", "/redoc", "/openapi.json")]

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert health.headers["cache-control"] == "no-store"
    assert _API_KEY not in health.text
    assert _SECRET not in health.text
    assert all(response.status_code == 404 for response in docs)
