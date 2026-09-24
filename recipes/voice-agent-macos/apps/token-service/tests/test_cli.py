from __future__ import annotations

from typing import Any

from muse_glimmer_token_service import __main__


def test_launcher_uses_fixed_loopback_address(monkeypatch: Any) -> None:
    invocation: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        invocation["app"] = app
        invocation.update(kwargs)

    monkeypatch.setattr(__main__.uvicorn, "run", fake_run)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9999")

    __main__.main()

    assert invocation == {
        "app": "muse_glimmer_token_service.app:create_app",
        "factory": True,
        "host": "127.0.0.1",
        "port": 8787,
        "proxy_headers": False,
        "server_header": False,
    }
