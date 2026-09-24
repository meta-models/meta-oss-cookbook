from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).parents[1] / "apps/muse-glimmer-server/launch.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_server_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_exec_environment_forces_offline_mode(
    launcher, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")
    monkeypatch.setenv("SSL_CERT_FILE", "/private/cert.pem")

    environment = launcher._server_environment(tmp_path)

    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYTHONPATH"] == f"{tmp_path / 'src'}:/existing/pythonpath"
    assert environment["SSL_CERT_FILE"] == "/private/cert.pem"
