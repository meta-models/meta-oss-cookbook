from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import dev_stack


@pytest.fixture
def receipt() -> dict[str, object]:
    roles = {
        "parakeet_helper": ".local/artifacts/bin/parakeet_helper",
        "parakeet_model": ".local/artifacts/parakeet/model.pte",
        "parakeet_tokenizer": ".local/artifacts/parakeet/tokenizer.model",
        "supertonic_runner": ".local/artifacts/bin/supertonic_runner",
        "supertonic_model": ".local/artifacts/supertonic/model.pte",
        "supertonic_assets": ".local/artifacts/supertonic/assets",
        "supertonic_voice_style": ".local/artifacts/supertonic/voice-style.json",
    }
    return {"artifacts": {role: {"path": path} for role, path in roles.items()}}


def test_validated_tools_rejects_missing_bootstrap_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dev_stack, "BOOTSTRAP_RECEIPT", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="not bootstrapped"):
        dev_stack._validated_tools()


def test_validated_tools_rejects_stale_bootstrap_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bootstrap.json"
    receipt.write_text(
        json.dumps(
            {
                "toolchain_lock": "lock",
                "bootstrap_inputs": "stale",
                "web_dist": "web",
                "tools": {},
            }
        )
    )
    web_dist = tmp_path / "dist"
    web_dist.mkdir()
    monkeypatch.setattr(dev_stack, "BOOTSTRAP_RECEIPT", receipt)
    monkeypatch.setattr(dev_stack, "WEB_DIST", web_dist)
    monkeypatch.setattr(dev_stack, "digest_json", lambda _path: "lock")
    monkeypatch.setattr(dev_stack, "digest_paths", lambda _paths: "current")
    monkeypatch.setattr(dev_stack, "sha256_tree", lambda _path: "web")

    with pytest.raises(RuntimeError, match="stale"):
        dev_stack._validated_tools()


def test_validated_tools_rejects_changed_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    node = tmp_path / "node"
    livekit = tmp_path / "livekit-server"
    for tool in (python, node, livekit):
        tool.write_text("tool")
        tool.chmod(0o755)
    receipt = tmp_path / "bootstrap.json"
    receipt.write_text(
        json.dumps(
            {
                "toolchain_lock": "lock",
                "bootstrap_inputs": "inputs",
                "web_dist": "web",
                "python_environment": "old",
                "tools": {
                    "python": {"path": str(python)},
                    "node": {"path": str(node)},
                    "livekit-server": {"path": str(livekit)},
                },
            }
        )
    )
    web_dist = tmp_path / "dist"
    web_dist.mkdir()
    monkeypatch.setattr(dev_stack, "BOOTSTRAP_RECEIPT", receipt)
    monkeypatch.setattr(dev_stack, "WEB_DIST", web_dist)
    monkeypatch.setattr(dev_stack, "digest_json", lambda _path: "lock")
    monkeypatch.setattr(dev_stack, "digest_paths", lambda _paths: "inputs")
    monkeypatch.setattr(dev_stack, "sha256_tree", lambda _path: "web")
    monkeypatch.setattr(dev_stack, "python_environment_fingerprint", lambda _path: "new")

    with pytest.raises(RuntimeError, match="Python environment changed"):
        dev_stack._validated_tools()


def test_base_environment_forces_model_libraries_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/tmp/test-home")
    environment = dev_stack._base_environment()

    assert environment["HOME"] == "/tmp/test-home"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_credentials_are_private_before_secret_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_file = tmp_path / "livekit.keys"
    observed_modes: list[int] = []
    real_fdopen = os.fdopen

    def checked_fdopen(descriptor: int, *args, **kwargs):
        observed_modes.append(os.fstat(descriptor).st_mode & 0o777)
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(dev_stack, "CREDENTIAL_FILE", credential_file)
    monkeypatch.setattr(dev_stack.os, "fdopen", checked_fdopen)

    api_key, api_secret = dev_stack._new_credentials()

    assert credential_file.read_text() == f"{api_key}: {api_secret}\n"
    assert observed_modes == [0o600]
    assert credential_file.stat().st_mode & 0o777 == 0o600


def test_service_order_and_local_environment(
    receipt: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dev_stack,
        "_validated_tools",
        lambda: {
            "python": "/test/.venv/bin/python",
            "node": "/test/bin/node",
            "livekit-server": "/test/bin/livekit-server",
        },
    )
    monkeypatch.setattr(dev_stack.os, "access", lambda _path, _mode: True)
    services = dev_stack._services(receipt, "test-key", "test-secret")

    assert [service.name for service in services] == list(dev_stack.SERVICE_ORDER)
    worker = services[-1]
    assert worker.environment["LIVEKIT_URL"] == "ws://127.0.0.1:7880"
    assert worker.environment["MUSE_GLIMMER_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert worker.environment["MUSE_GLIMMER_REASONING_STRENGTH"] == "low"
    assert worker.environment["MUSE_GLIMMER_MAX_TOKENS"] == "256"
    assert worker.environment["ORT_DISABLE_TELEMETRY"] == "1"
    assert "OPENAI_API_KEY" not in worker.environment


def test_stop_records_uses_reverse_order_and_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    signalled: list[tuple[str, int]] = []
    records = {name: {"name": name} for name in dev_stack.SERVICE_ORDER}
    active = {name for name in dev_stack.SERVICE_ORDER}

    def matches(record: dict[str, object]) -> bool:
        return str(record["name"]) in active

    def signal_record(record: dict[str, object], signum: int) -> None:
        name = str(record["name"])
        signalled.append((name, signum))
        if signum == dev_stack.signal.SIGKILL:
            active.discard(name)

    now = 0.0

    def monotonic() -> float:
        nonlocal now
        now += 20.0
        return now

    monkeypatch.setattr(dev_stack, "_record_group_owned", matches)
    monkeypatch.setattr(dev_stack, "_signal_service", signal_record)
    monkeypatch.setattr(dev_stack.time, "monotonic", monotonic)
    monkeypatch.setattr(dev_stack.time, "sleep", lambda _seconds: None)

    dev_stack._stop_records(records)

    expected_reverse = list(reversed(dev_stack.SERVICE_ORDER))
    assert [
        name for name, signum in signalled if signum == dev_stack.signal.SIGTERM
    ] == expected_reverse
    assert [
        name for name, signum in signalled if signum == dev_stack.signal.SIGKILL
    ] == expected_reverse


def test_up_rejects_dead_leader_with_live_child_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"pid": 42, "pgid": 42}
    monkeypatch.setattr(
        dev_stack,
        "_read_state",
        lambda: {"schema_version": 1, "services": {"agent": record}},
    )
    monkeypatch.setattr(dev_stack, "_record_group_owned", lambda _record: True)

    with pytest.raises(RuntimeError, match="orphaned process group"):
        dev_stack._up_locked()


def test_dead_leader_with_live_child_group_is_still_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"pid": 42, "pgid": 42}
    alive = True
    signals: list[int] = []

    monkeypatch.setattr(dev_stack, "_service_matches", lambda _record: False)
    monkeypatch.setattr(dev_stack, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(dev_stack, "_process_group_alive", lambda _pgid: alive)

    def killpg(_pgid: int, signum: int) -> None:
        nonlocal alive
        signals.append(signum)
        alive = False

    monkeypatch.setattr(dev_stack.os, "killpg", killpg)

    assert dev_stack._record_group_owned(record)
    dev_stack._stop_records({"agent": record})
    assert signals == [dev_stack.signal.SIGTERM]


def test_stop_waits_for_delayed_exit_after_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"name": "agent"}
    owned = iter((True, False))
    signals: list[int] = []
    times = iter((0.0, 11.0, 20.0, 20.0))

    def monotonic() -> float:
        return next(times)

    monkeypatch.setattr(dev_stack, "_record_group_owned", lambda _record: next(owned))
    monkeypatch.setattr(
        dev_stack,
        "_signal_service",
        lambda _record, signum: signals.append(signum),
    )
    monkeypatch.setattr(dev_stack.time, "monotonic", monotonic)
    monkeypatch.setattr(dev_stack.time, "sleep", lambda _seconds: None)

    dev_stack._stop_records({"agent": record})

    assert signals == [dev_stack.signal.SIGTERM, dev_stack.signal.SIGKILL]


def test_agent_readiness_returns_health_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        'HTTP server listening on 127.0.0.1:54321\nregistered worker {"agent_name": "assistant"}\n'
    )
    monkeypatch.setattr(dev_stack, "ROOT", tmp_path)
    monkeypatch.setattr(dev_stack, "_http_ready", lambda url: url.endswith(":54321/"))
    monkeypatch.setattr(dev_stack, "_service_matches", lambda _record: True)

    assert dev_stack._wait_for_agent({"log": "agent.log"}) == "http://127.0.0.1:54321/"


def test_process_identity_allows_same_pid_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {
        "pid": 42,
        "pgid": 42,
        "start_time": "Mon Aug 24 00:00:00 2026",
        "command_marker": "python",
    }
    monkeypatch.setattr(dev_stack, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(dev_stack, "_process_start", lambda _pid: record["start_time"])
    monkeypatch.setattr(dev_stack.os, "getpgid", lambda _pid: 42)
    monkeypatch.setattr(
        dev_stack,
        "_process_command",
        lambda _pid: ".venv/bin/python -m executorch.examples.models.muse_glimmer.serving.serve",
    )

    assert dev_stack._service_matches(record)


def test_process_identity_rejects_start_group_or_marker_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ".venv/bin/muse-glimmer-worker dev"
    record = {
        "pid": 42,
        "pgid": 42,
        "start_time": "Mon Aug 24 00:00:00 2026",
        "command_marker": "muse-glimmer-worker",
    }
    monkeypatch.setattr(dev_stack, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(dev_stack, "_process_start", lambda _pid: record["start_time"])
    monkeypatch.setattr(dev_stack.os, "getpgid", lambda _pid: 42)
    monkeypatch.setattr(dev_stack, "_process_command", lambda _pid: command)
    assert dev_stack._service_matches(record)

    monkeypatch.setattr(dev_stack, "_process_start", lambda _pid: "different process start")
    assert not dev_stack._service_matches(record)
    monkeypatch.setattr(dev_stack, "_process_start", lambda _pid: record["start_time"])
    monkeypatch.setattr(dev_stack.os, "getpgid", lambda _pid: 99)
    assert not dev_stack._service_matches(record)
    monkeypatch.setattr(dev_stack.os, "getpgid", lambda _pid: 42)
    monkeypatch.setattr(dev_stack, "_process_command", lambda _pid: "different-service")
    assert not dev_stack._service_matches(record)


def test_new_process_group_termination_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[tuple[int, int]] = []

    class Process:
        pid = 42

        def wait(self, timeout: float) -> int:
            return 0

    waits = iter((False, True))
    monkeypatch.setattr(dev_stack, "_wait_for_process_group", lambda _pgid, _timeout: next(waits))
    monkeypatch.setattr(dev_stack.os, "killpg", lambda pgid, signum: signals.append((pgid, signum)))

    dev_stack._terminate_new_process_group(Process())  # type: ignore[arg-type]

    assert signals == [
        (42, dev_stack.signal.SIGTERM),
        (42, dev_stack.signal.SIGKILL),
    ]


def test_restart_holds_one_lifecycle_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Lock:
        def __enter__(self):
            events.append("locked")

        def __exit__(self, *_args):
            events.append("unlocked")

    monkeypatch.setattr(dev_stack, "_lifecycle_lock", Lock)
    monkeypatch.setattr(dev_stack, "_down_locked", lambda: events.append("down"))
    monkeypatch.setattr(dev_stack, "_up_locked", lambda: events.append("up"))

    assert dev_stack.restart() == 0
    assert events == ["locked", "down", "up", "unlocked"]


def test_missing_artifact_role_fails(
    receipt: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dev_stack,
        "_validated_tools",
        lambda: {
            "python": "/test/.venv/bin/python",
            "node": "/test/bin/node",
            "livekit-server": "/test/bin/livekit-server",
        },
    )
    monkeypatch.setattr(dev_stack.os, "access", lambda _path, _mode: True)
    del receipt["artifacts"]["supertonic_voice_style"]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="supertonic_voice_style"):
        dev_stack._services(receipt, "test-key", "test-secret")
