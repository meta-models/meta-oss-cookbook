from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import privacy_audit


def _write_stack(tmp_path) -> None:
    run_dir = tmp_path / ".local/run"
    run_dir.mkdir(parents=True)
    (run_dir / "stack.json").write_text(
        json.dumps({"services": {"web": {"pid": 100, "pgid": 100}}})
    )


def _set_connection_output(tmp_path, monkeypatch: pytest.MonkeyPatch, row: str) -> None:
    _write_stack(tmp_path)
    output = f"COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n{row}\n"
    monkeypatch.setattr(privacy_audit, "ROOT", tmp_path)
    monkeypatch.setattr(
        privacy_audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    monkeypatch.setattr(privacy_audit.os, "getpgid", lambda pid: 100 if pid == 321 else pid)


def test_external_connection_audit_accepts_loopback_with_state_column(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_connection_output(
        tmp_path,
        monkeypatch,
        "node 321 user 10u IPv4 1 0t0 TCP 127.0.0.1:5000->127.0.0.1:7880 (ESTABLISHED)",
    )

    privacy_audit._assert_no_external_connections()


def test_external_connection_audit_accepts_ipv6_loopback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_connection_output(
        tmp_path,
        monkeypatch,
        "node 321 user 10u IPv6 1 0t0 TCP [::1]:5000->[::1]:7880 (ESTABLISHED)",
    )

    privacy_audit._assert_no_external_connections()


def test_external_connection_audit_includes_managed_process_group(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_connection_output(
        tmp_path,
        monkeypatch,
        "node 321 user 10u IPv4 1 0t0 TCP 127.0.0.1:5000->203.0.113.10:443 (ESTABLISHED)",
    )

    with pytest.raises(RuntimeError, match="non-loopback connection"):
        privacy_audit._assert_no_external_connections()


def test_external_connection_audit_rejects_external_ipv6(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_connection_output(
        tmp_path,
        monkeypatch,
        "node 321 user 10u IPv6 1 0t0 TCP [::1]:5000->[2001:db8::1]:443 (ESTABLISHED)",
    )

    with pytest.raises(RuntimeError, match="non-loopback connection"):
        privacy_audit._assert_no_external_connections()


@pytest.mark.parametrize(
    "row",
    [
        "node 321 user 10u IPv4 1 0t0 TCP 127.0.0.1:5000 (ESTABLISHED)",
        "node 321 user 10u IPv4 1 0t0 TCP 127.0.0.1:5000->not-an-address:443 (ESTABLISHED)",
    ],
)
def test_external_connection_audit_rejects_unparseable_endpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    _set_connection_output(tmp_path, monkeypatch, row)

    with pytest.raises(RuntimeError, match="unparseable connection"):
        privacy_audit._assert_no_external_connections()
