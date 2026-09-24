from __future__ import annotations

import ipaddress
import json
import os
import re
import stat
import subprocess
import sys
import urllib.request

from scripts.repository import CREDENTIAL_FILE, ROOT

_TCP_PORTS = (8000, 7880, 8787, 5173)
_UDP_PORTS = (7882,)
_FORBIDDEN_BROWSER_PATTERNS = {
    "LLM port": re.compile(r"127\.0\.0\.1:8000"),
    "model variant": re.compile(r"muse-glimmer-k-quant|17G|128K|dflash", re.IGNORECASE),
    "model runtime": re.compile(r"Parakeet|Supertonic|MUSE_GLIMMER_|PARAKEET_|SUPERTONIC_"),
    "private path": re.compile(r"/" + r"Users/|\.local/artifacts|\.pte\b"),
    "credential name": re.compile(r"LIVEKIT_API_SECRET"),
    "cloud URL": re.compile(r"wss://|https://[^\s\"']*livekit", re.IGNORECASE),
}


def _socket_rows(kind: str, port: int) -> list[str]:
    command = ["lsof", "-nP", f"-i{kind}:{port}"]
    if kind == "TCP":
        command.append("-sTCP:LISTEN")
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.splitlines()[1:]


def _assert_loopback_sockets() -> None:
    for kind, ports in (("TCP", _TCP_PORTS), ("UDP", _UDP_PORTS)):
        for port in ports:
            rows = _socket_rows(kind, port)
            if not rows:
                raise RuntimeError(f"no {kind} listener found on required port {port}")
            for row in rows:
                if "127.0.0.1" not in row and "[::1]" not in row:
                    raise RuntimeError(f"{kind} port {port} is exposed beyond loopback: {row}")


def _assert_credentials() -> None:
    if not CREDENTIAL_FILE.is_file():
        raise RuntimeError("runtime LiveKit credentials are missing")
    mode = stat.S_IMODE(CREDENTIAL_FILE.stat().st_mode)
    if mode != 0o600:
        raise RuntimeError("runtime LiveKit credentials must have mode 0600")


def _assert_token_boundary() -> None:
    request = urllib.request.Request("http://127.0.0.1:8787/api/token", method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.load(response)
        if response.headers.get("Cache-Control") != "no-store":
            raise RuntimeError("token response is cacheable")
    if not isinstance(payload, dict):
        raise RuntimeError("token response is not an object")
    expected = {"serverUrl", "participantToken", "roomName", "participantIdentity"}
    if set(payload) != expected:
        raise RuntimeError("token response exposes unapproved fields")
    if payload["serverUrl"] != "ws://127.0.0.1:7880":
        raise RuntimeError("token response points beyond local LiveKit")


def _assert_browser_bundle() -> None:
    bundle = ROOT / "apps/web/dist"
    if not bundle.is_dir():
        raise RuntimeError("production web bundle is missing; run `npm run build`")
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in _FORBIDDEN_BROWSER_PATTERNS.items():
            if pattern.search(text):
                raise RuntimeError(f"browser bundle contains forbidden {label}: {path.name}")


def _assert_no_external_connections() -> None:
    result = subprocess.run(
        ["lsof", "-nP", "-iTCP", "-sTCP:ESTABLISHED"], capture_output=True, text=True
    )
    managed_pgids: set[int] = set()
    state_path = ROOT / ".local/run/stack.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        managed_pgids = {
            record["pgid"]
            for record in state.get("services", {}).values()
            if isinstance(record, dict) and isinstance(record.get("pgid"), int)
        }
    for row in result.stdout.splitlines()[1:]:
        columns = row.split()
        if len(columns) < 9:
            continue
        try:
            process_group = os.getpgid(int(columns[1]))
        except (ProcessLookupError, ValueError):
            continue
        if process_group not in managed_pgids:
            continue
        endpoint = next((column for column in reversed(columns) if "->" in column), None)
        if endpoint is None:
            raise RuntimeError(f"managed process has an unparseable connection: {row}")
        remote = endpoint.rsplit("->", 1)[-1]
        host = remote.rsplit(":", 1)[0].strip("[]")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise RuntimeError(
                f"managed process has an unparseable connection: {endpoint}"
            ) from None
        if not address.is_loopback:
            raise RuntimeError(f"managed process has a non-loopback connection: {endpoint}")


def main() -> int:
    if sys.platform != "darwin":
        raise RuntimeError("privacy audit currently supports macOS only")
    _assert_credentials()
    _assert_loopback_sockets()
    _assert_token_boundary()
    _assert_browser_bundle()
    _assert_no_external_connections()
    print("Local privacy audit passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"privacy-audit: {error}", file=sys.stderr)
        raise SystemExit(1) from error
