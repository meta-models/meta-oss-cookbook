from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from scripts.repository import (
    BOOTSTRAP_INPUTS,
    BOOTSTRAP_RECEIPT,
    CREDENTIAL_FILE,
    LOG_DIR,
    ROOT,
    RUN_DIR,
    TOOLCHAIN_LOCK,
    WEB_DIST,
    atomic_write_json,
    digest_json,
    digest_paths,
    ensure_local_directories,
    load_valid_receipt,
    python_environment_fingerprint,
    relative_local_path,
    sha256_tree,
)

STATE_FILE = RUN_DIR / "stack.json"
LOCK_FILE = RUN_DIR / "stack.lock"
SERVICE_ORDER = ("llm", "livekit", "token", "web", "agent")
HTTP_ENDPOINTS = {
    "llm": "http://127.0.0.1:8000/health",
    "livekit": "http://127.0.0.1:7880",
    "token": "http://127.0.0.1:8787/healthz",
    "web": "http://127.0.0.1:5173",
}


@dataclass(frozen=True)
class Service:
    name: str
    command: list[str]
    cwd: Path
    environment: dict[str, str]


def _command_digest(command: list[str]) -> str:
    return hashlib.sha256(b"\0".join(part.encode() for part in command)).hexdigest()


def _process_start(pid: int) -> str | None:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
    value = result.stdout.strip()
    return value or None


def _process_command(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
    )
    value = result.stdout.strip()
    return value or None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _read_state() -> dict[str, object]:
    if not STATE_FILE.is_file():
        return {"schema_version": 1, "services": {}}
    with STATE_FILE.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("services"), dict):
        raise RuntimeError("managed stack state is invalid")
    return value


def _service_matches(record: dict[str, object]) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    if _process_start(pid) != record.get("start_time"):
        return False
    pgid = record.get("pgid")
    try:
        actual_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    if not isinstance(pgid, int) or actual_pgid != pgid:
        return False
    command = _process_command(pid)
    expected = record.get("command_marker")
    return isinstance(expected, str) and command is not None and expected in command


@contextlib.contextmanager
def _lifecycle_lock() -> IO[str]:
    ensure_local_directories()
    stream = LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise RuntimeError("another stack lifecycle operation is in progress") from error
    try:
        yield stream
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _base_environment() -> dict[str, str]:
    allowed = ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "SSL_CERT_FILE")
    return {
        **{name: os.environ[name] for name in allowed if name in os.environ},
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _artifact(receipt: dict[str, object], role: str) -> str:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or role not in artifacts:
        raise RuntimeError(f"prepared artifact is missing from receipt: {role}")
    item = artifacts[role]
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise RuntimeError(f"prepared artifact receipt is invalid: {role}")
    return str(relative_local_path(item["path"]))


def _new_credentials() -> tuple[str, str]:
    api_key = f"local_{secrets.token_hex(8)}"
    api_secret = secrets.token_urlsafe(36)
    temporary = CREDENTIAL_FILE.with_name(
        f".{CREDENTIAL_FILE.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{api_key}: {api_secret}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CREDENTIAL_FILE)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return api_key, api_secret


def _validated_tools() -> dict[str, str]:
    if not BOOTSTRAP_RECEIPT.is_file():
        raise RuntimeError("source dependencies are not bootstrapped; run `make bootstrap`")
    with BOOTSTRAP_RECEIPT.open(encoding="utf-8") as stream:
        receipt = json.load(stream)
    if not WEB_DIST.is_dir() or (
        receipt.get("toolchain_lock") != digest_json(TOOLCHAIN_LOCK)
        or receipt.get("bootstrap_inputs") != digest_paths(BOOTSTRAP_INPUTS)
        or receipt.get("web_dist") != sha256_tree(WEB_DIST)
    ):
        raise RuntimeError("bootstrap receipt is stale; run `make bootstrap`")
    tools = receipt.get("tools")
    required = {"python", "node", "livekit-server"}
    if not isinstance(tools, dict) or not required <= set(tools):
        raise RuntimeError("bootstrap receipt is invalid; run `make bootstrap`")
    paths: dict[str, str] = {}
    for name in required:
        item = tools[name]
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError("bootstrap receipt is invalid; run `make bootstrap`")
        path = item["path"]
        if not os.access(path, os.X_OK):
            raise RuntimeError(f"bootstrapped tool is unavailable: {name}")
        paths[name] = path
    if receipt.get("python_environment") != python_environment_fingerprint(Path(paths["python"])):
        raise RuntimeError("Python environment changed; run `make bootstrap`")
    return paths


def _services(receipt: dict[str, object], api_key: str, api_secret: str) -> list[Service]:
    base = _base_environment()
    venv_bin = ROOT / ".venv/bin"
    token_service = str(venv_bin / "muse-glimmer-token-service")
    worker = str(venv_bin / "muse-glimmer-worker")
    for executable in (token_service, worker):
        if not os.access(executable, os.X_OK):
            raise RuntimeError("Python workspace is not bootstrapped; run `make bootstrap`")
    tools = _validated_tools()
    python = tools["python"]
    livekit_environment = {
        **base,
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": api_key,
        "LIVEKIT_API_SECRET": api_secret,
    }
    worker_environment = {
        **livekit_environment,
        "GLIMMER_AGENT_NAME": "assistant",
        "GLIMMER_LANGUAGE": "en",
        "MUSE_GLIMMER_BASE_URL": "http://127.0.0.1:8000/v1",
        "MUSE_GLIMMER_API_KEY": "local",
        "MUSE_GLIMMER_REASONING_STRENGTH": "low",
        "MUSE_GLIMMER_MAX_TOKENS": "256",
        "ORT_DISABLE_TELEMETRY": "1",
        "PARAKEET_HELPER_PATH": _artifact(receipt, "parakeet_helper"),
        "PARAKEET_MODEL_PATH": _artifact(receipt, "parakeet_model"),
        "PARAKEET_TOKENIZER_PATH": _artifact(receipt, "parakeet_tokenizer"),
        "SUPERTONIC_RUNNER_PATH": _artifact(receipt, "supertonic_runner"),
        "SUPERTONIC_PTE_PATH": _artifact(receipt, "supertonic_model"),
        "SUPERTONIC_ASSET_DIR": _artifact(receipt, "supertonic_assets"),
        "SUPERTONIC_VOICE_STYLE_PATH": _artifact(receipt, "supertonic_voice_style"),
        "SUPERTONIC_SPEED": "1.05",
        "SUPERTONIC_SEED": "42",
    }
    return [
        Service(
            "llm",
            [python, "apps/muse-glimmer-server/launch.py"],
            ROOT,
            base,
        ),
        Service(
            "livekit",
            [
                tools["livekit-server"],
                "--config",
                str(ROOT / "config/livekit/macos-arm64.yaml"),
                "--key-file",
                str(CREDENTIAL_FILE),
            ],
            ROOT,
            base,
        ),
        Service(
            "token",
            [token_service],
            ROOT,
            livekit_environment,
        ),
        Service(
            "web",
            [tools["node"], "server.mjs", "--host", "127.0.0.1", "--port", "5173"],
            ROOT / "apps/web",
            base,
        ),
        Service(
            "agent",
            [worker, "dev"],
            ROOT,
            worker_environment,
        ),
    ]


def _port_available(port: int, *, udp: bool = False) -> bool:
    sock_type = socket.SOCK_DGRAM if udp else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, sock_type) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _assert_ports_available() -> None:
    for port in (8000, 7880, 8787, 5173):
        if not _port_available(port):
            raise RuntimeError(f"TCP port {port} is already in use")
    if not _port_available(7882, udp=True):
        raise RuntimeError("UDP port 7882 is already in use")


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_process_group(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_alive(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_alive(pgid)


def _terminate_new_process_group(process: subprocess.Popen[bytes]) -> None:
    pgid = process.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    if not _wait_for_process_group(pgid, 5):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        if not _wait_for_process_group(pgid, 5):
            raise RuntimeError(f"process group {pgid} survived SIGKILL")
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _start_service(service: Service) -> tuple[subprocess.Popen[bytes], dict[str, object]]:
    log_path = LOG_DIR / f"{service.name}.log"
    log_stream = log_path.open("wb")
    try:
        process = subprocess.Popen(
            service.command,
            cwd=service.cwd,
            env=service.environment,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_stream.close()
    time.sleep(0.2)
    if process.poll() is not None:
        _terminate_new_process_group(process)
        raise RuntimeError(f"{service.name} exited during startup; see {log_path}")
    start_time = _process_start(process.pid)
    command = _process_command(process.pid)
    if start_time is None or command is None:
        _terminate_new_process_group(process)
        raise RuntimeError(f"could not establish process identity for {service.name}")
    marker = Path(service.command[0]).name
    return process, {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_time": start_time,
        "command_marker": marker,
        "command_digest": _command_digest(service.command),
        "log": str(log_path.relative_to(ROOT)),
    }


def _http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except urllib.error.HTTPError as error:
        return error.code < 500
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_http(name: str, record: dict[str, object], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _http_ready(HTTP_ENDPOINTS[name]):
            return
        if not _service_matches(record):
            raise RuntimeError(f"{name} exited before readiness")
        time.sleep(0.5)
    raise RuntimeError(f"{name} timed out waiting for readiness")


def _wait_for_agent(record: dict[str, object], timeout: float = 60) -> str:
    log_path = ROOT / str(record["log"])
    registration = re.compile(r'registered worker.*"agent_name"\s*:\s*"assistant"')
    health_endpoint = re.compile(r"HTTP server listening on 127\.0\.0\.1:(\d+)")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        match = health_endpoint.search(text)
        if registration.search(text) and match is not None:
            url = f"http://127.0.0.1:{match.group(1)}/"
            if _http_ready(url):
                return url
        if not _service_matches(record):
            raise RuntimeError("agent exited before registration")
        time.sleep(0.5)
    raise RuntimeError("agent timed out waiting for assistant registration")


def _record_group_owned(record: dict[str, object]) -> bool:
    pgid = record.get("pgid")
    if not isinstance(pgid, int) or not _process_group_alive(pgid):
        return False
    if _service_matches(record):
        return True
    # A surviving child keeps the original PGID after its leader exits. If a
    # process now occupies the leader PID, the PGID may have been recycled.
    return not _pid_alive(pgid)


def _signal_service(record: dict[str, object], signum: int) -> None:
    pgid = record.get("pgid")
    if not isinstance(pgid, int) or not _record_group_owned(record):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signum)


def _stop_records(records: dict[str, object]) -> None:
    for name in reversed(SERVICE_ORDER):
        record = records.get(name)
        if isinstance(record, dict):
            _signal_service(record, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        active = [
            record
            for record in records.values()
            if isinstance(record, dict) and _record_group_owned(record)
        ]
        if not active:
            return
        time.sleep(0.25)
    for name in reversed(SERVICE_ORDER):
        record = records.get(name)
        if isinstance(record, dict):
            _signal_service(record, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while True:
        if not any(
            isinstance(record, dict) and _record_group_owned(record) for record in records.values()
        ):
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    raise RuntimeError("managed process group survived SIGKILL; state was retained")


def _write_state(records: dict[str, object]) -> None:
    atomic_write_json(
        STATE_FILE,
        {"schema_version": 1, "updated_at": time.time(), "services": records},
    )


def _run_probe(command: list[str], *, timeout: float = 180) -> None:
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout)


def _up_locked() -> None:
    state = _read_state()
    existing = state["services"]
    if any(
        isinstance(record, dict) and _record_group_owned(record) for record in existing.values()
    ):
        raise RuntimeError(
            "a managed stack or orphaned process group is still running; use `make down`"
        )
    _assert_ports_available()
    receipt = load_valid_receipt()
    api_key, api_secret = _new_credentials()
    records: dict[str, object] = {}
    interrupted = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    previous = {
        signum: signal.signal(signum, handle_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for service in _services(receipt, api_key, api_secret):
            _, record = _start_service(service)
            records[service.name] = record
            _write_state(records)
            if service.name == "llm":
                _wait_for_http(service.name, record, 360)
                _run_probe([sys.executable, "-m", "scripts.llm_readiness"])
            elif service.name == "agent":
                record["health_url"] = _wait_for_agent(record)
                _write_state(records)
            elif service.name in HTTP_ENDPOINTS:
                _wait_for_http(service.name, record, 120)
        _run_probe([sys.executable, "-m", "scripts.privacy_audit"])
    except BaseException:
        _stop_records(records)
        STATE_FILE.unlink(missing_ok=True)
        CREDENTIAL_FILE.unlink(missing_ok=True)
        if interrupted:
            print("startup interrupted; rolled back managed services", file=sys.stderr)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _down_locked() -> None:
    state = _read_state()
    records = state["services"]
    _stop_records(records)
    STATE_FILE.unlink(missing_ok=True)
    CREDENTIAL_FILE.unlink(missing_ok=True)


def up() -> int:
    with _lifecycle_lock():
        _up_locked()
    print("Muse Glimmer is ready at http://127.0.0.1:5173")
    return 0


def down() -> int:
    with _lifecycle_lock():
        _down_locked()
    print("Muse Glimmer stack is stopped.")
    return 0


def status() -> int:
    state = _read_state()
    records = state["services"]
    healthy = True
    for name in SERVICE_ORDER:
        record = records.get(name)
        process_ok = isinstance(record, dict) and _service_matches(record)
        if name in HTTP_ENDPOINTS:
            endpoint_ok = _http_ready(HTTP_ENDPOINTS[name])
        elif name == "agent" and isinstance(record, dict):
            health_url = record.get("health_url")
            endpoint_ok = isinstance(health_url, str) and _http_ready(health_url)
        else:
            endpoint_ok = False
        service_ok = process_ok and endpoint_ok
        healthy = healthy and service_ok
        print(f"{name:<8} {'healthy' if service_ok else 'unavailable'}")
    return 0 if healthy else 1


def logs() -> int:
    paths = [LOG_DIR / f"{name}.log" for name in SERVICE_ORDER]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise RuntimeError("no managed logs exist")
    return subprocess.run(["tail", "-n", "80", "-F", *map(str, existing)]).returncode


def restart() -> int:
    with _lifecycle_lock():
        _down_locked()
        _up_locked()
    print("Muse Glimmer is ready at http://127.0.0.1:5173")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the local Muse Glimmer voice stack")
    parser.add_argument("operation", choices=("up", "down", "restart", "status", "logs"))
    operation = parser.parse_args().operation
    return {"up": up, "down": down, "restart": restart, "status": status, "logs": logs}[operation]()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"dev-stack: {error}", file=sys.stderr)
        raise SystemExit(1) from error
