from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
RUN_DIR = LOCAL / "run"
LOG_DIR = LOCAL / "logs"
STATE_DIR = LOCAL / "state"
ARTIFACT_DIR = LOCAL / "artifacts"
SOURCE_DIR = LOCAL / "src"
PREPARED_RECEIPT = STATE_DIR / "prepared.json"
BOOTSTRAP_RECEIPT = STATE_DIR / "bootstrap.json"
CREDENTIAL_FILE = RUN_DIR / "livekit.keys"
COMPATIBILITY_LOCK = ROOT / "config" / "dependencies" / "compatibility.lock.json"
ARTIFACT_LOCK = ROOT / "artifacts" / "macos-arm64.lock.json"
TOOLCHAIN_LOCK = ROOT / "config" / "dependencies" / "toolchain.lock.json"
BOOTSTRAP_INPUTS = (
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    ROOT / "apps/token-service/pyproject.toml",
    ROOT / "apps/worker/pyproject.toml",
    ROOT / "packages/livekit-plugins-executorch/pyproject.toml",
    ROOT / "apps/web/package.json",
    ROOT / "apps/web/package-lock.json",
    ROOT / "apps/web/index.html",
    ROOT / "apps/web/server.mjs",
    ROOT / "apps/web/src",
    ROOT / "apps/web/tsconfig.app.json",
    ROOT / "apps/web/tsconfig.json",
    ROOT / "apps/web/tsconfig.node.json",
    ROOT / "apps/web/vite.config.ts",
)
WEB_DIST = ROOT / "apps/web/dist"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def digest_json(path: Path) -> str:
    return sha256_file(path)


def digest_paths(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"bootstrap input is missing: {path.relative_to(ROOT)}")
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update((sha256_tree(path) if path.is_dir() else sha256_file(path)).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_environment_fingerprint(
    distributions: Iterable[Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    installed = distributions if distributions is not None else importlib.metadata.distributions()
    ordered = sorted(
        installed,
        key=lambda distribution: (
            str(distribution.metadata.get("Name", "")).lower(),
            distribution.version,
        ),
    )
    for distribution in ordered:
        name = str(distribution.metadata.get("Name", "")).lower()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(distribution.version.encode())
        digest.update(b"\0")
        for relative in sorted(distribution.files or (), key=str):
            relative_text = str(relative)
            if relative_text.endswith(".pyc") or "__pycache__" in Path(relative_text).parts:
                continue
            digest.update(relative_text.encode())
            digest.update(b"\0")
            path = Path(distribution.locate_file(relative))
            if path.is_file():
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
                if path.name in {"RECORD", "direct_url.json"}:
                    digest.update(sha256_file(path).encode())
            else:
                digest.update(b"missing")
            digest.update(b"\0")
    return digest.hexdigest()


def python_environment_fingerprint(python: Path) -> str:
    script = (
        "from scripts.repository import installed_environment_fingerprint; "
        "print(installed_environment_fingerprint())"
    )
    return subprocess.run(
        [str(python), "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def landed_gate_commits(compatibility: dict[str, Any]) -> tuple[str, ...]:
    executorch = compatibility.get("executorch")
    if not isinstance(executorch, dict):
        raise RuntimeError("compatibility lock has no ExecuTorch configuration")
    gates = executorch.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError("compatibility lock has no upstream gates")
    commits: list[str] = []
    for name, gate in gates.items():
        if not isinstance(gate, dict):
            raise RuntimeError(f"compatibility gate must be an object: {name}")
        if gate.get("status") != "landed":
            continue
        commit = gate.get("commit")
        if not isinstance(commit, str) or len(commit) != 40:
            raise RuntimeError(f"landed compatibility gate requires a commit: {name}")
        commits.append(commit)
    return tuple(sorted(commits))


def validate_executorch_checkout(
    checkout: Path, expected_commit: str, *, required_ancestors: Iterable[str] = ()
) -> None:
    if not checkout.is_dir() or not (checkout / ".git").exists():
        raise RuntimeError(f"ExecuTorch checkout is missing: {checkout}")
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"ExecuTorch checkout is {actual}; expected {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("ExecuTorch checkout must remain clean")
    for ancestor in required_ancestors:
        result = subprocess.run(
            ["git", "-C", str(checkout), "merge-base", "--is-ancestor", ancestor, actual],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pinned ExecuTorch commit does not contain landed capability {ancestor}"
            )


def require_supported_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Muse Glimmer currently supports macOS on Apple silicon only")


def ensure_local_directories() -> None:
    for path in (RUN_DIR, LOG_DIR, STATE_DIR, ARTIFACT_DIR, SOURCE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def relative_local_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(LOCAL.resolve())
    except ValueError as error:
        raise ValueError(f"artifact destination must stay under .local: {value}") from error
    return path


def atomic_write_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_valid_receipt() -> dict[str, Any]:
    if not PREPARED_RECEIPT.is_file():
        raise RuntimeError("artifacts are not prepared; run `make prepare-artifacts`")
    receipt = read_json(PREPARED_RECEIPT)
    expected_locks = {
        "compatibility": digest_json(COMPATIBILITY_LOCK),
        "artifacts": digest_json(ARTIFACT_LOCK),
        "toolchain": digest_json(TOOLCHAIN_LOCK),
    }
    if receipt.get("locks") != expected_locks:
        raise RuntimeError("artifact receipt is stale; run `make prepare-artifacts`")
    compatibility = read_json(COMPATIBILITY_LOCK)
    commit = compatibility.get("executorch", {}).get("commit")
    if not commit or receipt.get("executorch_commit") != commit:
        raise RuntimeError("artifact receipt does not match the pinned ExecuTorch commit")
    checkout_value = receipt.get("executorch_checkout")
    if not isinstance(checkout_value, str):
        raise RuntimeError("artifact receipt has no ExecuTorch checkout")
    validate_executorch_checkout(
        Path(checkout_value).expanduser().resolve(),
        commit,
        required_ancestors=landed_gate_commits(compatibility),
    )
    recorded = receipt.get("artifacts")
    if not isinstance(recorded, dict):
        raise RuntimeError("artifact receipt has no artifact inventory")
    manifest = read_json(ARTIFACT_LOCK)
    requirements = {item["role"]: item for item in manifest["artifacts"]}
    if set(recorded) != set(requirements):
        raise RuntimeError("artifact receipt roles do not match the manifest")
    for role, item in recorded.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"invalid artifact receipt entry: {role}")
        path = relative_local_path(item["path"])
        if not path.exists():
            raise RuntimeError(f"prepared artifact is missing: {role}")
        requirement = requirements[role]
        if requirement["kind"] == "file" and not path.is_file():
            raise RuntimeError(f"prepared artifact must remain a regular file: {role}")
        if requirement["kind"] == "directory" and not path.is_dir():
            raise RuntimeError(f"prepared artifact must remain a directory: {role}")
        if requirement["executable"] and not os.access(path, os.X_OK):
            raise RuntimeError(f"prepared artifact must remain executable: {role}")
        actual = sha256_tree(path) if path.is_dir() else sha256_file(path)
        if actual != item.get("sha256"):
            raise RuntimeError(f"prepared artifact checksum changed: {role}")
        actual_size = artifact_size(path)
        expected_size = requirement.get("size_bytes")
        if actual_size != item.get("size_bytes") or (
            expected_size is not None and actual_size != expected_size
        ):
            raise RuntimeError(f"prepared artifact size changed: {role}")
    return receipt
