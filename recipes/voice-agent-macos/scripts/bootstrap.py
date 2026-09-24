from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.repository import (
    BOOTSTRAP_INPUTS,
    BOOTSTRAP_RECEIPT,
    ROOT,
    TOOLCHAIN_LOCK,
    WEB_DIST,
    atomic_write_json,
    digest_json,
    digest_paths,
    ensure_local_directories,
    landed_gate_commits,
    python_environment_fingerprint,
    read_json,
    require_supported_platform,
    sha256_tree,
    validate_executorch_checkout,
)


def _version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip().splitlines()[0]


_VERSION = re.compile(r"\d+(?:\.\d+){0,2}")


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = _VERSION.search(value)
    if match is None:
        raise RuntimeError(f"could not parse tool version: {value!r}")
    parts = tuple(int(part) for part in match.group().split("."))
    return (parts + (0, 0, 0))[:3]


def _require_version(name: str, actual: str, requirement: str) -> None:
    version = _version_tuple(actual)
    for constraint in requirement.split(","):
        constraint = constraint.strip()
        if constraint.startswith(">=") and version < _version_tuple(constraint[2:]):
            raise RuntimeError(f"{name} {actual!r} does not satisfy {requirement}")
        if constraint.startswith("<") and version >= _version_tuple(constraint[1:]):
            raise RuntimeError(f"{name} {actual!r} does not satisfy {requirement}")


def _require_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required tool is missing: {name}")
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare source dependencies for local development"
    )
    parser.add_argument(
        "--skip-install", action="store_true", help="validate without installing packages"
    )
    args = parser.parse_args()

    require_supported_platform()
    ensure_local_directories()
    compatibility = read_json(ROOT / "config/dependencies/compatibility.lock.json")
    toolchain = read_json(ROOT / "config/dependencies/toolchain.lock.json")
    tools = {
        name: _require_tool(name)
        for name in ("git", "uv", "node", "npm", "cmake", "livekit-server")
    }
    versions = {
        "python": sys.version.split()[0],
        **{name: _version([path, "--version"]) for name, path in tools.items()},
    }
    for name, requirement in toolchain["tools"].items():
        _require_version(name, versions[name], requirement)

    commit = compatibility["executorch"].get("commit")
    if not commit:
        raise RuntimeError(
            "the compatibility lock is gated until one ExecuTorch commit contains "
            "supports_cancel and supertonic_server_jsonl"
        )

    checkout = (
        Path(os.environ.get("GLIMMER_EXECUTORCH_ROOT", ROOT / ".local/src/executorch"))
        .expanduser()
        .resolve()
    )
    validate_executorch_checkout(
        checkout,
        commit,
        required_ancestors=landed_gate_commits(compatibility),
    )

    if not args.skip_install:
        subprocess.run(
            [tools["uv"], "sync", "--all-packages", "--all-groups", "--frozen"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([tools["npm"], "ci", "--prefix", "apps/web"], cwd=ROOT, check=True)
        subprocess.run([tools["npm"], "run", "build", "--prefix", "apps/web"], cwd=ROOT, check=True)

    if not WEB_DIST.is_dir():
        raise RuntimeError("web application is not built; run bootstrap without --skip-install")
    runtime_python = ROOT / ".venv/bin/python"
    if not os.access(runtime_python, os.X_OK):
        raise RuntimeError(
            "Python workspace environment is missing; run bootstrap without --skip-install"
        )
    runtime_python_version = _version([str(runtime_python), "--version"]).removeprefix("Python ")
    _require_version("python", runtime_python_version, toolchain["tools"]["python"])

    receipt = {
        "schema_version": 1,
        "toolchain_lock": digest_json(TOOLCHAIN_LOCK),
        "bootstrap_inputs": digest_paths(BOOTSTRAP_INPUTS),
        "web_dist": sha256_tree(WEB_DIST),
        "python_environment": python_environment_fingerprint(runtime_python),
        "tools": {
            "python": {"path": str(runtime_python), "version": runtime_python_version},
            **{name: {"path": path, "version": versions[name]} for name, path in tools.items()},
        },
    }
    atomic_write_json(BOOTSTRAP_RECEIPT, receipt)
    print(json.dumps({"status": "ok", "tools": versions}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"bootstrap: {error}", file=sys.stderr)
        raise SystemExit(1) from error
