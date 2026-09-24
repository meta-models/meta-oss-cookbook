from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.repository import (
    ARTIFACT_LOCK,
    COMPATIBILITY_LOCK,
    PREPARED_RECEIPT,
    ROOT,
    TOOLCHAIN_LOCK,
    artifact_size,
    atomic_write_json,
    digest_json,
    ensure_local_directories,
    landed_gate_commits,
    read_json,
    relative_local_path,
    require_supported_platform,
    sha256_file,
    sha256_tree,
    validate_executorch_checkout,
)


def main() -> int:
    require_supported_platform()
    ensure_local_directories()
    compatibility = read_json(COMPATIBILITY_LOCK)
    expected_commit = compatibility["executorch"].get("commit")
    if not expected_commit:
        raise RuntimeError(
            "artifact preparation is gated: set one immutable ExecuTorch commit "
            "containing supports_cancel and supertonic_server_jsonl"
        )

    checkout = (
        Path(os.environ.get("GLIMMER_EXECUTORCH_ROOT", ROOT / ".local/src/executorch"))
        .expanduser()
        .resolve()
    )
    validate_executorch_checkout(
        checkout,
        expected_commit,
        required_ancestors=landed_gate_commits(compatibility),
    )

    manifest = read_json(ARTIFACT_LOCK)
    inventory: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for item in manifest["artifacts"]:
        path = relative_local_path(item["destination"])
        if not path.exists():
            missing.append(f"{item['role']}: {item['destination']} ({item['prepare']})")
            continue
        if item["kind"] == "file" and not path.is_file():
            raise RuntimeError(f"{item['role']} must be a regular file")
        if item["kind"] == "directory" and not path.is_dir():
            raise RuntimeError(f"{item['role']} must be a directory")
        if item["executable"] and not os.access(path, os.X_OK):
            raise RuntimeError(f"{item['role']} must be executable")
        checksum = sha256_tree(path) if path.is_dir() else sha256_file(path)
        expected_checksum = item.get("sha256")
        if expected_checksum and checksum != expected_checksum:
            raise RuntimeError(f"checksum mismatch for {item['role']}")
        size_bytes = artifact_size(path)
        expected_size = item.get("size_bytes")
        if expected_size is not None and size_bytes != expected_size:
            raise RuntimeError(f"size mismatch for {item['role']}")
        inventory[item["role"]] = {
            "path": item["destination"],
            "sha256": checksum,
            "size_bytes": size_bytes,
        }
    if missing:
        details = "\n  ".join(missing)
        raise RuntimeError(f"required artifacts are missing:\n  {details}")

    receipt = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "executorch_commit": expected_commit,
        "executorch_checkout": str(checkout),
        "locks": {
            "compatibility": digest_json(COMPATIBILITY_LOCK),
            "artifacts": digest_json(ARTIFACT_LOCK),
            "toolchain": digest_json(TOOLCHAIN_LOCK),
        },
        "artifacts": inventory,
    }
    atomic_write_json(PREPARED_RECEIPT, receipt)
    print(json.dumps({"status": "prepared", "receipt": str(PREPARED_RECEIPT)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"prepare-artifacts: {error}", file=sys.stderr)
        raise SystemExit(1) from error
