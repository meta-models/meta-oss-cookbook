from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.repository import (
    ARTIFACT_LOCK,
    DOWNLOAD_LOCK,
    LOCAL,
    ROOT,
    SOURCE_DIR,
    artifact_size,
    ensure_local_directories,
    landed_gate_commits,
    read_json,
    relative_local_path,
    require_supported_platform,
    sha256_file,
    sha256_tree,
    validate_executorch_checkout,
)

Download = Callable[..., str]


def _artifact_map() -> dict[str, dict[str, Any]]:
    return {item["role"]: item for item in read_json(ARTIFACT_LOCK)["artifacts"]}


def _verify(path: Path, expected: dict[str, Any], label: str) -> None:
    if expected["kind"] == "file":
        if not path.is_file():
            raise RuntimeError(f"downloaded artifact is not a file: {label}")
        actual_hash = sha256_file(path)
    else:
        if not path.is_dir():
            raise RuntimeError(f"downloaded artifact is not a directory: {label}")
        actual_hash = sha256_tree(path)
    if actual_hash != expected["sha256"]:
        raise RuntimeError(f"checksum mismatch for downloaded artifact: {label}")
    if artifact_size(path) != expected["size_bytes"]:
        raise RuntimeError(f"size mismatch for downloaded artifact: {label}")


def _link_or_copy(source: Path, destination: Path, *, executable: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        if executable:
            shutil.copy2(source, temporary)
            temporary.chmod(temporary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download(download: Download, item: dict[str, Any]) -> Path:
    try:
        return Path(
            download(
                repo_id=item["repository"],
                revision=item["revision"],
                filename=item["filename"],
            )
        )
    except Exception as error:
        repository = item["repository"]
        raise RuntimeError(
            f"could not download {repository}/{item['filename']}; "
            "run `hf auth login` if the repository requires access"
        ) from error


def _install_file(source: Path, expected: dict[str, Any], role: str) -> None:
    _verify(source, expected, role)
    destination = relative_local_path(expected["destination"])
    if destination.exists():
        try:
            _verify(destination, expected, role)
            if not expected["executable"] or os.access(destination, os.X_OK):
                print(f"setup: reusing {role}")
                return
        except RuntimeError:
            pass
    _link_or_copy(source, destination, executable=expected["executable"])
    _verify(destination, expected, role)
    print(f"setup: installed {role}")


def _safe_extract(archive: Path, destination: Path, *, allowed_files: set[str]) -> None:
    with tarfile.open(archive, mode="r:gz") as bundle:
        files: set[str] = set()
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError(f"unsafe runtime bundle member: {member.name}")
            if member.isfile():
                if member.name not in allowed_files:
                    raise RuntimeError(f"unexpected runtime bundle member: {member.name}")
                files.add(member.name)
        if files != allowed_files:
            missing = sorted(allowed_files - files)
            raise RuntimeError(f"runtime bundle is missing required members: {missing}")
        bundle.extractall(destination, filter="data")


def _install_runtime_bundle(
    download: Download,
    specification: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> set[str]:
    archive = _download(download, specification)
    if archive.stat().st_size != specification["size_bytes"]:
        raise RuntimeError("runtime bundle size does not match the download lock")
    if sha256_file(archive) != specification["sha256"]:
        raise RuntimeError("runtime bundle checksum does not match the download lock")

    roles = set(specification["members"])
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=LOCAL) as temporary:
        extracted = Path(temporary)
        license_members = {
            "LICENSES/EXECUTORCH-BSD-3-CLAUSE.txt",
            "LICENSES/MLX-MIT.txt",
        }
        _safe_extract(
            archive,
            extracted,
            allowed_files=set(specification["members"].values()) | license_members,
        )
        for role, member in specification["members"].items():
            _install_file(extracted / member, artifacts[role], role)
        licenses = extracted / "LICENSES"
        if not licenses.is_dir():
            raise RuntimeError("runtime bundle has no LICENSES directory")
        for license_file in licenses.iterdir():
            if license_file.is_file():
                _link_or_copy(license_file, LOCAL / "artifacts/LICENSES" / license_file.name)
    return roles


def _install_tree(
    download: Download,
    specification: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    destination = relative_local_path(expected["destination"])
    if destination.exists():
        try:
            _verify(destination, expected, specification["role"])
            print(f"setup: reusing {specification['role']}")
            return
        except RuntimeError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assets-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "tree"
        for item in specification["files"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"artifact tree path escapes its root: {item['path']}")
            source = _download(
                download,
                {
                    "repository": specification["repository"],
                    "revision": specification["revision"],
                    "filename": item["filename"],
                },
            )
            _link_or_copy(source, staged / Path(*relative.parts))
        _verify(staged, expected, specification["role"])
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)
    print(f"setup: installed {specification['role']}")


def ensure_checkout() -> None:
    require_supported_platform()
    ensure_local_directories()
    compatibility = read_json(ROOT / "config/dependencies/compatibility.lock.json")
    executorch = compatibility["executorch"]
    revision = executorch["commit"]
    override = os.environ.get("GLIMMER_EXECUTORCH_ROOT")
    checkout = Path(override).expanduser().resolve() if override else SOURCE_DIR / "executorch"
    if checkout.exists():
        validate_executorch_checkout(
            checkout,
            revision,
            required_ancestors=landed_gate_commits(compatibility),
        )
        print(f"setup: reusing ExecuTorch {revision}")
        return
    if override:
        raise RuntimeError(f"ExecuTorch checkout is missing: {checkout}")

    temporary = SOURCE_DIR / f".executorch.{os.getpid()}.partial"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                executorch["repository"],
                temporary,
            ],
            check=True,
        )
        subprocess.run(["git", "-C", temporary, "checkout", "--detach", revision], check=True)
        validate_executorch_checkout(
            temporary,
            revision,
            required_ancestors=landed_gate_commits(compatibility),
        )
        os.replace(temporary, checkout)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"setup: installed ExecuTorch {revision}")


def install_artifacts(*, download: Download | None = None) -> None:
    require_supported_platform()
    ensure_local_directories()
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download

    specification = read_json(DOWNLOAD_LOCK)
    artifacts = _artifact_map()
    handled = _install_runtime_bundle(download, specification["runtime_bundle"], artifacts)

    for item in specification["files"]:
        role = item["role"]
        _install_file(_download(download, item), artifacts[role], role)
        handled.add(role)

    for item in specification["trees"]:
        role = item["role"]
        _install_tree(download, item, artifacts[role])
        handled.add(role)

    if handled != set(artifacts):
        missing = sorted(set(artifacts) - handled)
        extra = sorted(handled - set(artifacts))
        raise RuntimeError(
            f"download roles do not match artifact roles: missing={missing}, extra={extra}"
        )

    for item in specification["license_files"]:
        source = _download(download, item)
        _link_or_copy(source, relative_local_path(item["destination"]))
    print("setup: downloaded and verified all artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the pinned local voice-agent runtime")
    parser.add_argument("operation", choices=("checkout", "artifacts"))
    args = parser.parse_args()
    if args.operation == "checkout":
        ensure_checkout()
    else:
        install_artifacts()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"setup: {error}", file=sys.stderr)
        raise SystemExit(1) from error
