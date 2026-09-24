from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from scripts import setup


def test_install_file_verifies_and_sets_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"runtime")
    destination = tmp_path / "artifacts" / "bin" / "runtime"
    monkeypatch.setattr(setup, "relative_local_path", lambda _value: destination)
    expected = {
        "kind": "file",
        "destination": ".local/artifacts/bin/runtime",
        "sha256": setup.sha256_file(source),
        "size_bytes": source.stat().st_size,
        "executable": True,
    }

    setup._install_file(source, expected, "runtime")

    assert destination.read_bytes() == b"runtime"
    assert os.access(destination, os.X_OK)


def test_safe_extract_rejects_path_escape(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(RuntimeError, match="unsafe runtime bundle member"):
        setup._safe_extract(archive, tmp_path / "output", allowed_files={"bin/runtime"})


def test_safe_extract_rejects_unexpected_file(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("bin/unexpected")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(RuntimeError, match="unexpected runtime bundle member"):
        setup._safe_extract(archive, tmp_path / "output", allowed_files={"bin/runtime"})


def test_download_error_explains_private_repository_authentication() -> None:
    def fail(**_kwargs: str) -> str:
        raise OSError("denied")

    with pytest.raises(RuntimeError, match="hf auth login"):
        setup._download(
            fail,
            {
                "repository": "owner/private-repo",
                "revision": "a" * 40,
                "filename": "artifact",
            },
        )


def test_checkout_clones_and_detaches_at_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / ".local" / "src"
    revision = "a" * 40
    calls: list[list[object]] = []
    validations: list[tuple[Path, str, tuple[str, ...]]] = []
    monkeypatch.delenv("GLIMMER_EXECUTORCH_ROOT", raising=False)
    monkeypatch.setattr(setup, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(setup, "require_supported_platform", lambda: None)
    monkeypatch.setattr(setup, "ensure_local_directories", lambda: source_dir.mkdir(parents=True))
    monkeypatch.setattr(
        setup,
        "read_json",
        lambda _path: {
            "executorch": {
                "repository": "https://github.com/pytorch/executorch.git",
                "commit": revision,
                "gates": {},
            }
        },
    )

    def run(command: list[object], *, check: bool) -> None:
        assert check
        calls.append(command)
        if "clone" in command:
            Path(command[-1]).mkdir(parents=True)

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(
        setup,
        "validate_executorch_checkout",
        lambda checkout, commit, *, required_ancestors: validations.append(
            (checkout, commit, required_ancestors)
        ),
    )

    setup.ensure_checkout()

    checkout = source_dir / "executorch"
    temporary = source_dir / f".executorch.{os.getpid()}.partial"
    assert calls[0][:4] == ["git", "clone", "--filter=blob:none", "--no-checkout"]
    assert calls[1] == ["git", "-C", temporary, "checkout", "--detach", revision]
    assert validations == [(temporary, revision, ())]
    assert checkout.is_dir()
