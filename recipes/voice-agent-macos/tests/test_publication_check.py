from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import publication_check


def test_rejects_model_and_environment_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    model = tmp_path / "model.pte"
    model.write_bytes(b"artifact")
    environment = tmp_path / ".env"
    environment.write_text("LIVEKIT_API_SECRET=real-secret\n")

    assert publication_check._check_path(model)
    assert publication_check._check_path(environment)


def test_allows_explicit_test_fixture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    source = tmp_path / "test_config.py"
    source.write_text('value = "LIVEKIT_API_SECRET=test-secret"\n')

    assert publication_check._check_path(source) == []


def test_detects_nested_git_metadata_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    nested = tmp_path / "dependency"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: ../.git/modules/dependency\n")

    assert publication_check._nested_repositories() == [nested / ".git"]


def test_tracked_only_rejects_repository_without_tracked_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["publication-check", "--tracked-only"])
    monkeypatch.setattr(
        publication_check.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b""),
    )

    assert publication_check.main() == 1


def test_rejects_literal_private_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    source = tmp_path / "config.py"
    source.write_bytes(b"root = " + b'"/' + b"Users/example/models" + b'"')

    assert any("absolute user path" in error for error in publication_check._check_path(source))


def test_candidate_files_are_scoped_to_nested_application(tmp_path: Path, monkeypatch) -> None:
    application = tmp_path / "recipes" / "voice-agent-macos"
    application.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    inside = application / "inside.py"
    inside.write_text("safe = True\n")
    outside = tmp_path / "sibling.env"
    outside.write_text("LIVEKIT_API_SECRET=real-secret\n")
    subprocess.run(
        ["git", "add", "recipes/voice-agent-macos/inside.py", "sibling.env"],
        cwd=tmp_path,
        check=True,
    )
    untracked = application / "untracked.py"
    untracked.write_text("safe = True\n")
    monkeypatch.setattr(publication_check, "ROOT", application)

    assert set(publication_check._candidate_files()) == {inside.resolve(), untracked.resolve()}
    assert publication_check._git_files("--cached") == [inside.resolve()]


def test_git_candidate_cannot_escape_application_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    monkeypatch.setattr(
        publication_check.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=b"../secret\0"),
    )

    try:
        publication_check._candidate_files()
    except RuntimeError as error:
        assert "escapes application root" in str(error)
    else:
        raise AssertionError("escaping Git candidate was accepted")
