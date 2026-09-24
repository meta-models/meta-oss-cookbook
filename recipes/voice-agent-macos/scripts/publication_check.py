from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from scripts.repository import ROOT

_DENIED_SUFFIXES = {
    ".a",
    ".bin",
    ".dylib",
    ".gguf",
    ".metallib",
    ".onnx",
    ".o",
    ".pcm",
    ".pem",
    ".pt",
    ".ptd",
    ".pte",
    ".safetensors",
    ".so",
    ".wav",
}
_DENIED_PARTS = {
    ".dev-stack",
    ".local",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "recordings",
    "reports",
    "museglimmer-reports",
}
_DENIED_NAMES = {".env", ".env.cloud.disabled", "livekit.keys"}
_DENIED_CONTENT = {
    "absolute user path": re.compile(b"/" + b"Users" + b"/"),
    "internal URL": re.compile((b"internal" + b"fb\\.com|fburl\\.com"), re.IGNORECASE),
    "AGPL avatar package": re.compile((b"@bible-strong/" + b"avatar"), re.IGNORECASE),
    "private key": re.compile(b"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}
_MAX_FILE_BYTES = 5 * 1024 * 1024


def _git_files(*arguments: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *arguments, "-z", "--", "."],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    root = ROOT.resolve()
    files: list[Path] = []
    for value in result.stdout.split(b"\0"):
        if not value:
            continue
        relative = Path(os.fsdecode(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Git candidate escapes application root: {relative}")
        files.append(root / relative)
    return files


def _candidate_files() -> list[Path]:
    return _git_files("--cached", "--others", "--exclude-standard")


def _check_path(path: Path) -> list[str]:
    relative = path.relative_to(ROOT)
    errors: list[str] = []
    if path.is_symlink():
        errors.append(f"symlink is not allowed: {relative}")
        return errors
    if any(part in _DENIED_PARTS for part in relative.parts):
        errors.append(f"generated/private path is not allowed: {relative}")
    if path.name in _DENIED_NAMES or (path.name.startswith(".env") and path.name != ".env.example"):
        errors.append(f"environment or credential file is not allowed: {relative}")
    if path.suffix.lower() in _DENIED_SUFFIXES:
        errors.append(f"binary/model artifact is not allowed: {relative}")
    if not path.is_file():
        return errors
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        errors.append(f"file exceeds {_MAX_FILE_BYTES} bytes: {relative}")
        return errors
    payload = path.read_bytes()
    for label, pattern in _DENIED_CONTENT.items():
        if pattern.search(payload):
            errors.append(f"{label} found in {relative}")
    assignment = re.compile(
        rb"^(?:export\s+)?(?:LIVEKIT_API_SECRET|OPENAI_API_KEY|AWS_SECRET_ACCESS_KEY)\s*=\s*(.+)$"
    )
    for raw_line in payload.splitlines():
        match = assignment.fullmatch(raw_line.rstrip())
        if match is None:
            continue
        value = match.group(1).strip().strip(b"\"'")
        if value and not value.startswith((b"test-", b"<", b"${")):
            errors.append(f"credential assignment found in {relative}")
            break
    return errors


def _nested_repositories() -> list[Path]:
    nested = []
    for current, directories, files in os.walk(ROOT):
        path = Path(current)
        if path == ROOT:
            directories[:] = [name for name in directories if name not in {".git", ".local"}]
            continue
        if ".git" in directories:
            nested.append(path / ".git")
            directories.remove(".git")
        if ".git" in files:
            nested.append(path / ".git")
        directories[:] = [name for name in directories if name != ".local"]
    return nested


def main() -> int:
    parser = argparse.ArgumentParser(description="Reject files unsafe for public source control")
    parser.add_argument("--tracked-only", action="store_true")
    args = parser.parse_args()
    files = _candidate_files()
    if args.tracked_only:
        allowed = set(_git_files("--cached"))
        if not allowed:
            print(
                "publication-check: application subtree contains no tracked files", file=sys.stderr
            )
            return 1
        files = [path for path in files if path in allowed]
    errors = [error for path in files for error in _check_path(path)]
    errors.extend(
        f"nested repository is not allowed: {path.relative_to(ROOT)}"
        for path in _nested_repositories()
    )
    if errors:
        for error in errors:
            print(f"publication-check: {error}", file=sys.stderr)
        return 1
    print(f"Publication check passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"publication-check: {error}", file=sys.stderr)
        raise SystemExit(1) from error
