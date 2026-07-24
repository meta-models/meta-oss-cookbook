"""
run_agent.py — a worked example of the Muse Glimmer agent loop on a real task, fully offline.

The agent is pointed at a real directory on your disk and asked to review the code in
it: list the source files, read them, and write a review to disk. Nothing is mocked —
it reads your actual files and produces an artifact you can open. By default it
reviews this cookbook, so it works the moment you clone the repo.

    python run_agent.py --model meta-models/Muse-Glimmer-30B

If --model is omitted, MUSE_GLIMMER_MODEL_PATH from the environment is used.
Point it at your own code with --root:

    python run_agent.py --root ~/code/my-project

This is also the starting point for "make it yours": swap the tools below for your
own, or hand it a different --task.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import pathlib

from agent_loop import MuseGlimmerAgent, ToolRegistry

# --------------------------------------------------------------------------------------
# Define a small, fully-local toolset: read the real filesystem, write to a sandbox.
# --------------------------------------------------------------------------------------
tools = ToolRegistry()

WORKSPACE = pathlib.Path(__file__).parent / "workspace"
# Files the agent is allowed to read. Overridden by main() from --root.
REVIEW_ROOT = pathlib.Path(__file__).parent.parent

MAX_READ_BYTES = 20_000
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "workspace"}


def _resolve_in_root(relative_path: str) -> pathlib.Path | None:
    """Resolve a path inside REVIEW_ROOT, or None if it escapes."""
    target = (REVIEW_ROOT / relative_path).resolve()
    if not str(target).startswith(str(REVIEW_ROOT.resolve())):
        return None
    return target


@tools.tool(
    description=(
        "List source files in the project being reviewed. Returns paths relative to the "
        "project root, along with each file's size in bytes and line count."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob to match filenames, e.g. '*.py' or '*.md'. Defaults to '*.py'.",
            },
        },
        "required": [],
    },
)
def list_files(pattern: str = "*.py") -> str:
    found = []
    for path in sorted(REVIEW_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not fnmatch.fnmatch(path.name, pattern):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        found.append({
            "path": str(path.relative_to(REVIEW_ROOT)),
            "bytes": path.stat().st_size,
            "lines": text.count("\n") + 1,
        })
    if not found:
        return json.dumps({"error": f"no files matched {pattern!r}", "pattern": pattern})
    return json.dumps({"root": str(REVIEW_ROOT), "count": len(found), "files": found})


@tools.tool(
    description=(
        "Read a source file from the project being reviewed. Takes a path relative to the "
        "project root, exactly as returned by list_files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Relative path, e.g. 'agentic-fundamentals/agent_loop.py'"},
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    target = _resolve_in_root(path)
    if target is None:
        return json.dumps({"error": "path escapes the project root"})
    if not target.is_file():
        return json.dumps({"error": f"no such file: {path}",
                           "hint": "call list_files to see valid paths"})
    try:
        text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    encoded = text.encode("utf-8")
    truncated = len(encoded) > MAX_READ_BYTES
    if truncated:
        text = encoded[:MAX_READ_BYTES].decode("utf-8", errors="ignore")
    return json.dumps({"path": path, "truncated": truncated, "content": text})


@tools.tool(
    description="Write text to a file in the agent's sandboxed workspace directory.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Relative filename, e.g. 'code_review.md'"},
            "content": {"type": "string", "description": "The full file content to write"},
        },
        "required": ["filename", "content"],
    },
)
def write_file(filename: str, content: str) -> str:
    WORKSPACE.mkdir(exist_ok=True)
    # Sandbox: keep writes inside WORKSPACE.
    target = (WORKSPACE / filename).resolve()
    if not str(target).startswith(str(WORKSPACE.resolve())):
        return json.dumps({"error": "path escapes workspace sandbox"})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return json.dumps({"ok": True, "path": str(target), "bytes": len(content)})


SYSTEM_PROMPT = (
    "You are a code-review agent working entirely on the local filesystem. Use the "
    "provided tools to inspect real files before you draw any conclusion — never guess "
    "at a file's contents. Think step by step, and when you are done, give the user a "
    "concise final answer."
)

TASK = (
    "Review the Python modules in this project. First list them, then read each one. "
    "Write a markdown code review to 'code_review.md' containing: a table of every "
    "module with its line count and a one-line description of what it does, and a "
    "'Findings' section noting anything that looks like a bug, a rough edge, or a "
    "missing test. Finally, tell me which module carries the most complexity and why."
)

OUTPUT_FILE = "code_review.md"


def main() -> None:
    global REVIEW_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MUSE_GLIMMER_MODEL_PATH"),
                    help="Path or HF id of the Muse Glimmer HF checkpoint.")
    ap.add_argument("--root", default=None,
                    help="Directory to review. Defaults to this cookbook.")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("Provide --model or set MUSE_GLIMMER_MODEL_PATH.")

    if args.root:
        REVIEW_ROOT = pathlib.Path(args.root).expanduser().resolve()
        if not REVIEW_ROOT.is_dir():
            raise SystemExit(f"--root is not a directory: {REVIEW_ROOT}")

    print(f"Reviewing: {REVIEW_ROOT}")
    print(f"Loading Muse Glimmer from {args.model} ...")
    agent = MuseGlimmerAgent(model_path=args.model, tools=tools, system_prompt=SYSTEM_PROMPT,
                      max_steps=args.max_steps, max_new_tokens=args.max_new_tokens).load()
    print(f"\nTASK: {args.task}\n" + "=" * 70)
    final = agent.run(args.task)
    print("\n" + "=" * 70 + "\nFINAL ANSWER:\n" + final)
    report = WORKSPACE / OUTPUT_FILE
    if report.exists():
        print(f"\n--- {report} ---\n{report.read_text()}")


if __name__ == "__main__":
    main()
