# macOS Apple Silicon native targets

Native binaries are built from the single ExecuTorch checkout pinned by
`config/dependencies/compatibility.lock.json`. They are installed under the
ignored `.local/artifacts/bin/` directory and are never committed.

The supported milestone requires a MuseGlimmer worker that advertises
`supports_cancel` and a Supertonic runner with `--server_jsonl`. Startup fails
if either capability is unavailable.
