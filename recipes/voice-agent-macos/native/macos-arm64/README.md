# macOS Apple Silicon native targets

Native binaries are built from the single ExecuTorch checkout pinned by
`config/dependencies/compatibility.lock.json`. The setup command downloads the
verified release bundle into the ignored `.local/artifacts/bin/` directory;
maintainers can reproduce it from the same checkout.

Rebuilding the bundle requires CMake 3.24 or later and a compatible Xcode
toolchain. These build tools are not required when using the prebuilt bundle.

The supported milestone requires a MuseGlimmer worker that advertises
`supports_cancel` and a Supertonic runner with `--server_jsonl`. Startup fails
if either capability is unavailable.
