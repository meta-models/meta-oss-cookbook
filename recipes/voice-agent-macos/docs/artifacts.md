# Artifact Preparation

`artifacts/macos-arm64.lock.json` is the source-of-truth inventory. Every entry
records its role, distribution method, independent license, ignored
`.local/artifacts` destination, immutable revision, checksum, and payload size.
`artifacts/macos-arm64.download.json` maps those roles to immutable Hugging Face
revisions. Directory sizes are the sum of their regular-file bytes.

Review the upstream license terms, then download and validate everything with:

```bash
make setup
```

Setup accepts only ExecuTorch
`20ad5ee43ff53804030899d621590af3daadda53` selected in
`config/dependencies/compatibility.lock.json`, rejects a dirty or mismatched
checkout, validates every downloaded file, and writes
`.local/state/prepared.json`.
The manifest also tracks the shared `mlx.metallib` beside the three native
executables because statically linked MLX discovers that file at runtime. Its
provenance is the pinned MIT-licensed MLX submodule used by the ExecuTorch build.
Startup verifies the receipt and every checksum. It never installs, builds,
downloads, exports, or repairs artifacts. Maintainers can still reproduce the
native binaries and converted PTEs from the pinned upstream revisions before
publishing a new artifact set.
