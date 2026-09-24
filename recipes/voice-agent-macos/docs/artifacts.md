# Artifact Preparation

`artifacts/macos-arm64.lock.json` is the source-of-truth inventory. Every entry
records its role, distribution method, independent license, ignored
`.local/artifacts` destination, immutable revision, checksum, and payload size.
Directory sizes are the sum of their regular-file bytes.

Artifacts marked `user-provided` are not downloaded automatically. Obtain them
under their upstream terms, place them at the documented destination, and run:

```bash
make prepare-artifacts
```

Preparation accepts only ExecuTorch
`20ad5ee43ff53804030899d621590af3daadda53` selected in
`config/dependencies/compatibility.lock.json`, rejects a dirty or mismatched
checkout, validates all files, and writes `.local/state/prepared.json`.
The manifest also tracks the shared `mlx.metallib` beside the three native
executables because statically linked MLX discovers that file at runtime. Its
provenance is the pinned MIT-licensed MLX submodule used by the ExecuTorch build.
Startup verifies the receipt and every checksum. It never installs, builds,
downloads, exports, or repairs artifacts.
