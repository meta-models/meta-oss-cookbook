# Local artifacts

This directory contains manifests only. Models, complete tokenizer bundles,
voice styles, exported programs, and native binaries live under ignored
`.local/artifacts/`.

Each artifact is governed by its own upstream license. The Apache-2.0
license for product-owned source does not apply to those artifacts; for
example, the pinned MLX-generated metallib is MIT-licensed. Run
`make prepare-artifacts` after reviewing the licenses and providing any
artifacts marked `user-provided` in `macos-arm64.lock.json`.

Preparation verifies checksums and payload sizes, then writes
`.local/state/prepared.json`. Directory sizes are the sum of regular-file bytes. The
inventory includes the shared `mlx.metallib` that must be colocated with all
three native executables under `.local/artifacts/bin/`. Daily startup consumes
that receipt and never downloads, builds, or exports assets.
