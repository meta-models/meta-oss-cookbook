# Local artifacts

This directory contains the runtime inventory and download lock. Models,
complete tokenizer bundles, voice styles, exported programs, and native
binaries are downloaded into ignored `.local/artifacts/` paths.

Each artifact is governed by its own upstream license. The Apache-2.0
license for product-owned source does not apply to those artifacts; for
example, the pinned MLX-generated metallib is MIT-licensed. Review the licenses,
then run `make setup` to download and validate the pinned files.

Setup verifies checksums and payload sizes, then writes
`.local/state/prepared.json`. Directory sizes are the sum of regular-file bytes. The
inventory includes the shared `mlx.metallib` that must be colocated with all
three native executables under `.local/artifacts/bin/`. Daily startup consumes
that receipt and never downloads, builds, or exports assets.
