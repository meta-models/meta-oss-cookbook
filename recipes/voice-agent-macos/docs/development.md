# Development

The one-time setup downloads pinned artifacts and prepares source dependencies;
daily startup only launches already verified files:

```bash
make setup
make dev
```

Use `make check` for static and publication checks, `make test` for unit tests
and the production web build, and `make e2e` for model-heavy macOS integration.

Setup installs the exact ExecuTorch checkout, downloads the model and runtime
artifacts from immutable Hugging Face revisions, verifies their hashes, and
records the dependency and artifact receipts. Daily startup rejects stale
state and revalidates the checkout and artifacts; it never installs or
rebuilds.

Do not place secrets in `.env` files. The supervisor creates ephemeral local
LiveKit credentials. Do not add cloud provider fallbacks or browser-configured
model endpoints.
