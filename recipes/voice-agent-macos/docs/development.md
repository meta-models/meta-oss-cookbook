# Development

The repository separates source setup, artifact preparation, and daily runtime:

```bash
make bootstrap
make prepare-artifacts
make dev
```

Use `make check` for static and publication checks, `make test` for unit tests
and the production web build, and `make e2e` for model-heavy macOS integration.

Bootstrap records the locked dependency inputs, validated tool paths, and the
production web build digest. Daily startup rejects stale setup state and also
revalidates that the prepared ExecuTorch checkout remains clean at the exact
compatibility commit; it never installs or rebuilds.

Do not place secrets in `.env` files. The supervisor creates ephemeral local
LiveKit credentials. Do not add cloud provider fallbacks or browser-configured
model endpoints.
