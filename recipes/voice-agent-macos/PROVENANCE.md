# Provenance

## Canonical source

This macOS example is product-owned source in the canonical
[`meta-models/meta-oss-cookbook`](https://github.com/meta-models/meta-oss-cookbook)
repository under `recipes/voice-agent-macos`. Product-owned source is licensed
under Apache-2.0 as described in `LICENSE`.

The source snapshot used for this migration is
`12c7c7e85b042ef8a949acb5c0c9b792a53a86cc` from
[`meta-pytorch/executorch-examples#258`](https://github.com/meta-pytorch/executorch-examples/pull/258).
That snapshot records pre-publication development history; canonical
maintenance is in `meta-models/meta-oss-cookbook`.

## Original source and API integrations

The application, token service, worker, web UI, launchers, packaging, lifecycle
code, and LiveKit ExecuTorch adapters are original product source. They
integrate with public LiveKit APIs but do not copy LiveKit implementation
source.

The worker and adapter integrations were developed against the public
[`livekit/agents`](https://github.com/livekit/agents) API at commit
`bc5f3df3a2bd1b3b8c5d1df742be57b063374991`. Package-specific source mappings
and integration details are recorded in:

- `apps/worker/PROVENANCE.md`
- `packages/livekit-plugins-executorch/PROVENANCE.md`

The token service integrates with the LiveKit API to issue short-lived tokens;
it does not include LiveKit implementation source.

## Excluded components

This source subtree does not include ExecuTorch source, native runners, model
weights, exported programs, tokenizers, voice styles, recordings, generated
output, or dependency source. Those components retain their independent
licenses and notices as documented in `THIRD_PARTY_NOTICES.md` and `LICENSES/`.

Muse Glimmer, ExecuTorch, LiveKit, and other names may be trademarks of their
respective owners. The Apache-2.0 license does not grant trademark rights.
