# Contributing

This subtree follows the repository contribution requirements in
[`../../CONTRIBUTING.md`](../../CONTRIBUTING.md), including its licensing and
Contributor License Agreement terms. The guidance below is specific to the
local-only, source-only macOS example.

- Do not commit models, native binaries, credentials, recordings, logs, caches,
  generated output, or another repository.
- Keep browser-visible data within the policy documented in
  `docs/security-model.md`.
- Keep ASR, LLM, and TTS native components on one pinned ExecuTorch revision.
- Preserve MuseGlimmer reasoning configuration under
  `chat_template_kwargs.reasoning_strength`; do not use `reasoning_effort`.
- Run `make check`, `make test`, and `make publication-check` before opening a
  pull request.
