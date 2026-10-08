# Muse Glimmer Voice Agent

A fully local voice agent for macOS on Apple silicon. The browser captures the
microphone, loopback-only LiveKit carries audio, and local ExecuTorch runtimes
perform Parakeet speech recognition, Muse Glimmer generation, and Supertonic
speech synthesis.

https://github.com/user-attachments/assets/c09cdc7d-91aa-453b-8663-4407d83adcf5

```text
browser microphone
  -> 127.0.0.1 LiveKit
  -> Parakeet ASR
  -> Muse Glimmer LLM
  -> Supertonic TTS
  -> browser speaker
```

No LiveKit Cloud account or cloud inference service is used.

## Models and artifacts

The recipe does not store model files in Git. `make setup` downloads the exact
files below, validates their sizes and SHA-256 hashes, and places them under
ignored `.local/artifacts/` paths.

### Speech-to-text: Parakeet-TDT-0.6B-v3

- Original model: [NVIDIA Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
- ExecuTorch source: [Parakeet export and persistent helper](https://github.com/pytorch/executorch/tree/20ad5ee43ff53804030899d621590af3daadda53/examples/models/parakeet).
- Files downloaded from [Parakeet-TDT-ExecuTorch-MLX](https://huggingface.co/younghan-meta/Parakeet-TDT-ExecuTorch-MLX/tree/b1d26047ee8eae69272a268d5eb85c8bf2de9957):
  [`model.pte`](https://huggingface.co/younghan-meta/Parakeet-TDT-ExecuTorch-MLX/blob/b1d26047ee8eae69272a268d5eb85c8bf2de9957/model.pte)
  and [`tokenizer.model`](https://huggingface.co/younghan-meta/Parakeet-TDT-ExecuTorch-MLX/blob/b1d26047ee8eae69272a268d5eb85c8bf2de9957/tokenizer.model).

### LLM: Muse Glimmer 30B

- Original model: [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B).
- ExecuTorch source: [Muse Glimmer export, runtime, and server](https://github.com/pytorch/executorch/tree/20ad5ee43ff53804030899d621590af3daadda53/examples/models/muse-glimmer).
- Files downloaded from [Muse-Glimmer-30B-ExecuTorch-PTE](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/tree/fc6fa93cdefeddc93abd7abf883e99279af7ea51):
  [K-quant-17G, 128K text, dFlash Metal `model.pte`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/blob/fc6fa93cdefeddc93abd7abf883e99279af7ea51/muse-glimmer-k-quant-17G-128K-text-dflash-metal/muse-glimmer-k-quant-17G-128K-text-dflash-metal.pte),
  [`tokenizer.json`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/blob/fc6fa93cdefeddc93abd7abf883e99279af7ea51/tokenizer.json),
  [`tokenizer_config.json`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/blob/fc6fa93cdefeddc93abd7abf883e99279af7ea51/tokenizer_config.json),
  and [`chat_template.jinja`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/blob/fc6fa93cdefeddc93abd7abf883e99279af7ea51/chat_template.jinja).

### Text-to-speech: Supertonic-3

- Original model and assets: [Supertone Supertonic-3](https://huggingface.co/Supertone/supertonic-3/tree/3cadd1ee6394adea1bd021217a0e650ede09a323).
- ExecuTorch source: [Supertonic export and persistent runtime](https://github.com/pytorch/executorch/tree/20ad5ee43ff53804030899d621590af3daadda53/examples/models/supertonic).
- ExecuTorch model: [`model.pte`](https://huggingface.co/younghan-meta/Supertonic-ExecuTorch-MLX/blob/fecf95fa31ab6b26262dae0414e657c8f6580427/model.pte)
  from [Supertonic-ExecuTorch-MLX](https://huggingface.co/younghan-meta/Supertonic-ExecuTorch-MLX/tree/fecf95fa31ab6b26262dae0414e657c8f6580427).
- Runtime assets downloaded from the original model repository:
  [`config.json`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/config.json),
  [`onnx/duration_predictor.onnx`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/duration_predictor.onnx),
  [`onnx/text_encoder.onnx`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/text_encoder.onnx),
  [`onnx/tts.json`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/tts.json),
  [`onnx/unicode_indexer.json`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/unicode_indexer.json),
  [`onnx/vector_estimator.onnx`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/vector_estimator.onnx),
  [`onnx/vocoder.onnx`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/vocoder.onnx), and
  [`voice_styles/F1.json`](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/voice_styles/F1.json).

### Native ExecuTorch runtime

- Source: pinned [ExecuTorch commit `20ad5ee`](https://github.com/pytorch/executorch/commit/20ad5ee43ff53804030899d621590af3daadda53).
- Download: [`runtime-macos-arm64-executorch-20ad5ee.tar.gz`](https://huggingface.co/younghan-meta/Muse-Glimmer-Voice-Agent-ExecuTorch/blob/15200ebe3c37aeb4a6aec43021ceb6025f41f6a6/runtime-macos-arm64-executorch-20ad5ee.tar.gz)
  from [Muse-Glimmer-Voice-Agent-ExecuTorch](https://huggingface.co/younghan-meta/Muse-Glimmer-Voice-Agent-ExecuTorch/tree/15200ebe3c37aeb4a6aec43021ceb6025f41f6a6).
- Extracted files: `bin/parakeet_helper`, `bin/muse_glimmer_worker`,
  `bin/supertonic_runner`, and `bin/mlx.metallib`.

For the complete supply chain, see:

- [`artifacts/macos-arm64.download.json`](artifacts/macos-arm64.download.json)
  for each Hugging Face repository, immutable revision, and filename.
- [`artifacts/macos-arm64.lock.json`](artifacts/macos-arm64.lock.json) for each
  local destination, license, size, and SHA-256 hash.
- [`stt.py`](packages/livekit-plugins-executorch/livekit/plugins/executorch/stt.py),
  [`launch.py`](apps/muse-glimmer-server/launch.py), and
  [`supertonic_tts.py`](packages/livekit-plugins-executorch/livekit/plugins/executorch/supertonic_tts.py)
  for the recipe integration code.

## Recipe banner

| | |
|---|---|
| Precision | K-quant-17G |
| Model server | ExecuTorch on MLX/Metal |
| Offline? | Yes. Parakeet ASR, Muse Glimmer, and Supertonic all run locally after artifact preparation. |
| Requires | macOS 26 or later on Apple silicon, Python 3.13, Node.js 22, Git, uv, and LiveKit Server 1.x |

## Status

The native compatibility pin is ExecuTorch
`20ad5ee43ff53804030899d621590af3daadda53`, which contains the landed
Supertonic runtime, bounded MuseGlimmer worker cancellation, and persistent
Supertonic JSONL mode. Release readiness remains false until clean-machine
macOS arm64 end-to-end validation is complete. See `docs/upstream-pins.md`.

## Supported platform

- macOS 26 or later on Apple silicon
- Python 3.13
- Node.js 22
- Git and uv
- LiveKit Server 1.x

Other platforms are not part of the first milestone.

## Quickstart

Run application commands from the recipe root:

```bash
cd recipes/voice-agent-macos
```

Review the independent model and runtime licenses, then install the pinned
ExecuTorch checkout, native runtime, and model artifacts:

```bash
make setup
make dev
```

The first setup downloads about 20.8 GB into ignored `.local/` paths. Downloads
are pinned to immutable Hugging Face revisions and are checked against the
recorded sizes and SHA-256 hashes. `make setup` also installs source
dependencies, builds the web UI, and writes compatibility receipts. It does
not compile the models or native runners.

## Daily development

```bash
make dev
make status
make logs
make restart
make down
```

`make dev up` is also supported and starts the stack exactly once. Once
artifacts have been prepared, startup requires no external network access.

The UI opens at `http://127.0.0.1:5173`.

## Local security boundary

- LiveKit signaling: `127.0.0.1:7880`
- LiveKit media: `127.0.0.1:7882/udp`
- Token service: `127.0.0.1:8787`
- Browser UI: `127.0.0.1:5173`
- MuseGlimmer server: backend-only `127.0.0.1:8000`
- Short-lived participant tokens grant microphone publication only.
- Runtime LiveKit credentials are generated locally per stack run.
- Browser code never receives model identifiers, artifact paths, the LLM
  endpoint, native worker details, or server credentials.

See `docs/security-model.md` for the local-process trust model.

## Make it yours

- Change the agent instructions in `apps/worker/src/muse_glimmer_worker/config.py`.
- Tune generation limits and speech settings through the documented worker
  environment variables.
- Replace the Supertonic voice style with another locally prepared compatible
  artifact and update the artifact lock.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `bootstrap receipt is stale` | Source or dependency inputs changed | Run `make bootstrap` again. |
| A required port is already in use | Another local process owns a stack port | Stop that process or run `make down` for a managed stack. |
| Artifact preparation fails | A native binary, model, tokenizer, or checksum does not match the lock | Run `make fetch-artifacts`, then `make prepare-artifacts`. |
| Conversation remains at `Joining` | LiveKit or the worker is unavailable | Run `make status`, then inspect `make logs`. |

## Development checks

From `recipes/voice-agent-macos`:

```bash
make check
make test
make publication-check
```

The publication check rejects secrets, models, native binaries, recordings,
generated output, nested repositories, absolute workstation paths, internal
URLs, and AGPL avatar dependencies.

## License

Product-owned source is Apache-2.0. Models, exported programs, native
binaries, fonts, and third-party packages retain their independent licenses.
See `LICENSE`, `PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.

## Next steps

- Learn how the Muse Glimmer ExecuTorch export and server work in
  [`../../inference-server/executorch.md`](../../inference-server/executorch.md).
- Compare other complete agents in [`../`](../).
