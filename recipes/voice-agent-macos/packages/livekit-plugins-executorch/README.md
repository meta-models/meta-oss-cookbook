# LiveKit ExecuTorch plugin

Local, source-only adapters for LiveKit Agents:

- `executorch.STT` runs batch Parakeet ASR through one persistent framed helper.
- `executorch.SupertonicTTS` owns one persistent `supertonic_runner --server_jsonl`
  process and sends synthesis text only through stdin JSONL.

Native binaries, model weights, tokenizers, voice styles, recordings, and generated
outputs are deliberately outside this package. Supply explicit local artifact paths
when constructing either provider.

The adapters serialize requests because each native helper accepts one active request.
Timeout, cancellation, protocol failure, and explicit close all terminate and reap the
helper within configured bounds.

Model weights and voice/style assets retain their upstream licenses. Supertonic 3 is
distributed under the OpenRAIL-M license described by its model card; review it before
redistributing model assets or generated output.
