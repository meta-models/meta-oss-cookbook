# Muse Glimmer worker

An installable, local-only LiveKit worker for the Muse Glimmer voice application.
It uses local Parakeet and Supertonic ExecuTorch providers and an OpenAI-compatible
Muse Glimmer endpoint fixed at `http://127.0.0.1:8000/v1`.

After installing the workspace, run the worker with:

```bash
muse-glimmer-worker dev
```

The worker accepts only `ws://127.0.0.1:7880` for LiveKit and requires local artifact
paths through `PARAKEET_*` and `SUPERTONIC_*` environment variables. Credentials are
read from the environment and are never included in diagnostics.

For deployment checks or a direct WAV pipeline, use `muse-glimmer-diagnostics doctor`
or `muse-glimmer-diagnostics pipeline INPUT.wav`. Diagnostic ZIPs omit the local
raw runtime log by default; do not attach local logs to public issues.
