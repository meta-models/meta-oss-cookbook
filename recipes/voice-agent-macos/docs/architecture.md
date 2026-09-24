# Architecture

## Runtime flow

1. The React application requests a short-lived token from the local token
   service only after the user selects **Start conversation**.
2. The browser connects to loopback LiveKit and publishes microphone audio.
3. The `assistant` worker receives audio and uses Silero VAD with the persistent
   Parakeet ExecuTorch helper.
4. Final transcripts are sent to the loopback MuseGlimmer OpenAI-compatible
   server. One native worker owns one loaded model and reusable sessions.
5. The worker passes visible response text to a persistent Supertonic JSONL
   runner. The model is loaded and warmed once.
6. Generated PCM audio is published through LiveKit to the browser.

## Process ownership

The repository supervisor owns five process groups in dependency order:
MuseGlimmer, LiveKit, token service, production web server, and worker. It
stores identity-qualified state under `.local/run`, rolls startup failures back
in reverse order, and never kills a process based on PID alone. Shutdown owns the
recorded process groups even after a launcher exits, and worker status probes the
dynamic loopback health endpoint reported by LiveKit Agents.

## Dependency boundary

LiveKit and ExecuTorch are external dependencies. Their repositories are not
vendored. One compatibility lock pins all ExecuTorch Python and native pieces
together. Models, builds, and checkouts live under ignored `.local` paths.

The temporary `packages/livekit-plugins-executorch` package is product-owned
until a compatible upstream package is released. Its provenance file records
the precise source and removal condition.
