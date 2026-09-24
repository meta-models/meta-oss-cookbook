# MuseGlimmer server launcher

`launch.py` validates the prepared artifact receipt and directly executes the
OpenAI-compatible server from the single pinned ExecuTorch checkout. It is not
a proxy and does not reimplement the upstream API.

The launcher forces `127.0.0.1:8000`, a 131072-token context limit, DFlash
artifact mode, no tool parser, and the prepared cancellation-capable native
worker.
