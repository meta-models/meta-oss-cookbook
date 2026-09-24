"""Local Muse Glimmer LiveKit worker."""

import os

# ONNX Runtime enables macOS telemetry at import time unless explicitly disabled.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from .config import GlimmerConfig

__all__ = ["GlimmerConfig"]
