from __future__ import annotations

import sys
from pathlib import Path

_WORKER_SRC = Path(__file__).parents[1] / "src"
_PLUGIN_ROOT = Path(__file__).parents[3] / "packages" / "livekit-plugins-executorch"
for path in (str(_WORKER_SRC), str(_PLUGIN_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
