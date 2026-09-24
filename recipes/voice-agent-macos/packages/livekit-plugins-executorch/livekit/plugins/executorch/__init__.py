"""Local ExecuTorch providers for LiveKit Agents."""

from livekit.agents import Plugin

from .log import logger
from .stt import STT
from .supertonic_tts import SupertonicTTS
from .version import __version__

__all__ = ["STT", "SupertonicTTS", "__version__"]


class ExecuTorchPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ExecuTorchPlugin())
