"""Supertonic — Lightning Fast, On-Device TTS.

Supertonic is a high-performance, on-device text-to-speech system powered by
ONNX Runtime. It delivers state-of-the-art speech synthesis with unprecedented
speed and efficiency.

Example:
    ```python
    from supertonic import TTS

    tts = TTS()
    style = tts.get_voice_style("M1")
    wav, duration = tts.synthesize("Welcome to Supertonic text to speech synthesis.", voice_style=style)
    tts.save_audio(wav, "output.wav")
    ```
"""

from __future__ import annotations

import logging

from .core import Style, UnicodeProcessor
from .pipeline import TTS

__version__ = "1.0.0"

__all__ = [
    "TTS",
    "Style",
    "UnicodeProcessor",
    "__version__",
]

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
