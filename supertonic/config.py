"""Configuration and constants for Supertonic TTS package.

This module centralizes all configuration values, magic numbers, and default
settings used throughout the package.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_REPO = os.getenv("SUPERTONIC_MODEL_REPO", "Supertone/supertonic")
DEFAULT_CACHE_DIR = os.getenv("SUPERTONIC_CACHE_DIR", str(Path.home() / ".cache" / "supertonic"))
# FIXME: change to the latest tag
DEFAULT_MODEL_REVISION = os.getenv("SUPERTONIC_MODEL_REVISION", "v1.0.0")

# Model paths
ONNX_DIR = Path("onnx")
VOICE_STYLES_DIR = Path("voice_styles")

CFG_REL_PATH = ONNX_DIR / "tts.json"
UNICODE_INDEXER_REL_PATH = ONNX_DIR / "unicode_indexer.json"
DP_ONNX_REL_PATH = ONNX_DIR / "duration_predictor.onnx"
TEXT_ENC_ONNX_REL_PATH = ONNX_DIR / "text_encoder.onnx"
VECTOR_EST_ONNX_REL_PATH = ONNX_DIR / "vector_estimator.onnx"
VOCODER_ONNX_REL_PATH = ONNX_DIR / "vocoder.onnx"

# TTS parameters - defaults
DEFAULT_TOTAL_STEPS = 5
DEFAULT_SPEED = 1.05
DEFAULT_MAX_CHUNK_LENGTH = 300
DEFAULT_SILENCE_DURATION = 0.3  # seconds

# TTS parameters - constraints
MIN_SPEED = 0.7
MAX_SPEED = 2.0
MIN_TOTAL_STEPS = 1
MAX_TOTAL_STEPS = 100

# ONNX Runtime configuration
# TODO: Add parsing of SUPERTONIC_ONNX_PROVIDERS environment variable
DEFAULT_ONNX_PROVIDERS = ["CPUExecutionProvider"]  # GPU support can be added by extending this list


def _parse_env_int(env_var: str, default: Optional[int] = None) -> Optional[int]:
    """Parse integer from environment variable with validation.

    Args:
        env_var: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Parsed integer or default value
    """
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid value for {env_var}: '{value}'. Using default: {default}")
        return default


# Thread configuration - None means let ONNX Runtime decide automatically
DEFAULT_INTRA_OP_NUM_THREADS = _parse_env_int("SUPERTONIC_INTRA_OP_THREADS")
DEFAULT_INTER_OP_NUM_THREADS = _parse_env_int("SUPERTONIC_INTER_OP_THREADS")

# Text processing
MAX_TEXT_LENGTH = 100_000  # Maximum characters per single synthesis call

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("SUPERTONIC_LOG_LEVEL", "INFO")
