"""Model loading and voice style management utilities.

This module handles downloading, loading, and managing Supertonic TTS models
and voice styles from HuggingFace Hub.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from .config import (
    CFG_REL_PATH,
    DEFAULT_CACHE_DIR,
    DEFAULT_INTER_OP_NUM_THREADS,
    DEFAULT_INTRA_OP_NUM_THREADS,
    DEFAULT_MODEL_REPO,
    DEFAULT_MODEL_REVISION,
    DEFAULT_ONNX_PROVIDERS,
    DP_ONNX_REL_PATH,
    TEXT_ENC_ONNX_REL_PATH,
    UNICODE_INDEXER_REL_PATH,
    VECTOR_EST_ONNX_REL_PATH,
    VOCODER_ONNX_REL_PATH,
    VOICE_STYLES_DIR,
)
from .core import Style, Supertonic, UnicodeProcessor
from .utils import validate_voice_style_format

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get or create the default cache directory for Supertonic models.

    Returns:
        Path object pointing to the cache directory

    Note:
        Default location is ~/.cache/supertonic, but can be overridden
        with SUPERTONIC_CACHE_DIR environment variable
    """
    cache_dir = Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache directory: {cache_dir}")
    return cache_dir


def get_all_onnx_module_relative_paths() -> list[Path]:
    """Get list of all required ONNX model file paths.

    Returns:
        List of Path objects for each required ONNX model file
    """
    return [
        DP_ONNX_REL_PATH,
        TEXT_ENC_ONNX_REL_PATH,
        VECTOR_EST_ONNX_REL_PATH,
        VOCODER_ONNX_REL_PATH,
    ]


def has_all_onnx_modules(model_dir: Union[Path, str]) -> bool:
    """Check if all required ONNX model files exist in the directory.

    Args:
        model_dir: Directory to check for model files (str or Path)

    Returns:
        True if all required ONNX files exist, False otherwise
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    module_rel_paths = get_all_onnx_module_relative_paths()
    all_exist = all((model_dir / path).exists() for path in module_rel_paths)

    if not all_exist:
        missing = [str(path) for path in module_rel_paths if not (model_dir / path).exists()]
        logger.debug(f"Missing ONNX files: {missing}")

    return all_exist


def download_model(model_dir: Union[Path, str]) -> None:
    """Download Supertonic model from HuggingFace Hub.

    Args:
        model_dir: Directory where the model should be downloaded (str or Path)
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir

    # Use temporary directory for atomic download
    temp_dir = model_dir.parent / f".{model_dir.name}.tmp"

    try:
        from huggingface_hub import snapshot_download

        logger.info(
            f"Downloading model from {DEFAULT_MODEL_REPO} to temporary location: {temp_dir}"
        )
        snapshot_download(
            repo_id=DEFAULT_MODEL_REPO, local_dir=str(temp_dir), revision=DEFAULT_MODEL_REVISION
        )

        # Move from temporary to final location on success
        if model_dir.exists():
            logger.info(f"Removing existing model directory: {model_dir}")
            shutil.rmtree(model_dir)
        shutil.move(str(temp_dir), str(model_dir))

        logger.info("Model download completed successfully")

    except ImportError as e:
        logger.error("huggingface_hub not installed")
        raise RuntimeError(
            "Failed to import huggingface_hub. Please install it with: "
            "pip install huggingface-hub"
        ) from e
    except Exception as e:
        logger.error(f"Model download failed: {e}")

        # Clean up temporary files on failure
        if temp_dir.exists():
            logger.info("Cleaning up temporary files...")
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary files: {cleanup_error}")

        raise RuntimeError(
            f"Failed to download model from {DEFAULT_MODEL_REPO}. "
            f"Please check your internet connection and try again. "
            f"Error: {e}"
        ) from e


def load_configs(model_dir: Union[Path, str]) -> dict:
    """Load model configuration from JSON file.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        Dictionary containing model configuration
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    cfg_path = model_dir / CFG_REL_PATH

    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(
            f"Model configuration file not found at {cfg_path}. "
            f"Please ensure the model is properly downloaded."
        )

    try:
        with open(cfg_path, "r") as f:
            cfgs = json.load(f)
        logger.debug(f"Loaded config from {cfg_path}")
        return cfgs
    except json.JSONDecodeError as e:
        logger.error(f"Invalid config format: {e}")
        raise ValueError(
            f"Model configuration file is malformed at {cfg_path}. "
            f"Please re-download the model."
        ) from e


def load_onnx_modules(
    model_dir: Union[Path, str],
    intra_op_num_threads: Optional[int] = None,
    inter_op_num_threads: Optional[int] = None,
) -> tuple[
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
]:
    """Load all ONNX model modules for TTS synthesis.

    Args:
        model_dir: Directory containing the ONNX model files (str or Path)
        intra_op_num_threads: Number of threads for intra-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value
        inter_op_num_threads: Number of threads for inter-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value

    Returns:
        Tuple of (duration_predictor, text_encoder, vector_estimator, vocoder)
        ONNX Runtime inference sessions
    """

    def _load_onnx(
        onnx_path: Path, opts: ort.SessionOptions, providers: list[str]
    ) -> ort.InferenceSession:
        """Load a single ONNX model file."""
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model file not found: {onnx_path}. "
                f"Please ensure the model is properly downloaded."
            )

        logger.debug(f"Loading ONNX model: {onnx_path}")
        return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)

    opts = ort.SessionOptions()
    # Performance optimizations
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Only set thread counts if explicitly specified
    # Otherwise, let ONNX Runtime automatically determine optimal values
    intra_threads = (
        intra_op_num_threads if intra_op_num_threads is not None else DEFAULT_INTRA_OP_NUM_THREADS
    )
    inter_threads = (
        inter_op_num_threads if inter_op_num_threads is not None else DEFAULT_INTER_OP_NUM_THREADS
    )

    if intra_threads is not None:
        opts.intra_op_num_threads = intra_threads
    if inter_threads is not None:
        opts.inter_op_num_threads = inter_threads

    # Setup execution providers with fallback to CPU
    providers = DEFAULT_ONNX_PROVIDERS
    available_providers = ort.get_available_providers()

    # Filter to only use available providers
    valid_providers = [p for p in providers if p in available_providers]
    if not valid_providers:
        logger.warning(
            f"Requested providers {providers} not available. "
            f"Falling back to CPUExecutionProvider"
        )
        valid_providers = ["CPUExecutionProvider"]
    else:
        logger.info(f"Using ONNX providers: {valid_providers}")

    thread_info = (
        f"intra_threads={intra_threads if intra_threads is not None else 'auto'}, "
        f"inter_threads={inter_threads if inter_threads is not None else 'auto'}"
    )
    logger.info(f"ONNX Runtime config: {thread_info}")

    dp_onnx_path = model_dir / DP_ONNX_REL_PATH
    text_enc_onnx_path = model_dir / TEXT_ENC_ONNX_REL_PATH
    vector_est_onnx_path = model_dir / VECTOR_EST_ONNX_REL_PATH
    vocoder_onnx_path = model_dir / VOCODER_ONNX_REL_PATH

    logger.info(f"Loading ONNX models with providers: {valid_providers}")
    dp_ort = _load_onnx(dp_onnx_path, opts, valid_providers)
    text_enc_ort = _load_onnx(text_enc_onnx_path, opts, valid_providers)
    vector_est_ort = _load_onnx(vector_est_onnx_path, opts, valid_providers)
    vocoder_ort = _load_onnx(vocoder_onnx_path, opts, valid_providers)

    logger.info("Successfully loaded all ONNX models")
    return dp_ort, text_enc_ort, vector_est_ort, vocoder_ort


def load_text_processor(model_dir: Union[Path, str]) -> UnicodeProcessor:
    """Load the unicode text processor for the model.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        Initialized UnicodeProcessor instance
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    unicode_indexer_path = model_dir / UNICODE_INDEXER_REL_PATH
    logger.debug(f"Loading text processor from {unicode_indexer_path}")
    text_processor = UnicodeProcessor(str(unicode_indexer_path))
    return text_processor


def load_model(
    model_dir: Union[Path, str],
    auto_download: bool,
    intra_op_num_threads: Optional[int] = None,
    inter_op_num_threads: Optional[int] = None,
) -> Supertonic:
    """Load the complete Supertonic TTS model.

    This function loads all model components including ONNX modules,
    configuration, and text processor. If model files are missing and
    auto_download is enabled, it will download them from HuggingFace Hub.

    Args:
        model_dir: Directory containing (or to contain) the model files (str or Path)
        auto_download: If True, automatically download missing model files
        intra_op_num_threads: Number of threads for intra-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value
        inter_op_num_threads: Number of threads for inter-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value

    Returns:
        Initialized Supertonic TTS engine
    """
    logger.info(f"Loading model from {model_dir}")
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir

    if not has_all_onnx_modules(model_dir):
        if not auto_download:
            logger.error(f"ONNX models not found in {model_dir}")
            raise FileNotFoundError(
                f"ONNX model files not found in {model_dir}. "
                f"Set auto_download=True to automatically download from HuggingFace Hub, "
                f"or manually download the model to this directory."
            )
        download_model(model_dir)

    cfgs = load_configs(model_dir)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_modules(
        model_dir, intra_op_num_threads, inter_op_num_threads
    )
    text_processor = load_text_processor(model_dir)

    logger.info("Model loaded successfully")
    return Supertonic(cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort)


def list_available_voice_style_paths(model_dir: Union[Path, str]) -> list[Path]:
    """List all available voice style JSON files in the model directory.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        Sorted list of paths to voice style JSON files
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    voice_styles_dir = model_dir / VOICE_STYLES_DIR

    if not voice_styles_dir.exists():
        logger.error(f"Voice styles directory not found: {voice_styles_dir}")
        raise FileNotFoundError(
            f"Voice styles directory not found at {voice_styles_dir}. "
            f"Please ensure the model is properly downloaded."
        )

    paths = sorted(list(voice_styles_dir.glob("*.json")))
    logger.debug(f"Found {len(paths)} voice styles in {voice_styles_dir}")
    return paths


def list_available_voice_style_names(model_dir: Union[Path, str]) -> list[str]:
    """List names of all available voice styles.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        List of voice style names (without .json extension)
    """
    voice_style_paths = list_available_voice_style_paths(model_dir)
    names = [path.stem for path in voice_style_paths]
    return names


def load_voice_style_from_json_file(voice_style_path: Union[Path, str]) -> Style:
    """Load a voice style from a JSON file.

    Args:
        voice_style_path: Path to the voice style JSON file (str or Path)

    Returns:
        Style object with loaded style vectors
    """
    # Convert to Path if string
    voice_style_path = Path(voice_style_path)

    def _load_style_from_json(json_data: dict) -> np.ndarray:
        """Parse style vector from JSON data."""
        try:
            dims = json_data["dims"]
            data = json_data["data"]
            return np.array(data, dtype=np.float32).reshape(*dims)
        except KeyError as e:
            raise ValueError(f"Invalid style format: missing key {e}") from e

    if not voice_style_path.exists():
        raise FileNotFoundError(f"Voice style file not found: {voice_style_path}")

    try:
        with open(voice_style_path, "r") as f:
            voice_style_json = json.load(f)

        # Validate voice style format
        if not validate_voice_style_format(voice_style_json):
            raise ValueError(
                f"Invalid voice style format in {voice_style_path}. "
                f"Expected 'style_ttl' and 'style_dp' with 'dims' and 'data' fields."
            )

        logger.debug(f"Loading voice style from {voice_style_path}")
        ttl_style = _load_style_from_json(voice_style_json["style_ttl"])
        dp_style = _load_style_from_json(voice_style_json["style_dp"])

        return Style(ttl_style, dp_style)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in voice style file: {e}")
        raise ValueError(
            f"Voice style file is malformed at {voice_style_path}. "
            f"Please check the file format."
        ) from e
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid voice style format: {e}")
        raise ValueError(
            f"Voice style file has invalid format at {voice_style_path}. "
            f"Expected 'style_ttl' and 'style_dp' fields. Error: {e}"
        ) from e


def load_voice_style_from_name(model_dir: Union[Path, str], voice_name: str) -> Style:
    """Load a voice style by name from the model directory.

    Args:
        model_dir: Directory containing the model files (str or Path)
        voice_name: Name of the voice style (without .json extension)

    Returns:
        Style object with loaded style vectors
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    voice_style_path = model_dir / VOICE_STYLES_DIR / f"{voice_name}.json"

    if not voice_style_path.exists():
        available = list_available_voice_style_names(model_dir)
        logger.error(f"Voice style '{voice_name}' not found")
        raise FileNotFoundError(
            f"Voice style '{voice_name}' not found. "
            f"Available voice styles: {', '.join(available)}"
        )

    return load_voice_style_from_json_file(voice_style_path)
