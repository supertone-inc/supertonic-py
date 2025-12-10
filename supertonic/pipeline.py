"""High-level TTS interface for Supertonic.

This module provides the main TTS class for easy text-to-speech synthesis
with automatic model loading and voice style management.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import (
    DEFAULT_MAX_CHUNK_LENGTH,
    DEFAULT_SILENCE_DURATION,
    DEFAULT_SPEED,
    DEFAULT_TOTAL_STEPS,
    MAX_TEXT_LENGTH,
    MAX_TOTAL_STEPS,
    MIN_TOTAL_STEPS,
)
from .core import Style
from .loader import (
    get_cache_dir,
    list_available_voice_style_names,
    load_model,
    load_voice_style_from_json_file,
    load_voice_style_from_name,
)
from .utils import chunk_text

logger = logging.getLogger(__name__)


class TTS:
    """High-level interface for Supertonic text-to-speech synthesis.

    Args:
        model_dir: Directory containing model files. If None, uses default cache
            directory (~/.cache/supertonic)
        auto_download: If True, automatically downloads model files from
            HuggingFace Hub if they're missing
        intra_op_num_threads: Number of threads for intra-op parallelism.
            If None (default), ONNX Runtime automatically determines optimal value based on your system.
            Can also be set via SUPERTONIC_INTRA_OP_THREADS environment variable
        inter_op_num_threads: Number of threads for inter-op parallelism.
            If None (default), ONNX Runtime automatically determines optimal value based on your system.
            Can also be set via SUPERTONIC_INTER_OP_THREADS environment variable

    Attributes:
        model (supertonic.core.Supertonic): The underlying Supertonic engine
        model_dir (pathlib.Path): Path to the model directory
        sample_rate (int): Audio sample rate in Hz
        voice_style_names (list[str]): List of available voice style names

    Example:
        ```python
        from supertonic import TTS
        tts = TTS()
        style = tts.get_voice_style("M1")
        wav, dur = tts.synthesize("The train delay was announced at 4:45 PM on Wed, Apr 3, 2024 due to track maintenance.", voice_style=style)
        tts.save_audio(wav, "output.wav")
        ```
    """

    def __init__(
        self,
        model_dir: Optional[Union[Path, str]] = None,
        auto_download: bool = True,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ):
        """Initialize the TTS engine.

        Args:
            model_dir (Union[Path, str]): Directory containing model files. If None, uses default
                cache directory
            auto_download: If True, automatically downloads missing model files
            intra_op_num_threads: Number of threads for intra-op parallelism.
                If None (default), ONNX Runtime automatically determines optimal value based on your system.
                Can also be set via SUPERTONIC_INTRA_OP_THREADS environment variable
            inter_op_num_threads: Number of threads for inter-op parallelism.
                If None (default), ONNX Runtime automatically determines optimal value based on your system.
                Can also be set via SUPERTONIC_INTER_OP_THREADS environment variable
        """
        if model_dir is None:
            model_dir = get_cache_dir()

        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)

        self.model = load_model(
            model_dir, auto_download, intra_op_num_threads, inter_op_num_threads
        )
        self.model_dir = model_dir
        self.sample_rate = self.model.sample_rate
        self.voice_style_names = list_available_voice_style_names(model_dir)

    def get_voice_style(self, voice_name: str) -> Style:
        """Load a voice style by name. Avaliable voice style names can be listed with
            `list_available_voice_style_names()`.

        Args:
            voice_name: Name of the voice style (e.g., 'M1', 'F1', 'M2', 'F2')

        Returns:
            Style object containing voice style vectors
        """
        return load_voice_style_from_name(self.model_dir, voice_name)

    def get_voice_style_from_path(self, voice_style_path: Union[Path, str]) -> Style:
        """Load a voice style from a JSON file path.

        Args:
            voice_style_path: Path to the voice style JSON file (str or Path)

        Returns:
            Style object containing voice style vectors
        """
        return load_voice_style_from_json_file(voice_style_path)

    def synthesize(
        self,
        text: str,
        voice_style: Style,
        total_steps: int = DEFAULT_TOTAL_STEPS,
        speed: float = DEFAULT_SPEED,
        max_chunk_length: int = DEFAULT_MAX_CHUNK_LENGTH,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize speech from text.

        This method automatically chunks long text into smaller segments
        and concatenates them with silence in between.

        Args:
            text: Text to synthesize
            voice_style: Voice style object
            total_steps: Number of synthesis steps (default: 5)
            speed: Speech speed multiplier (default: 1.05)
            max_chunk_length: Max characters per chunk (default: 300)
            silence_duration: Silence between chunks in seconds (default: 0.3)
            verbose: If True, print detailed progress information (default: False)

        Returns:
            Tuple of (waveform, duration):
                - waveform: Audio array of shape (1, num_samples)
                - duration: Total duration in seconds

        Example:
            ```python
            tts = TTS()
            style = tts.get_voice_style("M1")
            wav, dur = tts.synthesize("The train delay was announced at 4:45 PM on Wed, Apr 3, 2024 due to track maintenance.", voice_style=style, total_steps=5)
            print(f"Generated {dur[0]:.2f}s of audio")
            ```
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if verbose:
            print(f"📝 Input text length: {len(text)} characters")

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text length ({len(text)}) exceeds maximum allowed length "
                f"({MAX_TEXT_LENGTH}). Please split your text into smaller chunks."
            )

        if not isinstance(voice_style, Style):
            raise TypeError(
                f"voice_style must be a Style object, got {type(voice_style).__name__}. "
                f"Use get_voice_style() to load a style."
            )

        if not (total_steps >= MIN_TOTAL_STEPS and total_steps <= MAX_TOTAL_STEPS):
            raise ValueError(
                f"total_steps must be between {MIN_TOTAL_STEPS} and {MAX_TOTAL_STEPS}, "
                f"got {total_steps}. Higher values = better quality but slower."
            )

        if silence_duration < 0:
            raise ValueError(f"silence_duration must be non-negative, got {silence_duration}")

        # Validate text characters if verbose
        is_valid, unsupported = self.model.text_processor.validate_text(text)
        if not is_valid:
            raise ValueError(f"Found {len(unsupported)} unsupported character(s): {unsupported}")

        # Chunk text for processing
        text_chunks = chunk_text(text, max_chunk_length)

        if verbose:
            print(f"Split into {len(text_chunks)} chunk(s)")
            if len(text_chunks) > 1:
                for i, chunk in enumerate(text_chunks[:3]):  # Show first 3 chunks
                    print(f"Chunk {i+1}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
                if len(text_chunks) > 3:
                    print(f"... and {len(text_chunks) - 3} more chunk(s)")
            print(
                f"Synthesizing audio... Settings: steps={total_steps}, speed={speed:.2f}x, sample_rate={self.sample_rate}Hz"
            )

        # Collect all waveforms and durations in lists to avoid repeated concatenation
        wav_list = []
        dur_list = []
        for i, text_chunk in enumerate(text_chunks):
            if verbose:
                print(f"   [{i+1}/{len(text_chunks)}] Processing chunk... ", end="", flush=True)

            logger.debug(f"Processing chunk {i+1}/{len(text_chunks)}")
            wav, dur_onnx = self.model([text_chunk], voice_style, total_steps, speed)

            if verbose:
                print(f"✓ ({dur_onnx[0]:.2f}s)")

            # Validate waveform shape
            if wav.shape[0] != 1:
                raise RuntimeError(f"Expected wav shape (1, samples), got {wav.shape}")

            wav_list.append(wav)
            dur_list.append(dur_onnx)

        # Type guard: lists should never be empty after processing
        assert len(wav_list) > 0 and len(dur_list) > 0, "No audio generated"

        # Build list of arrays to concatenate: [wav1, silence, wav2, silence, wav3, ...]
        silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
        arrays_to_concat = []
        for i, wav in enumerate(wav_list):
            arrays_to_concat.append(wav)
            if i < len(wav_list) - 1:  # Don't add silence after last chunk
                arrays_to_concat.append(silence)

        # Single concatenation operation
        wav_cat = np.concatenate(arrays_to_concat, axis=1)

        # Calculate total duration
        total_audio_dur = sum(dur_list)
        total_silence_dur = silence_duration * (len(wav_list) - 1)
        dur_cat = total_audio_dur + total_silence_dur

        if verbose:
            total_samples = wav_cat.shape[1]
            print("Generation complete!")
            print(f"Total duration: {dur_cat[0]:.2f}s")
            print(f"Total samples: {total_samples:,}")
            print(f"Array shape: {wav_cat.shape}")

        return wav_cat, dur_cat

    def save_audio(
        self,
        wav: np.ndarray,
        output_path: str,
    ) -> None:
        """Save synthesized audio to a WAV file.

        Args:
            wav: Audio waveform array from synthesize()
            output_path: Path where to save the WAV file
        """
        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError as e:
            logger.error("soundfile not installed")
            raise ImportError(
                "soundfile library is required to save audio. "
                "Install it with: pip install soundfile"
            ) from e

        output_path_obj = Path(output_path)

        # Create parent directories if they don't exist
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Check write permissions
        if not os.access(output_path_obj.parent, os.W_OK):
            raise PermissionError(f"No write permission for directory: {output_path_obj.parent}")

        logger.info(f"Saving audio to {output_path}")
        sf.write(str(output_path), wav.squeeze(), self.sample_rate)
        logger.info("Audio saved successfully")

    def __call__(
        self,
        text: str,
        voice_style: Style,
        total_steps: int = DEFAULT_TOTAL_STEPS,
        speed: float = DEFAULT_SPEED,
        max_chunk_length: int = DEFAULT_MAX_CHUNK_LENGTH,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shorthand for synthesize(). Allows using tts(...) instead of tts.synthesize(...).

        Args:
            text: Text to synthesize
            voice_style: Voice style object
            total_steps: Number of synthesis steps (default: 5)
            speed: Speech speed multiplier (default: 1.05)
            max_chunk_length: Max characters per chunk (default: 300)
            silence_duration: Silence between chunks in seconds (default: 0.3)
            verbose: If True, print detailed progress information (default: False)

        Returns:
            Tuple of (waveform, duration):
                - waveform: Audio array of shape (1, num_samples)
                - duration: Total duration in seconds

        Example:
            ```python
            tts = TTS()
            style = tts.get_voice_style("M1")
            wav, dur = tts("The train delay was announced at 4:45 PM on Wed, Apr 3, 2024 due to track maintenance.", voice_style=style, total_steps=5)
            print(f"Generated {dur[0]:.2f}s of audio")
            ```
        """
        return self.synthesize(
            text=text,
            voice_style=voice_style,
            total_steps=total_steps,
            speed=speed,
            max_chunk_length=max_chunk_length,
            silence_duration=silence_duration,
            verbose=verbose,
        )
