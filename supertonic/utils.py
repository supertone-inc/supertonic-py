"""Utility functions for Supertonic TTS.

This module provides various helper functions for text processing, file operations,
and timing operations used throughout the Supertonic TTS package.
"""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Common abbreviations that should not trigger sentence splitting
_COMMON_ABBREVIATIONS_PATTERN = (
    r"(?<!Mr\.)"
    r"(?<!Mrs\.)"
    r"(?<!Ms\.)"
    r"(?<!Dr\.)"
    r"(?<!Prof\.)"
    r"(?<!Sr\.)"
    r"(?<!Jr\.)"
    r"(?<!Ph\.D\.)"
    r"(?<!etc\.)"
    r"(?<!e\.g\.)"
    r"(?<!i\.e\.)"
    r"(?<!vs\.)"
    r"(?<!Inc\.)"
    r"(?<!Ltd\.)"
    r"(?<!Co\.)"
    r"(?<!Corp\.)"
    r"(?<!St\.)"
    r"(?<!Ave\.)"
    r"(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)"  # Single capital letter abbreviations
    r"(?<=[.!?])\s+"  # Split after sentence-ending punctuation followed by space
)


def sanitize_filename(text: str, max_len: int = 50) -> str:
    """
    Sanitize filename by replacing non-alphanumeric characters.

    Args:
        text: Input text to convert to filename
        max_len: Maximum length of filename

    Returns:
        Sanitized filename string
    """
    prefix = text[:max_len]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", prefix)


@contextmanager
def timer(name: str, verbose: bool = True):
    """
    Context manager for timing code execution.

    Args:
        name: Name of the operation being timed
        verbose: Whether to log timing information

    Example:
        ```python
        with timer("Processing"):
            # Your code here
            process_data()
        ```
    """
    if verbose:
        logger.info(f"{name}...")
    start = time.time()
    yield
    elapsed = time.time() - start
    if verbose:
        logger.info(f"{name} completed in {elapsed:.2f}s")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1.23s", "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_audio_duration(wav_length: int, sample_rate: int) -> float:
    """
    Calculate audio duration from waveform length.

    Args:
        wav_length: Number of samples in waveform
        sample_rate: Audio sample rate (Hz)

    Returns:
        Duration in seconds
    """
    return wav_length / sample_rate


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Absolute path to directory
    """
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def validate_voice_style_format(style_data: dict) -> bool:
    """
    Validate voice style JSON format.

    Args:
        style_data: Voice style dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["style_ttl", "style_dp"]
    if not all(key in style_data for key in required_keys):
        return False

    for key in required_keys:
        if "dims" not in style_data[key] or "data" not in style_data[key]:
            return False

    return True


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """
    Split text into chunks by paragraphs and sentences.

    This function intelligently splits long text into smaller chunks suitable
    for TTS processing, respecting paragraph and sentence boundaries.

    Args:
        text: Input text to chunk
        max_len: Maximum length of each chunk in characters (default: 300)

    Returns:
        List of text chunks

    Example:
        ```python
        text = "This is a long paragraph. It has multiple sentences. " * 10
        chunks = chunk_text(text, max_len=100)
        for chunk in chunks:
            print(f"Chunk ({len(chunk)} chars): {chunk[:50]}...")
        ```
    """
    # Validate minimum chunk length
    if max_len < 10:
        raise ValueError(
            f"max_len must be at least 10, got {max_len}. "
            f"Very small chunks may produce poor quality speech."
        )

    # Split by paragraph (two or more newlines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Split by sentence boundaries (period, question mark, exclamation mark followed by space)
        # But exclude common abbreviations like Mr., Mrs., Dr., etc. and single capital letters
        sentences = re.split(_COMMON_ABBREVIATIONS_PATTERN, paragraph)

        current_chunk = ""

        for sentence in sentences:
            # Skip empty sentences to prevent empty chunks
            # Strip once and reuse the result
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue

            if len(current_chunk) + len(sentence_stripped) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence_stripped
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence_stripped

        # Add final chunk if not empty
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

    return chunks
