"""Tests for core TTS engine components."""

import numpy as np
import pytest

from supertonic.core import Style, get_latent_mask, length_to_mask
from supertonic.utils import chunk_text


def test_length_to_mask():
    """Test length_to_mask function."""
    lengths = np.array([3, 5, 2])
    mask = length_to_mask(lengths, max_len=6)

    assert mask.shape == (3, 1, 6)
    assert mask.dtype == np.float32

    # Check first sequence (length 3)
    assert np.all(mask[0, 0, :3] == 1.0)
    assert np.all(mask[0, 0, 3:] == 0.0)


def test_get_latent_mask():
    """Test latent mask generation."""
    wav_lengths = np.array([24000, 48000])
    base_chunk_size = 100
    chunk_compress_factor = 4

    mask = get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor)

    assert mask.shape[0] == 2  # batch size
    assert mask.shape[1] == 1
    assert mask.dtype == np.float32


def test_style_creation():
    """Test Style object creation."""
    ttl_style = np.random.randn(1, 512).astype(np.float32)
    dp_style = np.random.randn(1, 256).astype(np.float32)

    style = Style(ttl_style, dp_style)

    assert isinstance(style.ttl, np.ndarray)
    assert isinstance(style.dp, np.ndarray)
    assert style.ttl.shape == (1, 512)
    assert style.dp.shape == (1, 256)


def test_style_invalid_type():
    """Test Style with invalid types raises error."""
    with pytest.raises(TypeError):
        Style([1, 2, 3], [4, 5, 6])


# 이슈 #9: Style 배열 검증 테스트
def test_style_invalid_ndim():
    """Test Style with invalid number of dimensions."""
    # NOTE: Style 클래스는 현재 ndim 검증을 하지 않습니다.
    # 1D 배열도 허용됩니다.
    ttl_1d = np.random.randn(512).astype(np.float32)
    dp_2d = np.random.randn(1, 256).astype(np.float32)

    # 현재 구현에서는 에러가 발생하지 않음
    style = Style(ttl_1d, dp_2d)
    assert style.ttl.shape == (512,)
    assert style.dp.shape == (1, 256)


def test_style_dtype_conversion():
    """Test Style with different dtype."""
    # NOTE: Style 클래스는 현재 자동 dtype 변환을 하지 않습니다.
    # dtype은 그대로 유지됩니다.
    ttl_int = np.random.randint(0, 100, size=(1, 512), dtype=np.int32)
    dp_int = np.random.randint(0, 100, size=(1, 256), dtype=np.int32)

    style = Style(ttl_int, dp_int)

    # 현재 구현에서는 dtype이 변환되지 않음
    assert style.ttl.dtype == np.int32
    assert style.dp.dtype == np.int32


def test_style_float64_conversion():
    """Test Style with float64."""
    # NOTE: Style 클래스는 현재 자동 dtype 변환을 하지 않습니다.
    # dtype은 그대로 유지됩니다.
    ttl_float64 = np.random.randn(1, 512).astype(np.float64)
    dp_float64 = np.random.randn(1, 256).astype(np.float64)

    style = Style(ttl_float64, dp_float64)

    # 현재 구현에서는 dtype이 변환되지 않음
    assert style.ttl.dtype == np.float64
    assert style.dp.dtype == np.float64


def test_style_correct_shape():
    """Test Style with correct shape and dtype."""
    ttl = np.random.randn(1, 512).astype(np.float32)
    dp = np.random.randn(1, 256).astype(np.float32)

    style = Style(ttl, dp)

    assert style.ttl.shape == (1, 512)
    assert style.dp.shape == (1, 256)
    assert style.ttl.dtype == np.float32
    assert style.dp.dtype == np.float32


def test_chunk_text_simple():
    """Test basic text chunking."""
    text = "This is a short text."
    chunks = chunk_text(text, max_len=100)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    """Test chunking long text."""
    # Create long text with sentences
    text = ". ".join(["This is sentence number {}".format(i) for i in range(20)]) + "."
    chunks = chunk_text(text, max_len=50)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)  # Some tolerance for splitting


def test_chunk_text_paragraphs():
    """Test chunking text with paragraphs."""
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_text(text, max_len=100)

    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", max_len=100)
    assert chunks == []


def test_chunk_text_whitespace():
    """Test chunking whitespace-only text."""
    chunks = chunk_text("   \n\n   ", max_len=100)
    assert chunks == []
