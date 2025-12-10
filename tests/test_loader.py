"""Tests for model loading and voice style management."""

from pathlib import Path

import pytest

from supertonic.loader import (
    get_cache_dir,
    has_all_onnx_modules,
    list_available_voice_style_names,
    load_voice_style_from_name,
)


def test_get_cache_dir():
    """Test cache directory creation."""
    cache_dir = get_cache_dir()
    assert isinstance(cache_dir, Path)
    assert cache_dir.exists()
    assert cache_dir.is_dir()


def test_has_all_onnx_modules():
    """Test ONNX module checking."""
    cache_dir = get_cache_dir()
    # Should return True if models are downloaded, False otherwise
    result = has_all_onnx_modules(cache_dir)
    assert isinstance(result, bool)


def test_list_voice_styles():
    """Test listing voice styles."""
    cache_dir = get_cache_dir()
    try:
        styles = list_available_voice_style_names(cache_dir)
        assert isinstance(styles, list)
        # If models are downloaded, should have styles
        if styles:
            assert all(isinstance(s, str) for s in styles)
    except FileNotFoundError:
        # OK if models not downloaded yet
        pass


def test_load_voice_style_nonexistent():
    """Test loading non-existent voice style raises error."""
    cache_dir = get_cache_dir()

    with pytest.raises(FileNotFoundError):
        load_voice_style_from_name(cache_dir, "NonExistentVoice12345")
