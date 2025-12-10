"""Tests for configuration module."""

import os

from supertonic.config import _parse_env_int


def test_parse_env_int_valid():
    """Test parsing valid integer from environment variable."""
    os.environ["TEST_VAR"] = "42"
    result = _parse_env_int("TEST_VAR")
    assert result == 42
    del os.environ["TEST_VAR"]


def test_parse_env_int_invalid():
    """Test parsing invalid integer from environment variable."""
    os.environ["TEST_VAR"] = "not_a_number"
    result = _parse_env_int("TEST_VAR", default=10)
    assert result == 10  # Should return default
    del os.environ["TEST_VAR"]


def test_parse_env_int_not_set():
    """Test parsing when environment variable is not set."""
    result = _parse_env_int("NONEXISTENT_VAR", default=5)
    assert result == 5


def test_parse_env_int_none_default():
    """Test parsing with None as default."""
    result = _parse_env_int("NONEXISTENT_VAR", default=None)
    assert result is None


def test_parse_env_int_negative():
    """Test parsing negative integer."""
    os.environ["TEST_VAR"] = "-10"
    result = _parse_env_int("TEST_VAR")
    assert result == -10
    del os.environ["TEST_VAR"]


def test_parse_env_int_zero():
    """Test parsing zero."""
    os.environ["TEST_VAR"] = "0"
    result = _parse_env_int("TEST_VAR")
    assert result == 0
    del os.environ["TEST_VAR"]


def test_parse_env_int_float_string():
    """Test parsing float string (should fail and return default)."""
    os.environ["TEST_VAR"] = "3.14"
    result = _parse_env_int("TEST_VAR", default=1)
    assert result == 1  # Should return default as "3.14" is not valid int
    del os.environ["TEST_VAR"]
