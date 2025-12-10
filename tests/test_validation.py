"""Tests for text validation functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from supertonic.core import UnicodeProcessor


@pytest.fixture
def unicode_indexer_file():
    """Create a temporary unicode indexer file for testing."""
    # Create a simple indexer with ASCII characters (0-127)
    indexer = list(range(128))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(indexer, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def processor(unicode_indexer_file):
    """Create a UnicodeProcessor instance for testing."""
    return UnicodeProcessor(unicode_indexer_file)


def test_get_supported_characters(processor):
    """Test getting supported characters."""
    # NOTE: UnicodeProcessor는 get_supported_characters() 메서드가 없고
    # supported_character_set 프로퍼티를 사용합니다.
    supported = processor.supported_character_set

    assert isinstance(supported, set)
    assert len(supported) > 0
    assert "a" in supported
    assert "A" in supported
    assert "0" in supported


def test_get_supported_unicode_ranges(processor):
    """Test getting supported unicode ranges."""
    # NOTE: UnicodeProcessor는 get_supported_unicode_ranges() 메서드가 없습니다.
    # supported_character_set에서 범위를 유추할 수 있습니다.
    supported = processor.supported_character_set

    # 테스트 목적으로 간단한 범위 추출 로직
    if len(supported) > 0:
        char_codes = sorted([ord(c) for c in supported])
        min_code = char_codes[0]
        max_code = char_codes[-1]

        assert min_code >= 0
        assert max_code < 128  # 우리의 테스트 픽스처는 ASCII만 지원


def test_get_unsupported_characters_none(processor):
    """Test getting unsupported characters from supported text."""
    # NOTE: UnicodeProcessor는 get_unsupported_characters() 메서드가 없습니다.
    # validate_text()를 사용하여 unsupported characters를 확인할 수 있습니다.
    text = "Hello World 123"
    is_valid, unsupported = processor.validate_text(text)

    assert is_valid is True
    assert isinstance(unsupported, list)
    assert len(unsupported) == 0


def test_get_unsupported_characters_exists(processor):
    """Test getting unsupported characters from text with unsupported chars."""
    # NOTE: UnicodeProcessor는 get_unsupported_characters() 메서드가 없습니다.
    # validate_text()를 사용하여 unsupported characters를 확인할 수 있습니다.
    # Unicode characters beyond ASCII range (>127) are unsupported
    text = "Hello 世界"  # Contains Chinese characters
    is_valid, unsupported = processor.validate_text(text)

    assert is_valid is False
    assert isinstance(unsupported, list)
    assert len(unsupported) > 0
    assert "世" in unsupported
    assert "界" in unsupported


def test_validate_text_valid(processor):
    """Test validating supported text."""
    text = "Hello World!"
    is_valid, unsupported = processor.validate_text(text)

    assert is_valid is True
    assert len(unsupported) == 0


def test_validate_text_invalid(processor):
    """Test validating text with unsupported characters."""
    text = "Hello 世界"
    is_valid, unsupported = processor.validate_text(text)

    assert is_valid is False
    assert len(unsupported) > 0
    assert "世" in unsupported or "界" in unsupported


def test_validate_text_empty_after_preprocess(processor):
    """Test validating text that becomes empty after preprocessing."""
    # NOTE: validate_text()는 preprocess 파라미터를 받지 않고 항상 전처리를 수행합니다.
    # Text with only emojis gets preprocessed to just "." which is valid
    # since _clean_whitespace adds a period if text doesn't end with punctuation
    text = "🎉🎊"
    is_valid, unsupported = processor.validate_text(text)

    # After preprocessing, emojis are removed and "." is added, which is valid
    assert is_valid is True
    assert len(unsupported) == 0


def test_validate_text_without_preprocess(processor):
    """Test validating text."""
    # NOTE: validate_text()는 preprocess 파라미터를 받지 않고 항상 전처리를 수행합니다.
    text = "Hello World!"
    is_valid, unsupported = processor.validate_text(text)

    assert is_valid is True
    assert len(unsupported) == 0


def test_validate_text_whitespace_only(processor):
    """Test validating whitespace-only text."""
    # NOTE: validate_text()는 preprocess 파라미터를 받지 않고 항상 전처리를 수행합니다.
    text = "   \n\t  "
    is_valid, unsupported = processor.validate_text(text)

    # After preprocessing, whitespace becomes "." which is valid
    assert is_valid is True
    assert len(unsupported) == 0


def test_indexer_type_validation():
    """Test that indexer must be a list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        # Write a dict instead of list
        json.dump({"invalid": "format"}, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="must be a list"):
            UnicodeProcessor(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_indexer_empty_validation():
    """Test that indexer cannot be empty."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        # Write empty list
        json.dump([], f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="empty"):
            UnicodeProcessor(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_unicode_processor_file_not_found():
    """Test that FileNotFoundError is raised for missing indexer file."""
    with pytest.raises(FileNotFoundError):
        UnicodeProcessor("/nonexistent/path/indexer.json")


def test_unicode_processor_malformed_json():
    """Test that ValueError is raised for malformed JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="malformed"):
            UnicodeProcessor(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)
