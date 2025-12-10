"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def sample_text():
    """Provide sample text for testing."""
    return "This is a test."


@pytest.fixture(scope="session")
def sample_texts():
    """Provide sample texts for batch testing."""
    return ["Hello", "World", "Test"]


@pytest.fixture
def voice_style():
    """Provide default voice style."""
    return "M1"


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
