"""Test that all public APIs are importable."""


def test_import_tts():
    """Test main TTS class import."""
    from supertonic import TTS

    assert TTS is not None


def test_import_version():
    """Test __version__ import."""
    from supertonic import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import supertonic

    assert hasattr(supertonic, "__all__")
    assert "TTS" in supertonic.__all__
    assert "__version__" in supertonic.__all__
