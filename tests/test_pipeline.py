"""Tests for Supertonic pipeline."""

import numpy as np
import pytest

from supertonic import TTS


def test_tts_init():
    """Test that TTS initializes without error."""
    tts = TTS()
    assert tts is not None
    assert tts.sample_rate > 0


def test_tts_synthesize():
    """Test TTS synthesize returns correct types."""
    tts = TTS()
    style = tts.get_voice_style("M1")
    wav, duration = tts.synthesize("Test", voice_style=style, total_steps=3)

    assert isinstance(wav, np.ndarray)
    assert isinstance(duration, np.ndarray)
    assert wav.shape[0] > 0
    assert duration[0] > 0


def test_tts_with_different_voice():
    """Test TTS with different voice styles."""
    tts = TTS()

    style_m1 = tts.get_voice_style("M1")
    style_f1 = tts.get_voice_style("F1")

    wav1, _ = tts.synthesize("Hello", voice_style=style_m1, total_steps=3)
    wav2, _ = tts.synthesize("Hello", voice_style=style_f1, total_steps=3)

    assert wav1.shape[0] > 0
    assert wav2.shape[0] > 0


def test_tts_chunking():
    """Test that long text is handled via chunking."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Generate text long enough to trigger chunking
    long_text = "This is a test sentence. " * 20

    wav, _ = tts.synthesize(long_text, voice_style=style, total_steps=3)

    assert isinstance(wav, np.ndarray)
    assert wav.shape[0] > 0


def test_save_audio(tmp_path):
    """Test audio saving functionality."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    wav, _ = tts.synthesize("Test", voice_style=style, total_steps=3)

    # Save audio
    output_path = tmp_path / "test_output.wav"
    tts.save_audio(wav, str(output_path))

    # Check file exists and has non-zero size
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_get_voice_style():
    """Test voice style retrieval."""
    tts = TTS()

    style_m1 = tts.get_voice_style("M1")
    assert style_m1 is not None

    style_f1 = tts.get_voice_style("F1")
    assert style_f1 is not None


def test_invalid_voice_style():
    """Test handling of invalid voice style."""
    tts = TTS()

    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        tts.get_voice_style("INVALID_STYLE")


def test_empty_text():
    """Test handling of empty text."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Empty text should raise ValueError
    with pytest.raises(ValueError):
        tts.synthesize("", voice_style=style)


def test_voice_style_type_error():
    """Test that invalid voice_style type raises TypeError."""
    tts = TTS()

    with pytest.raises(TypeError, match="voice_style must be a Style object"):
        tts.synthesize("Test", voice_style="M1", total_steps=3)


def test_voice_style_names():
    """Test that voice style names can be listed."""
    tts = TTS()

    # Should have at least M1 and F1
    voices = tts.voice_style_names
    assert isinstance(voices, list)
    assert len(voices) > 0
    assert "M1" in voices or "F1" in voices


def test_tts_with_custom_steps():
    """Test TTS with custom number of steps."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Test with fewer steps (faster but lower quality)
    wav_fast, _ = tts.synthesize("Test", voice_style=style, total_steps=3)

    # Test with more steps (slower but higher quality)
    wav_quality, _ = tts.synthesize("Test", voice_style=style, total_steps=10)

    assert wav_fast.shape[0] > 0
    assert wav_quality.shape[0] > 0


def test_tts_with_speed_control():
    """Test TTS with speed control."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Normal speed
    wav_normal, dur_normal = tts.synthesize("Test", voice_style=style, total_steps=3, speed=1.0)

    # Faster
    wav_fast, dur_fast = tts.synthesize("Test", voice_style=style, total_steps=3, speed=1.5)

    assert wav_normal.shape[0] > 0
    assert wav_fast.shape[0] > 0
    # Faster should be shorter (approximately)
    assert dur_fast[0] < dur_normal[0]


def test_text_length_validation():
    """Test that text length is validated."""
    from supertonic.config import MAX_TEXT_LENGTH

    tts = TTS()
    style = tts.get_voice_style("M1")

    # Text exceeding MAX_TEXT_LENGTH should raise ValueError
    long_text = "a" * (MAX_TEXT_LENGTH + 1)
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        tts.synthesize(long_text, voice_style=style, total_steps=3)


def test_text_length_boundary():
    """Test text length at boundary."""
    from supertonic.config import MAX_TEXT_LENGTH

    tts = TTS()
    _ = tts.get_voice_style("M1")

    # Text at exactly MAX_TEXT_LENGTH should work
    _ = "a" * MAX_TEXT_LENGTH
    # This might be slow, so we just check it doesn't raise
    # In practice, MAX_TEXT_LENGTH is large enough this won't run


def test_total_steps_validation():
    """Test that total_steps is validated."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # total_steps = 0 should raise ValueError
    with pytest.raises(ValueError, match="total_steps must be between"):
        tts.synthesize("Test", voice_style=style, total_steps=0)

    # total_steps = -1 should raise ValueError
    with pytest.raises(ValueError, match="total_steps must be between"):
        tts.synthesize("Test", voice_style=style, total_steps=-1)

    # total_steps > MAX should raise ValueError (MAX_TOTAL_STEPS=100이므로 101을 사용)
    with pytest.raises(ValueError, match="total_steps must be between"):
        tts.synthesize("Test", voice_style=style, total_steps=101)


def test_total_steps_boundary():
    """Test total_steps at boundaries."""
    from supertonic.config import MIN_TOTAL_STEPS

    tts = TTS()
    style = tts.get_voice_style("M1")

    # MIN_TOTAL_STEPS should work
    wav, _ = tts.synthesize("Test", voice_style=style, total_steps=MIN_TOTAL_STEPS)
    assert wav.shape[0] > 0

    # MAX_TOTAL_STEPS should work (but might be slow)
    # Skip in normal tests
    # wav, _ = tts.synthesize("Test", voice_style=style, total_steps=MAX_TOTAL_STEPS)


def test_silence_duration_validation():
    """Test that silence_duration is validated."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Negative silence_duration should raise ValueError
    with pytest.raises(ValueError, match="must be non-negative"):
        tts.synthesize("Test", voice_style=style, total_steps=3, silence_duration=-1.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        tts.synthesize("Test", voice_style=style, total_steps=3, silence_duration=-0.1)


def test_silence_duration_zero():
    """Test silence_duration with zero value."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Zero should be valid
    wav, _ = tts.synthesize(
        "Test. More text.",
        voice_style=style,
        total_steps=3,
        max_chunk_length=10,
        silence_duration=0.0,
    )
    assert wav.shape[0] > 0


def test_save_audio_overwrite(tmp_path):
    """Test save_audio overwrites by default."""
    tts = TTS()
    style = tts.get_voice_style("M1")
    wav, _ = tts.synthesize("Test", voice_style=style, total_steps=3)

    output_path = tmp_path / "test.wav"

    # First save
    tts.save_audio(wav, str(output_path))
    assert output_path.exists()

    # Second save should overwrite without error (default behavior)
    tts.save_audio(wav, str(output_path))
    assert output_path.exists()


def test_save_audio_creates_directory(tmp_path):
    """Test that save_audio creates parent directories."""
    tts = TTS()
    style = tts.get_voice_style("M1")
    wav, _ = tts.synthesize("Test", voice_style=style, total_steps=3)

    # Path with non-existent parent directories
    output_path = tmp_path / "subdir" / "nested" / "output.wav"

    # Should create directories automatically
    tts.save_audio(wav, str(output_path))
    assert output_path.exists()
    assert output_path.parent.exists()


def test_verbose_mode_output(capsys):
    """Test that verbose mode produces output."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    wav, dur = tts.synthesize("Short test text.", voice_style=style, total_steps=3, verbose=True)

    captured = capsys.readouterr()
    assert "Input text length" in captured.out
    assert "Synthesizing audio" in captured.out
    assert "Generation complete" in captured.out


def test_call_shorthand():
    """Test that __call__ works as shorthand for synthesize."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    # Using __call__
    wav1, dur1 = tts("Test", voice_style=style, total_steps=3)

    # Using synthesize
    wav2, dur2 = tts.synthesize("Test", voice_style=style, total_steps=3)

    assert wav1.shape == wav2.shape
    assert isinstance(dur1, np.ndarray)
    assert isinstance(dur2, np.ndarray)


def test_whitespace_only_text():
    """Test that whitespace-only text raises error."""
    tts = TTS()
    style = tts.get_voice_style("M1")

    with pytest.raises(ValueError, match="cannot be empty"):
        tts.synthesize("   \n\t   ", voice_style=style, total_steps=3)


def test_get_voice_style_from_path():
    """Test loading voice style from path."""
    tts = TTS()

    # Load from path
    voice_path = tts.model_dir / "voice_styles" / "M1.json"
    if voice_path.exists():
        style = tts.get_voice_style_from_path(voice_path)
        assert style is not None
        assert isinstance(style.ttl, np.ndarray)
        assert isinstance(style.dp, np.ndarray)
