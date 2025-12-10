"""Tests for CLI functionality."""

import sys

import pytest

from supertonic.cli import cmd_info, cmd_list_voices, cmd_version, create_parser, main


def test_cli_help(capsys):
    """Test that CLI help works without error."""
    with pytest.raises(SystemExit) as exc_info:
        sys.argv = ["supertonic", "--help"]
        main()

    assert exc_info.value.code == 0


def test_cli_no_command(capsys):
    """Test CLI with no command shows help."""
    with pytest.raises(SystemExit) as exc_info:
        sys.argv = ["supertonic"]
        main()

    assert exc_info.value.code == 1


def test_cmd_list_voices(capsys):
    """Test list-voices command."""
    from argparse import Namespace

    args = Namespace()
    cmd_list_voices(args)

    captured = capsys.readouterr()
    assert "Available voice styles" in captured.out
    assert "M1" in captured.out or "F1" in captured.out


def test_cmd_info(capsys):
    """Test info command."""
    from argparse import Namespace

    args = Namespace()
    cmd_info(args)

    captured = capsys.readouterr()
    assert "Model Information" in captured.out
    assert "Sample rate" in captured.out


def test_cmd_version(capsys):
    """Test version command."""
    from argparse import Namespace

    args = Namespace()
    cmd_version(args)

    captured = capsys.readouterr()
    assert "supertonic" in captured.out
    assert "0.1.1" in captured.out


def test_create_parser():
    """Test parser creation."""
    parser = create_parser()
    assert parser is not None

    # Test TTS command parsing
    args = parser.parse_args(["tts", "Hello", "-o", "test.wav"])
    assert args.command == "tts"
    assert args.text == "Hello"
    assert args.output == "test.wav"
    assert args.voice == "M1"  # default
    assert args.steps == 5  # default
    assert args.speed == 1.05  # default


def test_parser_with_options():
    """Test parser with various options."""
    parser = create_parser()

    args = parser.parse_args(
        [
            "tts",
            "Test text",
            "-o",
            "output.wav",
            "--voice",
            "F1",
            "--steps",
            "10",
            "--speed",
            "1.5",
            "--max-chunk-length",
            "200",
            "--silence-duration",
            "0.5",
            "--verbose",
        ]
    )

    assert args.voice == "F1"
    assert args.steps == 10
    assert args.speed == 1.5
    assert args.max_chunk_length == 200
    assert args.silence_duration == 0.5
    assert args.verbose is True
    assert args.custom_style_path is None  # default


def test_parser_custom_style_path():
    """Test --custom-style-path option."""
    parser = create_parser()

    # Test with custom style path
    args = parser.parse_args(
        ["tts", "Test", "-o", "out.wav", "--custom-style-path", "/path/to/custom_voice.json"]
    )
    assert args.custom_style_path == "/path/to/custom_voice.json"

    # Default should be None
    args = parser.parse_args(["tts", "Test", "-o", "out.wav"])
    assert args.custom_style_path is None


def test_cli_aliases():
    """Test command aliases."""
    parser = create_parser()

    # tts alias 't'
    args = parser.parse_args(["t", "Test", "-o", "out.wav"])
    assert args.command == "t"

    # list-voices alias 'lv'
    args = parser.parse_args(["lv"])
    assert args.command == "lv"

    # version alias 'v'
    args = parser.parse_args(["v"])
    assert args.command == "v"
