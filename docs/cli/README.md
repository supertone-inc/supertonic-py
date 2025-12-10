# Supertonic CLI Guide

```bash
# Quick playback (no file saved)
supertonic say TEXT [OPTIONS]

# Text-to-speech (saves to file)
supertonic tts TEXT -o OUTPUT.wav [OPTIONS]

# Options:
#   --voice STYLE             Voice style: M1, M2, F1, F2 (default: M1)
#   --steps N                 Quality steps: 2-15 typical (default: 5)
#   --speed RATE              Speed multiplier: 0.7-2.0 (default: 1.05)
#   --max-chunk-length N      Characters per chunk (default: 300)
#   --silence-duration SECS   Silence between chunks (default: 0.3)
#   --verbose, -v             Show detailed progress and text processing
#   --custom-style-path PATH  Path to custom voice style JSON file (overrides --voice if provided)

# Utilities
supertonic list-voices       # List available voices
supertonic info             # Show model information
supertonic version          # Show version
```

The `supertonic` command-line tool provides easy access to text-to-speech synthesis. You can start by viewing the help message with:

```bash
supertonic --help
```

Available Commands:

```bash
supertonic {say,tts,list-voices,info,download,version}
```

## say

Generate speech from text and play it directly without saving a file.

!!! note "Requires `sounddevice`"
    Install with: `pip install supertonic[playback]`

Basic usage:

```bash
supertonic say 'Hello, welcome to the world!'
```

With options:

```bash
# Specify voice style
supertonic say 'Hello, welcome to the world!' --voice F1

# Control quality (steps: 2-15 typical)
supertonic say 'Hello, welcome to the world!' --steps 10

# Adjust speed (0.7-2.0)
supertonic say 'Hello, welcome to the world!' --speed 1.5
```

See [supertonic say](./say.md) for the full reference of all available arguments.

## tts

Generate speech from text and save to a WAV file.

Basic usage:

```bash
supertonic tts 'Hello, welcome to the world!' -o output.wav
```

With options:

```bash
# Specify voice style
supertonic tts 'Hello, welcome to the world!' -o output.wav --voice F1

# Control quality (steps: 2-15 typical)
supertonic tts 'Hello, welcome to the world!' -o output.wav --steps 10

# Adjust speed (0.7-2.0)
supertonic tts 'Hello, welcome to the world!' -o output.wav --speed 1.5
```

See [supertonic tts](./tts.md) for the full reference of all available arguments.

## list-voices

List all available voice styles.

```bash
supertonic list-voices
```

**Aliases:** `lv`

## info

Show model information including cache location and available voices.

```bash
supertonic info
```

**Aliases:** `i`

## download

Download model from HuggingFace Hub.

```bash
supertonic download
```

**Aliases:** `d`

This is useful for pre-downloading the model before first use or in Docker/CI environments.

## version

Show installed version.

```bash
supertonic version
```

**Aliases:** `v`

---

## Environment Variables

**`SUPERTONIC_CACHE_DIR`**

Override the default cache directory for model files.

```bash
export SUPERTONIC_CACHE_DIR=/custom/cache/path
```

Default: `~/.cache/supertonic`

**`SUPERTONIC_INTRA_OP_THREADS`**

Configure ONNX Runtime intra-operator thread count.

```bash
export SUPERTONIC_INTRA_OP_THREADS=8
```

Default: Auto-detected

**`SUPERTONIC_INTER_OP_THREADS`**

Configure ONNX Runtime inter-operator thread count.

```bash
export SUPERTONIC_INTER_OP_THREADS=8
```

Default: Auto-detected

---

## More Help

For detailed options of any subcommand, use:

```bash
supertonic <subcommand> --help
```
