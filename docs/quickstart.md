# Quick Start

<figure markdown="block">
<video src="../assets/video/supertonic-pip-demo-low.mp4" autoplay loop muted playsinline controls style="width: 100%; max-width: 100%; border-radius: 8px;"></video>
</figure>

## Installation

```bash
pip install supertonic
```

## Basic Usage

=== "Python"

    ```python
    from supertonic import TTS

    # Note: First run downloads model automatically (~260MB)
    tts = TTS(auto_download=True)

    # Get a voice style
    style = tts.get_voice_style(voice_name="M1")

    # Generate speech
    text = "The train delay was announced at 4:45 PM on Wed, Apr 3, 2024 due to track maintenance."
    wav, duration = tts.synthesize(text, voice_style=style)
    # wav: np.ndarray, shape = (1, num_samples)
    # duration: np.ndarray, shape = (1,)

    # Save to file
    tts.save_audio(wav, "output.wav")
    ```

    ??? tip "Arguments for `tts.synthesize()`"
        | Parameter | Description | Default |
        |-----------|-------------|---------|
        | `text` | Text to synthesize | *required* |
        | `voice_style` | Voice style object | *required* |
        | `total_steps` | Quality: 2-15 typical (higher=better) | `5` |
        | `speed` | Speed: 0.7 (slow) to 2.0 (fast) | `1.05` |
        | `max_chunk_length` | Max characters per chunk | `300` |
        | `silence_duration` | Silence between chunks (seconds) | `0.3` |
        | `verbose` | Show detailed progress | `False` |

    ??? tip "Shorthand"
        Use `tts(...)` as shorthand for `tts.synthesize(...)`:
        ```python
        wav, duration = tts("This is a convenient shorthand method.", voice_style=style)
        ```

=== "CLI"

    ```bash
    # Note: First run downloads model automatically (~260MB)
    supertonic tts 'Supertonic is a lightning fast, on-device TTS system.' -o output.wav
    ```

    ??? tip "CLI Options"
        | Option | Description | Default |
        |--------|-------------|---------|
        | `-o`, `--output` | Output file path | *required* |
        | `--voice` | Voice style: M1, F1, M2, F2, ... | `M1` |
        | `--steps` | Quality steps: 2-15 typical | `5` |
        | `--speed` | Speed multiplier: 0.7-2.0 | `1.05` |
        | `--max-chunk-length` | Characters per chunk | `300` |
        | `--silence-duration` | Silence between chunks (seconds) | `0.3` |
        | `--custom-style-path` | Path to custom voice style JSON | - |
        | `-v`, `--verbose` | Show detailed progress | `False` |

    ??? tip "Utility Commands"
        ```bash
        supertonic list-voices   # List available voices
        supertonic info          # Show model information
        supertonic version       # Show version
        ```

---

## Try in Colab

Run and experiment with Supertonic in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/supertone-inc/supertonic-py/blob/main/notebook/supertonic_demo.ipynb)

---

## Advanced Usage

### Voice Styles

Supertonic provides multiple built-in voice styles to choose from:

=== "Python"

    ```python
    from supertonic import TTS

    tts = TTS()

    # List available voices
    voice_list = tts.voice_style_names  # ex) ['M1', 'M2', 'F1', 'F2']

    # Try different voices
    for voice_name in voice_list:
        style = tts.get_voice_style(voice_name)
        wav, dur = tts.synthesize(f"This is the {voice_name} voice style demonstration.", voice_style=style)
        tts.save_audio(wav, f"output_{voice_name}.wav")
    ```

=== "CLI"

    ```bash
    # List available voices
    supertonic list-voices

    # Try different voices
    supertonic tts 'This is the M1 voice style demonstration.' --voice M1 -o output_m1.wav
    supertonic tts 'This is the M2 voice style demonstration.' --voice M2 -o output_m2.wav
    supertonic tts 'This is the F1 voice style demonstration.' --voice F1 -o output_f1.wav
    supertonic tts 'This is the F2 voice style demonstration.' --voice F2 -o output_f2.wav
    ```

#### Custom Voice Styles

Load custom voice styles from JSON:

=== "Python"

    ```python
    from supertonic import TTS
    from pathlib import Path

    tts = TTS()

    # Load custom style from JSON
    custom_style = tts.get_voice_style_from_path(Path("custom_voice.json"))
    wav, dur = tts.synthesize("Using a custom voice style from JSON file.", voice_style=custom_style)
    ```

=== "CLI"

    ```bash
    supertonic tts 'Using a custom voice style from JSON file.' --custom-style-path custom_voice.json -o output_custom.wav
    ```

### Speech Speed Control

Adjust speech rate from 0.7× (slow) to 2.0× (fast):

| Speed | Description |
|-------|-------------|
| `0.7` | Slow pace |
| `1.0` | Normal pace |
| `1.3` | Faster pace |
| `2.0` | Fast pace |

=== "Python"

    ```python
    from supertonic import TTS

    tts = TTS()
    style = tts.get_voice_style("F1")

    texts = {
        0.7: "This is slow speed demonstration.",
        1.0: "This is normal speed demonstration.",
        1.3: "This is fast speed demonstration.",
        2.0: "This is fastest speed demonstration.",
    }

    for speed, text in texts.items():
        wav, dur = tts.synthesize(text, voice_style=style, speed=speed)
        tts.save_audio(wav, f"output_{speed:.1f}_speed.wav")
    ```

=== "CLI"

    ```bash
    supertonic tts 'This is slow speed demonstration.' --voice F1 --speed 0.7 -o output_0.7_speed.wav
    supertonic tts 'This is normal speed demonstration.' --voice F1 --speed 1.0 -o output_1.0_speed.wav
    supertonic tts 'This is fast speed demonstration.' --voice F1 --speed 1.3 -o output_1.3_speed.wav
    supertonic tts 'This is fastest speed demonstration.' --voice F1 --speed 2.0 -o output_2.0_speed.wav
    ```

### Speech Quality Control

Adjust synthesis quality with `total_steps` parameter:

| Steps | Quality | Synthesis Speed |
|-------|---------|-----------------|
| `2` | Low | Fast |
| `5` | Balanced | Normal |
| `10` | High | Slow |

=== "Python"

    ```python
    from supertonic import TTS

    tts = TTS()
    style = tts.get_voice_style("M1")

    texts = {
        2: "This is low steps demonstration.",
        5: "This is balanced steps demonstration.",
        10: "This is high steps demonstration.",
    }

    for steps, text in texts.items():
        wav, dur = tts.synthesize(text, voice_style=style, total_steps=steps)
        tts.save_audio(wav, f"output_{steps:02d}_steps.wav")
    ```

=== "CLI"

    ```bash
    supertonic tts 'This is low steps demonstration.' --steps 2 -o output_02_steps.wav
    supertonic tts 'This is balanced steps demonstration.' --steps 5 -o output_05_steps.wav
    supertonic tts 'This is high steps demonstration.' --steps 10 -o output_10_steps.wav
    ```

### Long Text Handling

Supertonic automatically chunks long texts for optimal processing:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_chunk_length` | Maximum characters per chunk | `300` |
| `silence_duration` | Silence between chunks (seconds) | `0.3` |

=== "Python"

    ```python
    from supertonic import TTS

    tts = TTS()
    style = tts.get_voice_style("F1")

    long_text = """
    Artificial intelligence has transformed many fields.
    From healthcare to transportation, AI systems are making impacts.
    Natural language processing allows computers to understand human language.
    These advances are opening up new possibilities.
    """

    wav, dur = tts.synthesize(
        long_text,
        voice_style=style,
        max_chunk_length=300,
        silence_duration=0.3
    )
    tts.save_audio(wav, "output.wav")
    ```

=== "CLI"

    ```bash
    TEXT="Artificial intelligence has transformed many fields. From healthcare to transportation, AI systems are making impacts. Natural language processing allows computers to understand human language. These advances are opening up new possibilities."

    supertonic tts "$TEXT" --max-chunk-length 150 --silence-duration 0.3 --voice F1 -o output.wav  # Note: Use double quotes for shell variables
    ```

!!! info "Auto-chunking"
    Text chunking is enabled by default. Supertonic splits by paragraphs and respects sentence boundaries, handling abbreviations like Mr., Dr., Ph.D. correctly.

### Text Validation

Check if your text can be processed before synthesis:

=== "Python"

    ```python
    from supertonic import TTS

    tts = TTS()
    text_processor = tts.model.text_processor

    # Check if text is supported
    text = "Hello World! Welcome to 世界."
    is_valid, unsupported = text_processor.validate_text(text)

    if not is_valid:
        print(f"Unsupported characters: {unsupported}")
        # Will show: ['世', '界'] (or similar)

    # Get all supported characters
    supported_chars = text_processor.supported_character_set
    print(f"Supported characters: {sorted(list(supported_chars))}")
    ```

=== "CLI"

    ```bash
    # This will output an error because the text contains unsupported characters
    supertonic tts 'Hello World! Welcome to 世界.' -o output.wav
    ```


---

## Performance Tuning

### Thread Configuration

By default, ONNX Runtime automatically detects and uses optimal thread counts for your system. For advanced use cases, you can manually configure threads:

| Parameter | Description |
|-----------|-------------|
| `intra_op_num_threads` | Threads for parallelism within each operation |
| `inter_op_num_threads` | Threads for parallelism between operations |

=== "Python"

    ```python
    from supertonic import TTS

    # Auto-detect (recommended)
    tts = TTS()

    # High-performance server
    tts = TTS(intra_op_num_threads=12, inter_op_num_threads=12)

    # Low-resource environment
    tts = TTS(intra_op_num_threads=2, inter_op_num_threads=2)
    ```

=== "CLI"

    ```bash
    # Set thread counts via environment variables
    export SUPERTONIC_INTRA_OP_THREADS=12
    export SUPERTONIC_INTER_OP_THREADS=12

    supertonic tts 'This is the thread configuration demonstration.' -o output.wav --voice M1
    ```

---

## Next Steps

- **[API Reference](api/index.md)** — Complete API documentation
- **[CLI Reference](cli/README.md)** — Full command-line interface guide
