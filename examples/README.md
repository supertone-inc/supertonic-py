# Examples

Example scripts demonstrating Supertonic TTS features.

## Running Examples

Install package:

```bash
pip install supertonic
cd examples
python test1_simple.py
```

## Examples

### 1. Basic Usage (`test1_simple.py`)

The simplest way to generate natural-sounding speech.

```bash
python test1_simple.py
```

### 2. Voice Styles (`test2_voices.py`)

Test all available voice styles (M1, M2, F1, F2).

```bash
python test2_voices.py
```

### 3. Quality Levels (`test3_quality.py`)

Compare different quality levels from fast to ultra-high quality.

```bash
python test3_quality.py
```

### 4. Custom Voice Styles (`test4_custom_reference.py`)

Use custom voice styles with JSON files and dictionaries.

```bash
python test4_custom_reference.py
```

### 5. Speed Control (`test5_speed_control.py`)

Adjust speech speed from 0.7× (slow) to 2.0× (ultra fast).

```bash
python test5_speed_control.py
```

### 6. Long Text Auto-Chunking (`test6_long_text.py`)

Automatically split and process long texts with intelligent chunking.

```bash
python test6_long_text.py
```

### 7. Text Validation (`test7_text_validation.py`)

Check if text can be processed and identify unsupported characters.

```bash
python test7_text_validation.py
```

**Features demonstrated:**
- `validate_text()` - Check if text is supported
- `get_supported_characters()` - Get all supported characters
- `get_unsupported_characters()` - Find unsupported characters
- `get_supported_unicode_ranges()` - Get unicode ranges

### 8. Verbose Mode (`test8_verbose_mode.py`)

Monitor synthesis progress with detailed output for debugging and monitoring.

```bash
python test8_verbose_mode.py
```

**Shows:**
- Text processing details
- Chunk splitting information
- Real-time synthesis progress
- Per-chunk timing
- Final statistics

## Output Structure

```
outputs/
├── test1/ - basic_output.wav
├── test2/ - voice_style_*.wav
├── test3/ - quality_*.wav
├── test4/ - method*.wav
├── test5/ - speed_*.wav
├── test6/ - auto_chunk_*.wav
├── test7/ - (validation output, no audio)
└── test8/ - verbose_demo.wav
```

## Performance Tuning

By default, ONNX Runtime automatically detects optimal thread settings. Manual configuration is optional:

```python
from supertonic import TTS

# Auto-detect (recommended) - works well for most cases
tts = TTS()

# Manual high-performance configuration (if needed)
tts = TTS(intra_op_num_threads=8, inter_op_num_threads=8)

# Manual low-resource configuration (if needed)
tts = TTS(intra_op_num_threads=2, inter_op_num_threads=2)
```

Or use environment variables:

```bash
export SUPERTONIC_INTRA_OP_THREADS=8
export SUPERTONIC_INTER_OP_THREADS=8
python test1_simple.py
```

**Manual configuration guide (only if auto-detection is suboptimal):**
- **Auto (default)**: Let ONNX Runtime decide (recommended)
- **2-4 cores**: `intra=2, inter=2`
- **8-16 cores**: `intra=8, inter=8`
- **16+ cores**: `intra=12, inter=12`

## Notes

- Models loaded from `~/.cache/supertonic/models`
- First run auto-downloads model from HuggingFace
- CPU-only inference for stability
- Thread count: Auto-detected by ONNX Runtime (configurable via initialization or environment variables if needed)
