> **This repository is archived. Development and support have ended.**
>
> The code remains available for reference under the terms of the [LICENSE](./LICENSE). No updates, bug fixes, security patches, or support will be provided, and issues and pull requests are no longer monitored. The software is provided "AS IS" as set forth in the license, without any warranty or ongoing responsibility of the original authors.

# Supertonic 3 — Lightning Fast, On-Device TTS

<p align="center">
  <img src="https://raw.githubusercontent.com/supertone-inc/supertonic-py/main/assets/images/Supertonic3_HeroImage.png" alt="Supertonic 3 Banner">
</p>

[![GitHub | Official Repo](https://img.shields.io/badge/GitHub-Official%20Repo-black?logo=github)](https://github.com/supertone-inc/supertonic)
[![GitHub | Python Package](https://img.shields.io/badge/GitHub-Python%20Package-black?logo=github)](https://github.com/supertone-inc/supertonic-py)
[![Docs | Python PyPI](https://img.shields.io/badge/Docs-Python%20PyPI-blue?logo=readthedocs&logoColor=white)](https://supertone-inc.github.io/supertonic-py)
[![DemoPage | Audio Samples](https://img.shields.io/badge/DemoPage-Audio%20Samples-F5D90A?labelColor=0B0C0E)](https://supertonic3.github.io/)
[![Voice Builder | Cloning Demo](https://img.shields.io/badge/Voice%20Builder-Cloning%20Demo-3457D5?logo=soundcloud&logoColor=white)](https://supertonic.supertone.ai/voice_builder)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/Supertone/supertonic-3)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/Supertone/supertonic-3)
[![PyPI version](https://badge.fury.io/py/supertonic.svg)](https://badge.fury.io/py/supertonic)

> **Supertonic-3**: Multilingual synthesis across **31 languages**, plus a `na` fallback for text whose language is unknown or outside the supported set.


## Quick Start

```bash
pip install supertonic
```

### Python

Every parameter is annotated inline, so the snippet doubles as
copy-and-paste documentation for an LLM assistant:

```python
from supertonic import TTS

# Note: first run downloads the model (~400MB) into ~/.cache/supertonic3/
tts = TTS(auto_download=True)       # Initialize TTS engine

style = tts.get_voice_style(voice_name="M1")   # 10 built-in voices: M1–M5, F1–F5

wav, duration = tts.synthesize(
    text="Supertonic is a lightning fast, on-device TTS system.",
    voice_style=style,              # Voice style object
    total_steps=8,                  # Quality: 5 (low) to 12 (high), default 8
    speed=1.05,                     # Speed: 0.7 (slow) to 2.0 (fast)
    max_chunk_length=300,           # Max characters per chunk (auto: 120 for Korean)
    silence_duration=0.3,           # Silence between chunks (seconds)
    lang="en",                      # ISO code; see "Supported Languages" below
    verbose=False,                  # Show detailed progress (default: False)
)
tts.save_audio(wav, "output.wav")

# Multilingual — just swap `lang` and the input text
wav_ko, _ = tts.synthesize("회의는 잠시 후에 시작되며 모두가 자리에 앉아 기다립니다.", voice_style=style, lang="ko")
wav_es, _ = tts.synthesize("La reunión comienza pronto y todos se sientan en silencio para escuchar.", voice_style=style, lang="es")
```

#### Custom voices (Voice Builder)

`get_voice_style()` loads one of the ten built-in voices (M1–M5, F1–F5).
To use a voice created in
[Voice Builder](https://supertonic.supertone.ai/voice_builder)
(zero-shot cloning from a short reference clip), pass its JSON export to
`get_voice_style_from_path()`:

```python
# Any voice-style JSON works here:
#   - a Voice Builder export, or
#   - one of the bundled defaults at
#     ~/.cache/supertonic3/voice_styles/{M1..M5,F1..F5}.json
#     (downloaded alongside the model on first run)
# ex)
# style = tts.get_voice_style_from_path("~/.cache/supertonic3/voice_styles/M1.json")

# download a custom voice style from a JSON file (e.g., exported from Voice Builder)
style = tts.get_voice_style_from_path("voices/my_voice.json")
wav, _ = tts.synthesize("Hello in my own cloned voice.", voice_style=style, lang="en")
tts.save_audio(wav, "output_own_voice.wav")

```

### CLI

```bash
# Note: first run will download the model (~400MB) from HuggingFace
supertonic tts 'Supertonic is a lightning fast, on-device TTS system.' -o output.wav

# Pick a built-in voice and bump quality
supertonic tts 'Use a different voice.' -o output.wav --voice F1 --steps 10

# Use a custom voice — Voice Builder export, or a bundled
# ~/.cache/supertonic3/voice_styles/*.json file
supertonic tts 'Hello in my own cloned voice.' -o output.wav \
  --custom-style-path voices/my_voice.json

# Multilingual support — each language with natural text handling
supertonic tts '회의는 잠시 후에 시작되며 모두가 자리에 앉아 기다립니다.' -o korean.wav --lang ko
supertonic tts 'La reunión comienza pronto y todos se sientan en silencio para escuchar.' -o spanish.wav --lang es
supertonic tts 'A reunião começa em breve e todos se sentam em silêncio para ouvir.' -o portuguese.wav --lang pt
```

## Local Server (HTTP)

`supertonic serve` runs a thin local HTTP wrapper around the same engine. It
makes Supertonic easy to call from environments where embedding a Python
interpreter is awkward — n8n, browser extensions, Electron apps, Unity, Home
Assistant, robotics devices, or anything that already speaks the OpenAI Audio
Speech API.

### Install and run

```bash
pip install 'supertonic[serve]'                    # adds fastapi + uvicorn
supertonic serve --host 127.0.0.1 --port 7788      # defaults; loopback only
```

The first run downloads the model (~400MB) just like the SDK. Once it's up:

- Synthesis endpoint: `http://127.0.0.1:7788/v1/tts`
- OpenAI-compatible alias: `http://127.0.0.1:7788/v1/audio/speech`
- Interactive OpenAPI docs: `http://127.0.0.1:7788/docs`

`--host` defaults to `127.0.0.1`. Binding to any other interface is opt-in
and prints a one-line warning — put it behind a reverse proxy if you do.

### Generate audio (two ways)

**Native `/v1/tts`** — full Supertonic parameter set:

```bash
curl -X POST http://127.0.0.1:7788/v1/tts \
  -H 'content-type: application/json' \
  -d '{
        "text": "Supertonic is a lightning fast, on-device TTS system.",
        "voice": "M1",
        "lang": "en",
        "steps": 8,
        "speed": 1.05,
        "response_format": "wav"
      }' \
  -o output.wav
```

Response is the audio bytes (`audio/wav` by default). Useful headers:
`X-Audio-Duration` (seconds), `X-Sample-Rate`, and `X-Supertonic-Version`.
Supported `response_format` values: `wav`, `flac`, `ogg` (Vorbis).

**OpenAI-compatible `/v1/audio/speech`** — clients that already speak the
OpenAI API only need to swap the base URL:

```bash
curl -X POST http://127.0.0.1:7788/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{
        "model": "supertonic-3",
        "input": "Supertonic is a lightning fast, on-device TTS system.",
        "voice": "M1",
        "response_format": "wav"
      }' \
  -o output.wav
```

Multilingual works the same way — set `lang` to any code from
[Supported Languages](#supported-languages) (or `na` for the fallback).

### Custom voices (Voice Builder JSON)

A voice JSON exported from
[Voice Builder](https://supertonic.supertone.ai/voice_builder) (or any of the
bundled `~/.cache/supertonic3/voice_styles/*.json` files) can be uploaded
once and then referenced by name on every subsequent request.

**Import** — `multipart/form-data` is the simplest path:

```bash
# Upload my_voice.json; the stem of the filename becomes its name.
curl -X POST http://127.0.0.1:7788/v1/styles/import \
  -F "file=@voices/my_voice.json"
# → {"name":"my_voice","stored_at":"~/.cache/supertonic3/custom_styles/my_voice.json"}

# Override the name explicitly, and allow overwriting an existing entry:
curl -X POST "http://127.0.0.1:7788/v1/styles/import?overwrite=true" \
  -F "file=@voices/my_voice.json" \
  -F "name=demo_voice"
```

**Synthesize with the imported voice** — just pass its name as `voice`:

```bash
curl -X POST http://127.0.0.1:7788/v1/tts \
  -H 'content-type: application/json' \
  -d '{"text":"Hello in my own cloned voice.","voice":"my_voice","lang":"en"}' \
  -o output_own_voice.wav
```

Imported voices are persisted **per model** alongside the bundled voice
styles — e.g. `~/.cache/supertonic3/custom_styles/<name>.json` for
`supertonic-3`, `~/.cache/supertonic2/custom_styles/<name>.json` for
`supertonic-2`. They are re-loaded automatically on the next `supertonic
serve` start. Names that collide with the built-ins (`M1`–`M5`, `F1`–`F5`)
are rejected; existing custom names return `409` unless you pass
`?overwrite=true`. `GET /v1/styles` lists everything currently available
for the loaded model.

### Batch synthesis

`POST /v1/tts/batch` accepts up to 64 items in a single request and returns
each result as base64-encoded audio. Per-item `voice` / `lang` / `speed` can
differ — useful for narration jobs that mix speakers or languages.

```bash
curl -X POST http://127.0.0.1:7788/v1/tts/batch \
  -H 'content-type: application/json' \
  -d '{
        "items": [
          {"text": "Supertonic is a lightning fast, on-device TTS system.", "voice": "M1", "lang": "en"},
          {"text": "회의는 잠시 후에 시작되며 모두가 자리에 앉아 기다립니다.", "voice": "F1", "lang": "ko"},
          {"text": "La reunión comienza pronto y todos se sientan en silencio para escuchar.", "voice": "F1", "lang": "es"}
        ],
        "response_format": "wav",
        "defaults": {"steps": 8, "speed": 1.05}
      }'
```

Response:

```json
{
  "items": [
    {"audio_base64": "...", "duration_s": 4.32, "format": "wav", "sample_rate": 44100},
    {"audio_base64": "...", "duration_s": 4.88, "format": "wav", "sample_rate": 44100},
    {"audio_base64": "...", "duration_s": 5.36, "format": "wav", "sample_rate": 44100}
  ]
}
```

Each item carries fully self-contained audio bytes, so writing them out is a
one-liner:

```bash
curl -fsS -X POST http://127.0.0.1:7788/v1/tts/batch \
  -H 'content-type: application/json' \
  -d '@payload.json' \
| python3 -c '
import sys, json, base64, pathlib
for i, item in enumerate(json.load(sys.stdin)["items"]):
    pathlib.Path(f"batch_{i}.wav").write_bytes(base64.b64decode(item["audio_base64"]))
'
```

Items are processed sequentially (the underlying ONNX session is serialized
per process), so batching is about cutting HTTP round-trips and packaging
related work together, not about parallel speed-up. Any per-item failure
returns a `400` with `items[<index>]` in the error message — no audio is
emitted partially.

## Requirements

Supertonic has **minimal dependencies** - just 4 core libraries:

- **onnxruntime** - Fast ONNX model inference
- **numpy** - Numerical operations
- **soundfile** - Audio file I/O
- **huggingface-hub** - Model downloads


## ✨ Highlights

**⚡ Blazingly Fast** — Low-latency, real-time synthesis across desktop, browser, mobile, and edge — fast enough to turn an entire webpage into audio in under a second

**🌍 31-Language Multilingual** — Synthesize directly from text across 31 languages, or pass `lang="na"` to let Supertonic process the text language-agnostically when you don't know the input language — no separate language adapters needed

**🪶 99M-Parameter Open-Weight Model** — A compact, fully open-weight checkpoint — a fraction of the size of 0.7B–2B class open TTS systems — for smaller downloads, faster cold starts, and lower memory footprint

**📱 Edge-Device Ready** — Runs locally on desktop, mobile, browsers, and resource-constrained hardware like Raspberry Pi or e-readers, with zero network dependency, complete privacy, and no GPU required

**🔊 44.1kHz High-Quality Audio** — Outputs studio-grade 44.1kHz 16-bit WAV directly, ready for production playback without any external upsampler

**🎭 Expression Tags** — 10 inline tags (e.g. `<laugh>`, `<breath>`, `<sigh>`) bring natural human nuance into generated speech without prompt engineering or reference audio

**🛠️ Multi-Runtime SDKs** — Ready-to-use examples through ONNX Runtime across Python, Node.js, Browser (WebGPU), Java, C++, C#, Go, Swift, iOS, Rust, and Flutter

## Supported Languages

Supertonic-3 supports the following 31 ISO codes, plus a special `na` fallback for unknown / unsupported languages:

| Code | Language | Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|------|----------|
| `en` | English | `ko` | Korean | `ja` | Japanese | `ar` | Arabic |
| `bg` | Bulgarian | `cs` | Czech | `da` | Danish | `de` | German |
| `el` | Greek | `es` | Spanish | `et` | Estonian | `fi` | Finnish |
| `fr` | French | `hi` | Hindi | `hr` | Croatian | `hu` | Hungarian |
| `id` | Indonesian | `it` | Italian | `lt` | Lithuanian | `lv` | Latvian |
| `nl` | Dutch | `pl` | Polish | `pt` | Portuguese | `ro` | Romanian |
| `ru` | Russian | `sk` | Slovak | `sl` | Slovenian | `sv` | Swedish |
| `tr` | Turkish | `uk` | Ukrainian | `vi` | Vietnamese | `na` | *unknown / fallback* |

```python
# Pick any supported code, or use 'na' for text whose language is unknown
wav, _ = tts.synthesize("Some uncommon text.", voice_style=style, lang="na")
```



## Performance

We evaluated Supertonic's performance (with 2 inference steps) using two key metrics across input texts of varying lengths: Short (59 chars), Mid (152 chars), and Long (266 chars).

**Metrics:**
- **Characters per Second**: Measures throughput by dividing the number of input characters by the time required to generate audio. Higher is better.
- **Real-time Factor (RTF)**: Measures the time taken to synthesize audio relative to its duration. Lower is better (e.g., RTF of 0.1 means it takes 0.1 seconds to generate one second of audio).

### Characters per Second
| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 912 | 1048 | 1263 |
| **Supertonic** (M4 pro - WebGPU) | 996 | 1801 | 2509 |
| **Supertonic** (RTX4090) | 2615 | 6548 | 12164 |
| `API` [ElevenLabs Flash v2.5](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | 144 | 209 | 287 |
| `API` [OpenAI TTS-1](https://platform.openai.com/docs/guides/text-to-speech) | 37 | 55 | 82 |
| `API` [Gemini 2.5 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation) | 12 | 18 | 24 |
| `API` [Supertone Sona speech 1](https://docs.supertoneapi.com/en/api-reference/endpoints/text-to-speech) | 38 | 64 | 92 |
| `Open` [Kokoro](https://github.com/hexgrad/kokoro/) | 104 | 107 | 117 |
| `Open` [NeuTTS Air](https://github.com/neuphonic/neutts-air) | 37 | 42 | 47 |

> **Notes:**
> `API` = Cloud-based API services (measured from Seoul)
> `Open` = Open-source models
> Supertonic (M4 pro - CPU) and (M4 pro - WebGPU): Tested with ONNX
> Supertonic (RTX4090): Tested with PyTorch model
> Kokoro: Tested on M4 Pro CPU with ONNX
> NeuTTS Air: Tested on M4 Pro CPU with Q8-GGUF

### Real-time Factor

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 0.015 | 0.013 | 0.012 |
| **Supertonic** (M4 pro - WebGPU) | 0.014 | 0.007 | 0.006 |
| **Supertonic** (RTX4090) | 0.005 | 0.002 | 0.001 |
| `API` [ElevenLabs Flash v2.5](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | 0.133 | 0.077 | 0.057 |
| `API` [OpenAI TTS-1](https://platform.openai.com/docs/guides/text-to-speech) | 0.471 | 0.302 | 0.201 |
| `API` [Gemini 2.5 Flash TTS](https://ai.google.dev/gemini-api/docs/speech-generation) | 1.060 | 0.673 | 0.541 |
| `API` [Supertone Sona speech 1](https://docs.supertoneapi.com/en/api-reference/endpoints/text-to-speech) | 0.372 | 0.206 | 0.163 |
| `Open` [Kokoro](https://github.com/hexgrad/kokoro/) | 0.144 | 0.124 | 0.126 |
| `Open` [NeuTTS Air](https://github.com/neuphonic/neutts-air) | 0.390 | 0.338 | 0.343 |

<details>
<summary><b>Additional Performance Data (5-step inference)</b></summary>

<br>

**Characters per Second (5-step)**

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 596 | 691 | 850 |
| **Supertonic** (M4 pro - WebGPU) | 570 | 1118 | 1546 |
| **Supertonic** (RTX4090) | 1286 | 3757 | 6242 |

**Real-time Factor (5-step)**

| System | Short (59 chars) | Mid (152 chars) | Long (266 chars) |
|--------|-----------------|----------------|-----------------|
| **Supertonic** (M4 pro - CPU) | 0.023 | 0.019 | 0.018 |
| **Supertonic** (M4 pro - WebGPU) | 0.024 | 0.012 | 0.010 |
| **Supertonic** (RTX4090) | 0.011 | 0.004 | 0.002 |

</details>

### Natural Text Handling

Supertonic is designed to handle complex, real-world text inputs that contain numbers, currency symbols, abbreviations, dates, and proper nouns.

> 🎧 **View audio samples more easily**: Check out our [**Interactive Demo**](https://huggingface.co/spaces/Supertone/supertonic-3#text-handling) for a better viewing experience of all audio examples

**Overview of Test Cases:**

| Category | Key Challenges | Supertonic | ElevenLabs | OpenAI | Gemini | Microsoft |
|:--------:|:--------------:|:----------:|:----------:|:------:|:------:|:---------:|
| Financial Expression | Decimal currency, abbreviated magnitudes (M, K), currency symbols, currency codes | ✅ | ❌ | ❌ | ❌ | ❌ |
| Time and Date | Time notation, abbreviated weekdays/months, date formats | ✅ | ❌ | ❌ | ❌ | ❌ |
| Phone Number | Area codes, hyphens, extensions (ext.) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Technical Unit | Decimal numbers with units, abbreviated technical notations | ✅ | ❌ | ❌ | ❌ | ❌ |

<details>
<summary><b>Example 1: Financial Expression</b></summary>

<br>

**Text:**
> "The startup secured **$5.2M** in venture capital, a huge leap from their initial **$450K** seed round."

**Challenges:**
- Decimal point in currency ($5.2M should be read as "five point two million")
- Abbreviated magnitude units (M for million, K for thousand)
- Currency symbol ($) that needs to be properly pronounced as "dollars"

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1eancUOhiSXCVoTu9ddh4S-OcVQaWrPV-/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1-r2scv7XQ1crIDu6QOh3eqVl445W6ap_/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1MFDXMjfmsAVOqwPx7iveS0KUJtZvcwxB/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1dEHpNzfMUucFTJPQK0k4RcFZvPwQTt09/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1b69XWBQnSZZ0WZeR3avv7E8mSdoN6p6P/view?usp=sharing) |

</details>

<details>
<summary><b>Example 2: Time and Date</b></summary>

<br>

**Text:**
> "The train delay was announced at **4:45 PM** on **Wed, Apr 3, 2024** due to track maintenance."

**Challenges:**
- Time expression with PM notation (4:45 PM)
- Abbreviated weekday (Wed)
- Abbreviated month (Apr)
- Full date format (Apr 3, 2024)

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1ehkZU8eiizBenG2DgR5tzBGQBvHS0Uaj/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1ta3r6jFyebmA-sT44l8EaEQcMLVmuOEr/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1sskmem9AzHAQ3Hv8DRSZoqX_pye-CXuU/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1zx9X8oMsLMXW0Zx_SURoqjju-By2yh_n/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1ZpGEstZr4hA0EdAWBMCUFFWuAkIpYsVh/view?usp=sharing) |

</details>

<details>
<summary><b>Example 3: Phone Number</b></summary>

<br>

**Text:**
> "You can reach the hotel front desk at **(212) 555-0142 ext. 402** anytime."

**Challenges:**
- Area code in parentheses that should be read as separate digits
- Phone number with hyphen separator (555-0142)
- Abbreviated extension notation (ext.)
- Extension number (402)

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1z-e5iTsihryMR8ll1-N1YXkB2CIJYJ6F/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1HAzVXFTZfZm0VEK2laSpsMTxzufcuaxA/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/15tjfAmb3GbjP_kmvD7zSdIWkhtAaCPOg/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1BCL8n7yligUZyso970ud7Gf5NWb1OhKD/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1c0c0YM_Qm7XxSk2uSVYLbITgEDTqaVzL/view?usp=sharing) |

</details>

<details>
<summary><b>Example 4: Technical Unit</b></summary>

<br>

**Text:**
> "Our drone battery lasts **2.3h** when flying at **30kph** with full camera payload."

**Challenges:**
- Decimal time duration with abbreviation (2.3h = two point three hours)
- Speed unit with abbreviation (30kph = thirty kilometers per hour)
- Technical abbreviations (h for hours, kph for kilometers per hour)
- Technical/engineering context requiring proper pronunciation

**Audio Samples:**

| System | Result | Audio Sample |
|--------|--------|--------------|
| **Supertonic** | ✅ | [🎧 Play Audio](https://drive.google.com/file/d/1kvOBvswFkLfmr8hGplH0V2XiMxy1shYf/view?usp=sharing) |
| ElevenLabs Flash v2.5 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1_SzfjWJe5YEd0t3R7DztkYhHcI_av48p/view?usp=sharing) |
| OpenAI TTS-1 | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1P5BSilj5xFPTV2Xz6yW5jitKZohO9o-6/view?usp=sharing) |
| Gemini 2.5 Flash TTS | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1GU82SnWC50OvC8CZNjhxvNZFKQb7I9_Y/view?usp=sharing) |
| VibeVoice Realtime 0.5B | ❌ | [🎧 Play Audio](https://drive.google.com/file/d/1lUTrxrAQy_viEK2Hlu3KLLtTCe8jvbdV/view?usp=sharing) |

</details>

> **Note:** These samples demonstrate how each system handles text normalization and pronunciation of complex expressions **without requiring pre-processing or phonetic annotations**.


## Citation

The following papers describe the core technologies used in Supertonic. If you use this system in your research or find these techniques useful, please consider citing the relevant papers:

### SupertonicTTS: Main Architecture

This paper introduces the overall architecture of SupertonicTTS, including the speech autoencoder, flow-matching based text-to-latent module, and efficient design choices.

```bibtex
@article{kim2025supertonic,
  title={SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System},
  author={Kim, Hyeongju and Yang, Jinhyeok and Yu, Yechan and Ji, Seunghun and Morton, Jacob and Bous, Frederik and Byun, Joon and Lee, Juheon},
  journal={arXiv preprint arXiv:2503.23108},
  year={2025},
  url={https://arxiv.org/abs/2503.23108}
}
```

### Length-Aware RoPE: Text-Speech Alignment

This paper presents Length-Aware Rotary Position Embedding (LARoPE), which improves text-speech alignment in cross-attention mechanisms.

```bibtex
@article{kim2025larope,
  title={Length-Aware Rotary Position Embedding for Text-Speech Alignment},
  author={Kim, Hyeongju and Lee, Juheon and Yang, Jinhyeok and Morton, Jacob},
  journal={arXiv preprint arXiv:2509.11084},
  year={2025},
  url={https://arxiv.org/abs/2509.11084}
}
```

### Self-Purifying Flow Matching: Training with Noisy Labels

This paper describes the self-purification technique for training flow matching models robustly with noisy or unreliable labels.

```bibtex
@article{kim2025spfm,
  title={Training Flow Matching Models with Reliable Labels via Self-Purification},
  author={Kim, Hyeongju and Yu, Yechan and Yi, June Young and Lee, Juheon},
  journal={arXiv preprint arXiv:2509.19091},
  year={2025},
  url={https://arxiv.org/abs/2509.19091}
}
```

## Related Projects

**🏠 Main Repository**: [github.com/supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)

**🎧 Try it live**: [Hugging Face Spaces](https://huggingface.co/spaces/Supertone/supertonic-3)

**🤗 Model Repository**: [Hugging Face Models (Supertonic-3)](https://huggingface.co/Supertone/supertonic-3)

## License

**Code**: [MIT License](LICENSE)

**Model**: [OpenRAIL-M License](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE)


Copyright © 2025 Supertone Inc.
