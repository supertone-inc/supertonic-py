# Supertonic — Lightning Fast, On-Device TTS

<p align="center">
  <img src="assets/images/Supertonic_IMG_v02_4x.webp" alt="Supertonic Banner">
</p>

[![GitHub](https://img.shields.io/badge/GitHub-supertonic-black?logo=github)](https://github.com/supertone-inc/supertonic)
[![GitHub](https://img.shields.io/badge/GitHub-supertonic--py-black?logo=github)](https://github.com/supertone-inc/supertonic-py)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/Supertone/supertonic)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/Supertone/supertonic)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/supertone-inc/supertonic-py/blob/main/notebook/supertonic_demo.ipynb)
[![Docs](https://img.shields.io/badge/Docs-supertonic--py-orange?logo=readthedocs&logoColor=white)](https://supertone-inc.github.io/supertonic-py)

## Quick Start

```bash
pip install supertonic
```

### CLI

```bash
# Note: First run will download the model (~260MB) from HuggingFace
supertonic tts 'Supertonic is a lightning fast, on-device TTS system.' -o output.wav
```

### Python

```python
from supertonic import TTS

# Note: First run downloads model automatically (~260MB)
tts = TTS(auto_download=True)

# Get a voice style
style = tts.get_voice_style(voice_name="M1")

# Generate speech
text = "The train delay was announced at 4:45 PM on Wed, Apr 3, 2024 due to track maintenance."
wav, duration = tts.synthesize(text, voice_style=style)

# Save to file
tts.save_audio(wav, "output.wav")
```

## Requirements

Supertonic has **minimal dependencies** - just 4 core libraries:

- **onnxruntime** - Fast ONNX model inference
- **numpy** - Numerical operations
- **soundfile** - Audio file I/O
- **huggingface-hub** - Model downloads


## Key Features

**⚡ Blazingly Fast**: Generates speech up to **167× faster than real-time** on consumer hardware (M4 Pro)

**🪶 Ultra Lightweight**: Only **66M parameters**, optimized for efficient on-device performance

**📱 On-Device Capable**: **Complete privacy** and **zero latency**

**🎨 Natural Text Handling**: Seamlessly processes complex expressions without G2P module

**⚙️ Highly Configurable**: Adjust inference steps, batch processing, and other parameters

**🧩 Flexible Deployment**: Deploy across servers, browsers, and edge devices



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

> 🎧 **View audio samples more easily**: Check out our [**Interactive Demo**](https://huggingface.co/spaces/Supertone/supertonic#text-handling) for a better viewing experience of all audio examples

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

**🎧 Try it live**: [Hugging Face Spaces](https://huggingface.co/spaces/Supertone/supertonic)

**🤗 Model Repository**: [Hugging Face Models](https://huggingface.co/Supertone/supertonic)

## License

**Code**: [MIT License](LICENSE)

**Model**: [OpenRAIL-M License](https://huggingface.co/Supertone/supertonic/blob/main/LICENSE)


Copyright © 2025 Supertone Inc.
