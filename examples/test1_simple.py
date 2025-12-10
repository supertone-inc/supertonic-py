"""
Example 1: Basic Usage

The simplest way to use Supertonic TTS.
"""

import os

from supertonic import TTS

os.makedirs("outputs/test1", exist_ok=True)

# Initialize
tts = TTS()

# Get voice style
style = tts.get_voice_style("M1")

# Generate speech
text = "Welcome to Supertonic, where cutting-edge technology meets natural speech synthesis!"
wav, duration = tts.synthesize(text, voice_style=style)

# Save to file
tts.save_audio(wav, "outputs/test1/basic_output.wav")

print(f"Generated {duration[0]:.2f}s audio → outputs/test1/basic_output.wav")
