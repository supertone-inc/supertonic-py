"""
Example 2: Voice Styles

Demonstrates using different voice styles (M1, M2, F1, F2).
"""

import os

from supertonic import TTS

os.makedirs("outputs/test2", exist_ok=True)

tts = TTS()

# Text to synthesize
text = "Each voice style brings unique tonal qualities and expressiveness to your content."

# Generate speech with each voice style
styles = tts.voice_style_names

for style_name in styles:
    style = tts.get_voice_style(style_name)
    wav, duration = tts.synthesize(text, voice_style=style)
    output_path = f"outputs/test2/voice_style_{style_name}.wav"
    tts.save_audio(wav, output_path)
    print(f"{style_name}: {duration[0]:.2f}s → {output_path}")
