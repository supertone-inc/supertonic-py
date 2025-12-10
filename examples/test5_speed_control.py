"""
Example 5: Speed Control

Adjust speech speed from 0.7x (slow) to 2.0x (ultra fast).
"""

import os

from supertonic import TTS

os.makedirs("outputs/test5", exist_ok=True)

tts = TTS()
style = tts.get_voice_style("F1")

# Test different speeds with speed-appropriate text
speeds = [
    ("slow", 0.7, "This is a slow speed demonstration. Every word is clearly articulated."),
    ("normal", 1.0, "This is normal speed. Natural and comfortable to listen to."),
    ("fast", 1.5, "This is fast speed. Quick but still understandable."),
    (
        "ultra_fast",
        2.0,
        "When the speed is set to ultra fast, sentences are spoken almost twice as quickly as usual.",
    ),
]

for name, speed, text in speeds:
    wav, duration = tts.synthesize(text, voice_style=style, speed=speed)
    output_path = f"outputs/test5/speed_{speed:.1f}_{name}.wav"
    tts.save_audio(wav, output_path)

    print(f"{speed:.1f}x ({name:>10}): {duration[0]:.2f}s → {output_path}")
