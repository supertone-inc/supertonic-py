"""
Example 3: Quality Levels

Compare different quality levels (speed vs quality trade-off).
"""

import os
import time

from supertonic import TTS

os.makedirs("outputs/test3", exist_ok=True)

tts = TTS()

text = "Quality levels balance computational efficiency and audio fidelity."
style = tts.get_voice_style("M1")

# Test different quality levels
quality_levels = [
    ("fast", 3),
    ("balanced", 5),
    ("high", 10),
    ("ultra", 15),
]

for name, steps in quality_levels:
    start = time.time()
    wav, duration = tts.synthesize(text, voice_style=style, total_steps=steps)
    elapsed = time.time() - start

    output_path = f"outputs/test3/quality_{name}_steps{steps:02d}.wav"
    tts.save_audio(wav, output_path)

    print(f"{name:>8} (steps={steps:2d}): {elapsed:.2f}s → {output_path}")
