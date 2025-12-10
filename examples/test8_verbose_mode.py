"""
Example 8: Verbose Mode

Monitor synthesis progress with detailed output.
"""

import os

from supertonic import TTS

os.makedirs("outputs/test8", exist_ok=True)

tts = TTS()
style = tts.get_voice_style("M1")

# Long text with multiple chunks
long_text = """
This is a longer demonstration text that will be split into multiple chunks.
Each chunk is processed separately and then combined with silence in between.
Verbose mode shows you exactly what's happening during synthesis.

This helps you understand how the text is being processed,
how many chunks are created, and how long each chunk takes to synthesize.
Perfect for debugging and monitoring production workloads.
"""

print("=" * 70)
print("Verbose Mode Example")
print("=" * 70)
print()

# Synthesize with verbose output
wav, duration = tts.synthesize(
    long_text,
    voice_style=style,
    total_steps=5,
    max_chunk_length=150,
    silence_duration=0.3,
    verbose=True,  # Enable detailed progress output
)

# Save
tts.save_audio(wav, "outputs/test8/verbose_demo.wav")

print()
print("=" * 70)
print(f"Final audio: {duration[0]:.2f}s → outputs/test8/verbose_demo.wav")
print("=" * 70)
