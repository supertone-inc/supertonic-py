"""
Example 6: Long Text Auto-Chunking

Automatically split long texts into manageable chunks with silence in between.
"""

import os

from supertonic import TTS

os.makedirs("outputs/test6", exist_ok=True)
tts = TTS()
style = tts.get_voice_style("F1")

# Long text that exceeds typical length limits
long_text = """
This is a very long text that will be automatically chunked into smaller parts.
The chunking algorithm splits text by paragraphs and sentences intelligently.
It respects sentence boundaries and common abbreviations like Mr., Mrs., Dr., and Prof.

Each chunk will be processed separately and then combined with silence in between.
This makes it possible to generate speech for arbitrarily long texts without running into memory issues.
The default chunk size is 300 characters, but you can adjust it based on your needs.

Here's another paragraph to make the text even longer. This demonstrates how the chunking
works across multiple paragraphs. The algorithm preserves the natural flow of speech by
adding appropriate silence between chunks.
"""

# Test 1: Auto-chunking with default settings
wav, dur = tts.synthesize(
    long_text,
    voice_style=style,
    max_chunk_length=300,  # 300 chars per chunk
    silence_duration=0.3,  # 0.3s silence between chunks
)
tts.save_audio(wav, "outputs/test6/auto_chunk_300.wav")
print(f"Default (300 chars):  {dur[0]:.2f}s")

# Test 2: Smaller chunks
wav, dur = tts.synthesize(long_text, voice_style=style, max_chunk_length=150)
tts.save_audio(wav, "outputs/test6/auto_chunk_150.wav")
print(f"Small (150 chars):    {dur[0]:.2f}s")

# Test 3: Longer silence between chunks
wav, dur = tts.synthesize(long_text, voice_style=style, silence_duration=0.8)
tts.save_audio(wav, "outputs/test6/long_silence.wav")
print(f"Long silence (0.8s):  {dur[0]:.2f}s")

# Test 4: Short text
short_text = "This is a short text that doesn't need chunking."
wav, dur = tts.synthesize(short_text, voice_style=style)
tts.save_audio(wav, "outputs/test6/no_chunk.wav")
print(f"No chunking (short):  {dur[0]:.2f}s")
