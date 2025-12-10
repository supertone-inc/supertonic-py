"""
Example 4: Custom Voice Styles

Use custom voice styles with JSON files and dictionaries.

CLI Usage:
    # Use custom voice style from JSON file
    supertonic tts "Custom voice test" -o output.wav --custom-style-path ./my_voice.json

    # With verbose output
    supertonic tts "Custom voice test" -o output.wav --custom-style-path /path/to/F1.json -v
"""

import os

from supertonic import TTS

os.makedirs("outputs/test4", exist_ok=True)

tts = TTS()

text = "Custom voice styles enable precise control over synthesis."

# Method 1: Style name (recommended)
style = tts.get_voice_style("M1")
wav, dur = tts.synthesize(text, voice_style=style)
tts.save_audio(wav, "outputs/test4/method1_name.wav")
print(f"1. Style name: {dur[0]:.2f}s")

# Method 2: JSON file path
voice_styles_dir = tts.model_dir / "voice_styles"
f1_path = voice_styles_dir / "F1.json"
style = tts.get_voice_style_from_path(f1_path)
wav, dur = tts.synthesize(text, voice_style=style)
tts.save_audio(wav, "outputs/test4/method2_json.wav")
print(f"2. JSON path:  {dur[0]:.2f}s")

# Method 3: Different styles for comparison
styles_to_compare = ["M1", "M2", "F1", "F2"]
for style_name in styles_to_compare:
    style = tts.get_voice_style(style_name)
    wav, dur = tts.synthesize(text, voice_style=style)
    tts.save_audio(wav, f"outputs/test4/comparison_{style_name}.wav")
    print(f"3. {style_name}: {dur[0]:.2f}s")
