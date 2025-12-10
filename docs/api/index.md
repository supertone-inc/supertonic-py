# API Reference

```python
from supertonic import TTS

tts = TTS(auto_download=True)       # Initialize TTS engine

style = tts.get_voice_style(voice_name="M1")   # Get a voice style: M1, M2, F1, F2

wav, duration = tts.synthesize(
    text="Your text here.",         # Text to synthesize
    voice_style=style,              # Voice style object
    total_steps=5,                  # Quality: 2 (low quality) to 15 (high quality)
    speed=1.05,                     # Speed: 0.7 (slow) to 2.0 (fast)
    max_chunk_length=300,           # Max characters per chunk
    silence_duration=0.3,           # Silence between chunks (seconds)
    verbose=False                   # Show detailed progress (default: False)
)
```

## Modules

- **[pipeline](pipeline.md)** - High-level TTS interface with automatic model loading and voice style management
- **[core](core.md)** - Core TTS engine classes and data structures
- **[loader](loader.md)** - Functions for loading models and voice styles
- **[utils](utils.md)** - Helper functions for text processing and audio utilities
- **[config](config.md)** - Configuration constants and default values
- **[cli](cli.md)** - Command-line interface implementation
