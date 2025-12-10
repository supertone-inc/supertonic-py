# supertonic say

Generate speech from text and play it directly without saving a file.

!!! note "Requires `sounddevice`"
    This command requires the `sounddevice` package for audio playback.
    Install it with: `pip install supertonic[playback]` or `pip install sounddevice`

## Usage

```bash
supertonic say TEXT [OPTIONS]
```

## Examples

```bash
# Basic usage - play speech directly
supertonic say 'Hello, welcome to the world!'

# Use a different voice
supertonic say 'This is a female voice style.' --voice F1

# Adjust speech speed (faster)
supertonic say 'This sentence is spoken at a faster speed than normal.' --speed 1.5

# Higher quality with more steps
supertonic say 'This is a high quality output.' --steps 10

# Use custom voice style
supertonic say 'This is a custom voice test.' --custom-style-path ./my_voice.json

# Verbose mode to see processing details
supertonic say 'Verbose mode shows detailed processing information.' -v
```

## Arguments

--8<-- "docs/argparse/say.inc.md"
