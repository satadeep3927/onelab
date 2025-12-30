# OneLab TTS Library

OneLab is a powerful Python library for Text-to-Speech (TTS) and Podcast generation, built on top of `ChatterboxTurboTTS`. It allows you to easily convert text to speech using various voices and create multi-speaker podcasts with custom logging and progress tracking.

## Features

*   **Text-to-Speech Conversion**: Convert individual text segments to audio using specific voices.
*   **Podcast Generation**: Create full conversations with multiple speakers.
*   **Dynamic Voice Loading**: Automatically detects available voices from a `sample` directory.
*   **Custom Logging**: Detailed logging and progress tracking during generation.
*   **Modular Design**: Clean and extensible architecture.

## Installation

You can install the package using pip (once published) or directly from the source.

### From Source

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Or install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

### Initialization

To use the library, you need to initialize the `OneLab` class. You must provide a `root_dir` which serves as the base path for looking up resources, specifically the `sample` directory containing voice samples.

```python
from src.onelab import OneLab

# Initialize OneLab
# root_dir should point to the folder containing the 'sample' directory
onelab = OneLab(device="cpu", root_dir=".") 
```

### What is `root_dir`?

The `root_dir` parameter is the absolute or relative path to the project root or the directory where your assets are stored. The library expects a `sample` folder inside this `root_dir` containing `.wav` files for each voice (e.g., `charlie.wav`, `emilia.wav`).

Structure expected:
```
root_dir/
└── sample/
    ├── charlie.wav
    ├── emilia.wav
    └── ...
```

### Listing Available Voices

You can list all voices detected in the `sample` directory:

```python
voices = onelab.tts.list_voices()
print("Available voices:", voices)
```

### Converting Text to Speech

To convert a single line of text:

```python
wav = onelab.tts.convert("Hello, world!", voice="charlie")
# Save the output
import torchaudio
torchaudio.save("output.wav", wav, onelab.tts.model.sr)
```

### Creating a Podcast

To create a multi-speaker podcast, define your conversation segments and use the `podcast.create` method:

```python
data = {
    "segments": [
        {"voice": "charlie", "text": "Hello Emilia!"},
        {"voice": "emilia", "text": "Hi Charlie! How are you?"},
    ]
}

# Generate the podcast
podcast_wav = onelab.tts.podcast.create(data)

# Save the podcast
torchaudio.save("podcast.wav", podcast_wav, onelab.tts.model.sr)
```

## API Reference

### `OneLab`

The main entry point for the library.

*   `__init__(device: str = "cpu", root_dir: str = ".")`: Initializes the library.

### `TextToSpeech` (accessed via `onelab.tts`)

Handles TTS operations.

*   `list_voices() -> List[str]`: Returns a list of available voice names.
*   `convert(text: str, voice: str) -> torch.Tensor`: Generates audio for the given text and voice.

### `Podcast` (accessed via `onelab.tts.podcast`)

Handles podcast generation.

*   `create(args: ConversationInput) -> torch.Tensor`: Generates a concatenated audio waveform for the entire conversation.

## License

MIT
