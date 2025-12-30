import logging
import os

from .schema import ConversationInput, Segment
from .tts import TextToSpeech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class OneLab:
    def __init__(self, device: str = "cpu", root_dir: str = "."):
        self.tts = TextToSpeech(device=device, root_dir=os.path.abspath(root_dir))


__all__ = ["OneLab", "ConversationInput", "Segment"]
