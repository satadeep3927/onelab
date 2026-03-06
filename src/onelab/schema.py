from dataclasses import dataclass
from typing import List, TypedDict

@dataclass
class Segment:
    voice: str
    text: str
    repetition_penalty: float = 1.2
    min_p: float = 0
    top_p: float = 0.95
    exaggeration: float = 0.5
    cfg_weight: float = 0
    temperature: float = 0.8
    top_k: int = 1000
    norm_loudness: bool = True


class ConversationInput(TypedDict):
    segments: List[Segment]
