from typing import TypedDict, List

class Segment(TypedDict):
    voice: str
    text: str

class ConversationInput(TypedDict):
    segments: List[Segment]
