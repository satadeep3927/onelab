import logging
import os
import re
from typing import Dict, List

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

from .schema import ConversationInput

logger = logging.getLogger("OneLabTTS")


def _chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """
    Split text into smaller chunks to handle token limitations.
    
    Args:
        text: The text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    # Try to split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is longer than max_chars, split it by words
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            words = sentence.split()
            current_word_chunk = ""
            
            for word in words:
                if len(current_word_chunk + " " + word) > max_chars:
                    if current_word_chunk:
                        chunks.append(current_word_chunk.strip())
                    current_word_chunk = word
                else:
                    current_word_chunk += (" " + word) if current_word_chunk else word
            
            if current_word_chunk:
                chunks.append(current_word_chunk.strip())
        
        elif len(current_chunk + " " + sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence) if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


class Podcast:
    def __init__(
        self, model: ChatterboxTurboTTS, voice_map: Dict[str, str], root_dir: str
    ):
        self.model = model
        self.voice_map = voice_map
        self.root_dir = root_dir

    def _get_audio_prompt_path(self, voice: str) -> str:
        rel_path = self.voice_map.get(voice)
        if not rel_path:
            raise ValueError(
                f"Voice '{voice}' not found. Available voices: {list(self.voice_map.keys())}"
            )

        # Construct absolute path
        path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio prompt file not found at: {path}")
        return path

    def create(self, args: ConversationInput, max_chars_per_chunk: int = 500) -> torch.Tensor:
        """
        Creates a podcast from a conversation input with automatic text chunking.
        
        Args:
            args: Conversation input data
            max_chars_per_chunk: Maximum characters per chunk to handle token limitations
        """
        segments: List[torch.Tensor] = []
        total_segments = len(args["segments"])

        logger.info(f"Starting podcast creation with {total_segments} segments.")

        for i, segment in enumerate(args["segments"]):
            voice = segment.voice
            text = segment.text

            # Custom progress logging
            progress = (i + 1) / total_segments * 100
            logger.info(
                f"[{progress:.1f}%] Processing segment {i+1}/{total_segments} ({voice}): {text[:50]}..."
            )

            try:
                prompt_path = self._get_audio_prompt_path(voice)
                
                # Split text into chunks if it's too long
                text_chunks = _chunk_text(text, max_chars_per_chunk)
                
                if len(text_chunks) > 1:
                    logger.info(f"Segment {i+1} split into {len(text_chunks)} chunks")
                    
                    segment_audio_chunks = []
                    for j, chunk in enumerate(text_chunks):
                        logger.info(f"Processing segment {i+1}, chunk {j+1}/{len(text_chunks)}: {chunk[:50]}...")
                        chunk_wav = self.model.generate(
                            chunk,
                            audio_prompt_path=prompt_path,
                            repetition_penalty=segment.repetition_penalty,
                            min_p=segment.min_p,
                            top_p=segment.top_p,
                            exaggeration=segment.exaggeration,
                            cfg_weight=segment.cfg_weight,
                            temperature=segment.temperature,
                            top_k=segment.top_k,
                            norm_loudness=segment.norm_loudness,
                        )
                        segment_audio_chunks.append(chunk_wav)
                    
                    # Concatenate chunks for this segment
                    wav = torch.cat(segment_audio_chunks, dim=-1)
                else:
                    wav = self.model.generate(
                        text,
                        audio_prompt_path=prompt_path,
                        repetition_penalty=segment.repetition_penalty,
                        min_p=segment.min_p,
                        top_p=segment.top_p,
                        exaggeration=segment.exaggeration,
                        cfg_weight=segment.cfg_weight,
                        temperature=segment.temperature,
                        top_k=segment.top_k,
                        norm_loudness=segment.norm_loudness,
                    )
                
                segments.append(wav)
            except Exception as e:
                logger.error(f"Failed to generate segment {i+1}: {e}")
                raise e

        # Concatenate all segments into a single waveform
        if not segments:
            logger.warning("No segments generated.")
            return torch.empty(0)

        conversation = torch.cat(segments, dim=-1)
        logger.info("Podcast creation completed successfully.")
        return conversation
