import logging
import os
import re
from typing import List

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

from .podcast import Podcast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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


class TextToSpeech:
    def __init__(self, device: str = "cpu"):
        """
        Initialize the TextToSpeech library.

        Args:
            device: Device to run the model on ("cpu" or "cuda").
        """
        logger.info(f"Initializing TextToSpeech on {device}...")
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

        # Determine the package directory where this file resides
        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        # Dynamically load voices from the sample directory inside the package
        self.voice_map = {}
        sample_dir = os.path.join(self.root_dir, "sample")

        if os.path.exists(sample_dir):
            logger.info(f"Scanning for voices in {sample_dir}...")
            for file in os.listdir(sample_dir):
                if file.lower().endswith(".wav"):
                    voice_name = os.path.splitext(file)[0]
                    # Store relative path to keep it consistent with root_dir usage in Podcast
                    self.voice_map[voice_name] = os.path.join("sample", file)
            logger.info(
                f"Found {len(self.voice_map)} voices: {list(self.voice_map.keys())}"
            )
        else:
            logger.warning(
                f"Sample directory not found at {sample_dir}. No voices loaded."
            )

        self.podcast = Podcast(self.model, self.voice_map, self.root_dir)
        logger.info("TextToSpeech initialized.")

    def list_voices(self) -> List[str]:
        """List all available voices."""
        return list(self.voice_map.keys())

    def convert(
        self,
        text: str,
        voice: str,
        repetition_penalty: float = 1.2,
        min_p: float = 0,
        top_p: float = 0.95,
        exaggeration: float = 0.5,
        cfg_weight: float = 0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
        max_chars_per_chunk: int = 500,
    ) -> torch.Tensor:
        """
        Convert a single text segment to speech with automatic chunking.
        
        Args:
            max_chars_per_chunk: Maximum characters per chunk to handle token limitations
        """
        logger.info(f"Converting text with voice '{voice}'...")
        path = self.podcast._get_audio_prompt_path(voice)
        
        # Split text into chunks if it's too long
        text_chunks = _chunk_text(text, max_chars_per_chunk)
        
        if len(text_chunks) > 1:
            logger.info(f"Text split into {len(text_chunks)} chunks to handle token limitations")
            
            audio_chunks = []
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                chunk_wav = self.model.generate(
                    chunk,
                    audio_prompt_path=path,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_k=top_k,
                    norm_loudness=norm_loudness,
                )
                audio_chunks.append(chunk_wav)
            
            # Concatenate all audio chunks
            wav = torch.cat(audio_chunks, dim=-1)
        else:
            wav = self.model.generate(
                text,
                audio_prompt_path=path,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_k=top_k,
                norm_loudness=norm_loudness,
            )
        
        return wav
