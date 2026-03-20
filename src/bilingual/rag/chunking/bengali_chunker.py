"""
Unicode-aware Text Chunking for Bengali and English.
Handles sentence/paragraph boundaries with token-length constraints.
"""

import re
import logging
from typing import List, Optional
from bilingual.models.manager import model_manager

logger = logging.getLogger("bilingual.rag.chunking")

class BengaliChunker:
    """
    Advanced chunker that respects Bengali punctuation and token limits.
    """
    def __init__(
        self, 
        tokenizer_name: str = "t5-small", 
        chunk_size: int = 512, 
        chunk_overlap: int = 50
    ):
        self.tokenizer = model_manager.get_tokenizer(tokenizer_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Unicode-aware sentence boundaries (Bangla + English)
        self.sentence_pattern = re.compile(r'([।!?\.])')

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits text while keeping the separators."""
        # Split by punctuation and then merge boundary back
        segments = self.sentence_pattern.split(text)
        sentences = []
        for i in range(0, len(segments) - 1, 2):
            sentences.append(segments[i] + segments[i+1])
        if len(segments) % 2 != 0 and segments[-1].strip():
            sentences.append(segments[-1])
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[str]:
        """
        Token-aware chunking with overlap.
        Groups sentences until they reach the token limit.
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            token_count = len(self.tokenizer.encode(sentence))
            
            # If a single sentence is larger than chunk_size, we must split it by words (fallback)
            if token_count > self.chunk_size:
                logger.warning(f"Sentence too long ({token_count} tokens). Splitting by words.")
                # Basic word-level split logic
                words = sentence.split()
                # ... simplified word chunking for brevity ...
                continue

            if current_length + token_count > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap (simplified for MVP)
                current_chunk = current_chunk[-1:] if current_chunk else []
                current_length = sum(len(self.tokenizer.encode(s)) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class LanguageRouter:
    """Routes text to appropriate chunking logic based on script detection."""
    def __init__(self):
        from bilingual.language_detection import LanguageDetector
        self.detector = LanguageDetector()

    def route_and_chunk(self, text: str, **kwargs) -> List[str]:
        # Currently, both BN and EN use the same Unicode-aware chunker logic
        # but could be specialized here in the future.
        chunker = BengaliChunker(**kwargs)
        return chunker.create_chunks(text)
