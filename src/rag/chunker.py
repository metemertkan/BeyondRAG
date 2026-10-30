"""Component 3: Pluggable Chunking Strategies.

Provides pluggable chunking strategies:
  - SlidingWindowChunker: Sliding window with sentence/word boundary detection
  - ParagraphChunker: Preserves logical paragraph blocks up to max chunk size
  - SentenceChunker: Strict sentence-boundary chunking
All strategies are configurable via parameters or environment variables.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, Final, List, Optional

DEFAULT_CHUNK_SIZE: Final[int] = int(os.getenv("RAG_CHUNK_SIZE", "650"))
DEFAULT_CHUNK_OVERLAP: Final[int] = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Default sliding-window chunker with sentence and whitespace boundary fallback."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_pos = normalized.rfind(". ", start, end)
            if split_pos == -1 or split_pos <= start + chunk_size // 2:
                split_pos = normalized.rfind(" ", start, end)
            if split_pos != -1 and split_pos > start:
                end = split_pos + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


class SlidingWindowChunker:
    """Sliding window chunking with soft boundary preservation."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        return chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)


class ParagraphChunker:
    """Paragraph-based chunker that bundles complete paragraphs up to chunk_size."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not raw_paras:
            return chunk_text(text, self.chunk_size, self.overlap)

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for p in raw_paras:
            p_len = len(p)
            if current and (current_len + p_len + 2) > self.chunk_size:
                chunks.append("\n\n".join(current))
                current = [p]
                current_len = p_len
            else:
                current.append(p)
                current_len += p_len + 2

        if current:
            chunks.append("\n\n".join(current))
        return chunks


class SentenceChunker:
    """Strict sentence-boundary chunker."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return chunk_text(text, self.chunk_size, self.overlap)

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for s in sentences:
            s_len = len(s)
            if current_sentences and (current_len + s_len + 1) > self.chunk_size:
                chunks.append(" ".join(current_sentences))
                current_sentences = [s]
                current_len = s_len
            else:
                current_sentences.append(s)
                current_len += s_len + 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))
        return chunks


# Alias for backward compatibility
TextChunker = SlidingWindowChunker

CHUNKER_REGISTRY: Final[Dict[str, Callable[[int, int], Any]]] = {
    "sliding_window": SlidingWindowChunker,
    "paragraph": ParagraphChunker,
    "sentence": SentenceChunker,
}


def create_chunker(
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Any:
    """Factory creating a pluggable chunker instance."""
    strat = (strategy or os.getenv("RAG_CHUNKING_STRATEGY", "sliding_window")).strip().lower()
    c_size = chunk_size or DEFAULT_CHUNK_SIZE
    c_overlap = overlap or DEFAULT_CHUNK_OVERLAP
    chunker_cls = CHUNKER_REGISTRY.get(strat, SlidingWindowChunker)
    return chunker_cls(c_size, c_overlap)
