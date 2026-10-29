"""Component 3: Pure Domain Text Chunking.

Splits extracted page text into overlapping semantic paragraphs/chunks.
Contains zero external dependencies.
"""

from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 650, overlap: int = 100) -> List[str]:
    """Split page text into overlapping semantic paragraphs/chunks."""
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


class TextChunker:
    """Configurable domain text chunker."""

    def __init__(self, chunk_size: int = 650, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        return chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
