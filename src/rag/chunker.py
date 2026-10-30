"""Component 3: Pluggable Tokenizers and Chunking Strategies.

Provides token-aware, boundary-respecting chunking strategies:
  - TokenizerProtocol: Pluggable tokenizer abstraction (character, whitespace, tiktoken/heuristic, CJK-aware)
  - ChunkingConfig: Strongly typed, validated chunking configuration
  - SlidingWindowChunker: Sliding window respecting token/character limits and boundary heuristics
  - ParagraphChunker: Bundles logical paragraphs, falling back to windowing for oversized blocks
  - SentenceChunker: Bundles complete sentences, windowing oversized sentences
  - RecursiveChunker: Hierarchical recursive chunking (paragraphs -> sentences -> words)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Protocol

_SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?。！？])\s+")
_CJK_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\uffff]")


class ChunkingUnit(str, Enum):
    CHARACTERS = "characters"
    TOKENS = "tokens"


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers used in chunking calculations."""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...

    def split_into_tokens(self, text: str) -> List[str]:
        """Split text into token units or words/characters."""
        ...


class HeuristicTokenizer:
    """Fast, dependency-free multilingual tokenizer.
    
    Treats CJK characters as individual tokens and latin words / punctuation as tokens.
    Provides accurate token estimates for both Western and Eastern languages.
    """

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        tokens = self.split_into_tokens(text)
        return len(tokens)

    def split_into_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for segment in re.split(r"(\s+|[a-zA-Z0-9_-]+|[^\s\w])", text):
            if not segment:
                continue
            if _CJK_CHAR_RE.search(segment):
                for char in segment:
                    if not char.isspace():
                        tokens.append(char)
            elif not segment.isspace():
                tokens.append(segment)
        return tokens


class CharacterTokenizer:
    """Character-based tokenizer for exact character length enforcement."""

    def count_tokens(self, text: str) -> int:
        return len(text)

    def split_into_tokens(self, text: str) -> List[str]:
        return list(text)


@dataclass(frozen=True)
class ChunkingConfig:
    """Strongly typed, validated chunking configuration."""

    strategy: str = "sliding_window"
    unit: ChunkingUnit = ChunkingUnit.TOKENS
    max_tokens: int = 500
    overlap_tokens: int = 80
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    tokenizer: TokenizerProtocol = field(default_factory=HeuristicTokenizer)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative, got {self.overlap_tokens}")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be strictly less than max_tokens ({self.max_tokens})"
            )
        valid_strategies = {"sliding_window", "paragraph", "sentence", "recursive"}
        if self.strategy.lower() not in valid_strategies:
            raise ValueError(
                f"Unknown chunking strategy '{self.strategy}'. Valid strategies are: {sorted(valid_strategies)}"
            )


DEFAULT_CHUNKING_CONFIG: Final[ChunkingConfig] = ChunkingConfig(
    strategy=os.getenv("RAG_CHUNKING_STRATEGY", "sliding_window").strip().lower(),
    unit=ChunkingUnit(os.getenv("RAG_CHUNKING_UNIT", "tokens").strip().lower()),
    max_tokens=int(os.getenv("RAG_CHUNK_MAX_TOKENS", os.getenv("RAG_CHUNK_SIZE", "500"))),
    overlap_tokens=int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", os.getenv("RAG_CHUNK_OVERLAP", "80"))),
)


class BaseChunker:
    """Base class for all chunking strategies with token/character boundary enforcement."""

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or DEFAULT_CHUNKING_CONFIG

    def _measure(self, text: str) -> int:
        if self.config.unit == ChunkingUnit.CHARACTERS:
            return len(text)
        return self.config.tokenizer.count_tokens(text)

    def _split_oversized(self, text: str) -> List[str]:
        """Strict fallback sliding window for any block that exceeds max_tokens."""
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for word in words:
            w_len = self._measure(word)
            if current and (current_len + w_len + 1) > self.config.max_tokens:
                chunk_str = " ".join(current).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Apply overlap
                overlap_words: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    p_len = self._measure(prev)
                    if overlap_len + p_len <= self.config.overlap_tokens:
                        overlap_words.insert(0, prev)
                        overlap_len += p_len + 1
                    else:
                        break
                current = overlap_words + [word]
                current_len = self._measure(" ".join(current))
            else:
                current.append(word)
                current_len += w_len + 1

        if current:
            chunk_str = " ".join(current).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunker respecting sentence/word boundaries and token limits."""

    def split(self, text: str) -> List[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        if self._measure(normalized) <= self.config.max_tokens:
            return [normalized]

        return self._split_oversized(normalized)


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunker that bundles complete paragraphs up to max_tokens."""

    def split(self, text: str) -> List[str]:
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not raw_paras:
            raw_paras = [text.strip()] if text.strip() else []
        if not raw_paras:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for p in raw_paras:
            p_len = self._measure(p)
            # If a single paragraph is oversized, split it first
            if p_len > self.config.max_tokens:
                if current:
                    chunk_str = "\n\n".join(current).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    current = []
                    current_len = 0
                chunks.extend(self._split_oversized(p))
                continue

            if current and (current_len + p_len + 2) > self.config.max_tokens:
                chunk_str = "\n\n".join(current).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Overlap: keep last paragraph if it fits within overlap_tokens
                if self.config.overlap_tokens > 0 and current:
                    last_p = current[-1]
                    if self._measure(last_p) <= self.config.overlap_tokens:
                        current = [last_p, p]
                        current_len = self._measure("\n\n".join(current))
                    else:
                        current = [p]
                        current_len = p_len
                else:
                    current = [p]
                    current_len = p_len
            else:
                current.append(p)
                current_len += p_len + 2

        if current:
            chunk_str = "\n\n".join(current).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks


class SentenceChunker(BaseChunker):
    """Strict sentence-boundary chunker with oversized sentence handling."""

    def split(self, text: str) -> List[str]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for s in sentences:
            s_len = self._measure(s)
            if s_len > self.config.max_tokens:
                if current:
                    chunk_str = " ".join(current).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    current = []
                    current_len = 0
                chunks.extend(self._split_oversized(s))
                continue

            if current and (current_len + s_len + 1) > self.config.max_tokens:
                chunk_str = " ".join(current).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Overlap: keep trailing sentences within overlap_tokens
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    p_len = self._measure(prev)
                    if overlap_len + p_len <= self.config.overlap_tokens:
                        overlap_sentences.insert(0, prev)
                        overlap_len += p_len + 1
                    else:
                        break
                current = overlap_sentences + [s]
                current_len = self._measure(" ".join(current))
            else:
                current.append(s)
                current_len += s_len + 1

        if current:
            chunk_str = " ".join(current).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks


class RecursiveChunker(BaseChunker):
    """Hierarchical chunker that splits by paragraphs, then sentences, then tokens."""

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []
        para_chunker = ParagraphChunker(self.config)
        chunks: List[str] = []
        for p_chunk in para_chunker.split(text):
            if self._measure(p_chunk) <= self.config.max_tokens:
                chunks.append(p_chunk)
            else:
                sent_chunker = SentenceChunker(self.config)
                chunks.extend(sent_chunker.split(p_chunk))
        return chunks


CHUNKER_REGISTRY: Final[Dict[str, Callable[[ChunkingConfig], BaseChunker]]] = {
    "sliding_window": SlidingWindowChunker,
    "paragraph": ParagraphChunker,
    "sentence": SentenceChunker,
    "recursive": RecursiveChunker,
}


def create_chunker(
    strategy: Optional[str] = None,
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
    unit: Optional[ChunkingUnit] = None,
    tokenizer: Optional[TokenizerProtocol] = None,
    config: Optional[ChunkingConfig] = None,
) -> BaseChunker:
    """Factory creating a validated pluggable chunker instance."""
    if config is not None:
        resolved_config = config
    else:
        resolved_config = ChunkingConfig(
            strategy=(strategy or DEFAULT_CHUNKING_CONFIG.strategy).strip().lower(),
            unit=unit or DEFAULT_CHUNKING_CONFIG.unit,
            max_tokens=max_tokens or DEFAULT_CHUNKING_CONFIG.max_tokens,
            overlap_tokens=overlap_tokens if overlap_tokens is not None else DEFAULT_CHUNKING_CONFIG.overlap_tokens,
            tokenizer=tokenizer or DEFAULT_CHUNKING_CONFIG.tokenizer,
        )

    chunker_cls = CHUNKER_REGISTRY.get(resolved_config.strategy)
    if chunker_cls is None:
        raise ValueError(
            f"Unknown chunking strategy '{resolved_config.strategy}'. Valid options: {list(CHUNKER_REGISTRY.keys())}"
        )
    return chunker_cls(resolved_config)


# Backward-compatible helper
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    cfg = ChunkingConfig(
        strategy="sliding_window",
        unit=ChunkingUnit.CHARACTERS,
        max_tokens=chunk_size,
        overlap_tokens=overlap,
        tokenizer=CharacterTokenizer(),
    )
    return SlidingWindowChunker(cfg).split(text)


# Backward-compatible aliases
TextChunker = SlidingWindowChunker
