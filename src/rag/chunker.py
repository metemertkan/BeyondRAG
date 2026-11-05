"""Component 3: Pluggable Tokenizers and Chunking Strategies.

Provides token-aware, boundary-respecting chunking strategies:
  - TokenizerProtocol: Pluggable tokenizer abstraction (character, whitespace, tiktoken/heuristic, CJK-aware)
  - ChunkingConfig: Strongly typed, validated chunking configuration
  - SlidingWindowChunker: Sliding window respecting token/character limits and boundary heuristics
  - ParagraphChunker: Bundles logical paragraphs, falling back to windowing for oversized blocks
  - SentenceChunker: Bundles complete sentences, windowing oversized sentences
  - RecursiveChunker: Hierarchical recursive chunking (paragraphs -> sentences -> words)
  - SemanticChunker: Splits at topic shifts using sentence embeddings, plus a sliding overlap
  - AdaptiveChunker: Routes each text to semantic or cheap fixed-size chunking
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Sequence

from src.rag.multilingual import NO_SPACE_RE, MultilingualTokenizer, join_units, split_sentences

# Script ranges only (not a language pack). Excludes Arabic presentation forms (U+FE70+).
_CJK_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


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

    Splits ideographic (CJK Unified) codepoints as single tokens; latin words /
    punctuation otherwise. Script-range behaviour only — not pack-language aware.
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
    tokenizer: TokenizerProtocol = field(default_factory=MultilingualTokenizer)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative, got {self.overlap_tokens}")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be strictly less than max_tokens ({self.max_tokens})"
            )
        valid_strategies = {"sliding_window", "paragraph", "sentence", "recursive", "semantic", "adaptive"}
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

    def _units(self, text: str) -> List[tuple[str, bool]]:
        """Return (unit, has_space_before) pairs.

        Whitespace-delimited words stay whole. Runs from scripts written without
        spaces are segmented by the tokenizer, otherwise an entire Chinese document
        would count as one unsplittable "word" and bypass ``max_tokens``.
        """
        units: List[tuple[str, bool]] = []
        for word in text.split():
            if NO_SPACE_RE.search(word) and self.config.unit == ChunkingUnit.TOKENS:
                pieces = self.config.tokenizer.split_into_tokens(word)
            else:
                pieces = [word]
            for index, piece in enumerate(pieces):
                units.append((piece, index == 0))
        return units

    @staticmethod
    def _render(units: Sequence[tuple[str, bool]]) -> str:
        return "".join((" " if spaced and i else "") + unit for i, (unit, spaced) in enumerate(units)).strip()

    def _split_oversized(self, text: str) -> List[str]:
        """Strict fallback sliding window for any block that exceeds max_tokens."""
        units = self._units(text)
        if not units:
            return []

        chunks: List[str] = []
        current: List[tuple[str, bool]] = []
        current_len = 0

        for unit in units:
            u_len = self._measure(unit[0])
            separator = 1 if unit[1] else 0
            if current and (current_len + u_len + separator) > self.config.max_tokens:
                chunk_str = self._render(current)
                if chunk_str:
                    chunks.append(chunk_str)
                overlap: List[tuple[str, bool]] = []
                overlap_len = 0
                for prev in reversed(current):
                    p_len = self._measure(prev[0])
                    if overlap_len + p_len <= self.config.overlap_tokens:
                        overlap.insert(0, prev)
                        overlap_len += p_len + (1 if prev[1] else 0)
                    else:
                        break
                current = overlap + [unit]
                current_len = self._measure(self._render(current))
            else:
                current.append(unit)
                current_len += u_len + separator

        if current:
            chunk_str = self._render(current)
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
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for s in sentences:
            s_len = self._measure(s)
            if s_len > self.config.max_tokens:
                if current:
                    chunk_str = join_units(current).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    current = []
                    current_len = 0
                chunks.extend(self._split_oversized(s))
                continue

            if current and (current_len + s_len + 1) > self.config.max_tokens:
                chunk_str = join_units(current).strip()
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
                current_len = self._measure(join_units(current))
            else:
                current.append(s)
                current_len += s_len + 1

        if current:
            chunk_str = join_units(current).strip()
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


_EMBED_BATCH_SIZE: Final[int] = 16
# Below this, a "valley" is numerical noise in a single-topic section.
_MIN_VALLEY_DEPTH: Final[float] = 0.02


@dataclass
class SemanticChunkStats:
    """Telemetry from one semantic split: use it to profile ingestion cost on your stack."""

    sentences: int = 0
    embed_calls: int = 0
    embed_ms: float = 0.0
    similarities: List[float] = field(default_factory=list)
    depths: List[float] = field(default_factory=list)
    threshold: float = 0.0
    boundaries: List[int] = field(default_factory=list)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _valley_depths(similarities: Sequence[float]) -> List[float]:
    """For each gap, the climb to the nearest peak on each side, added together."""
    depths: List[float] = []
    for index, value in enumerate(similarities):
        left = value
        for neighbor in reversed(similarities[:index]):
            if neighbor < left:
                break
            left = neighbor
        right = value
        for neighbor in similarities[index + 1 :]:
            if neighbor < right:
                break
            right = neighbor
        depths.append((left - value) + (right - value))
    return depths


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    return [sum(values) / len(vectors) for values in zip(*vectors)]


class SemanticChunker(BaseChunker):
    """Splits where the topic shifts, with a sliding overlap across each boundary.

    Every sentence is embedded (one inference per sentence is the latency cost this
    strategy pays). A window of sentences on each side of a gap is compared, so a
    single odd sentence does not read as a topic change.

    Boundaries come from valley depth (the TextTiling idea): how far similarity drops
    below the peaks on either side of a gap. A low similarity alone is a weak signal,
    since topics differ in how internally similar they are and a global threshold
    misses a real shift inside a tight topic. A gap becomes a boundary when its depth
    is ``breakpoint_std`` standard deviations above the document's mean depth.

    ``min_chunk_ratio`` stops strict boundaries from fragmenting a long single-topic
    section, ``max_tokens`` still caps every chunk, and ``overlap_ratio`` carries the
    tail of each chunk (10-20% is the usual range) into the next.
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedder: Any = None,
        *,
        window: int = 2,
        breakpoint_std: float = 0.8,
        min_chunk_ratio: float = 0.2,
        overlap_ratio: float = 0.15,
    ) -> None:
        super().__init__(config)
        if embedder is None:
            raise ValueError("SemanticChunker requires an embedder (it compares sentence embeddings).")
        if not 0.0 <= overlap_ratio < 0.5:
            raise ValueError(f"overlap_ratio must be in [0, 0.5), got {overlap_ratio}")
        self._embedder = embedder
        self.window = max(1, window)
        self.breakpoint_std = breakpoint_std
        self.min_chunk_tokens = max(1, int(self.config.max_tokens * min_chunk_ratio))
        self.overlap_ratio = overlap_ratio
        self.last_stats = SemanticChunkStats()

    def _embed_all(self, sentences: List[str]) -> List[List[float]]:
        started = time.perf_counter()
        if hasattr(self._embedder, "embed_documents"):
            # Small batches: one request for a long document can exceed a provider timeout.
            vectors: List[List[float]] = []
            for start in range(0, len(sentences), _EMBED_BATCH_SIZE):
                vectors.extend(self._embedder.embed_documents(sentences[start : start + _EMBED_BATCH_SIZE]))
        else:
            vectors = [self._embedder.embed(sentence) for sentence in sentences]
        self.last_stats.embed_calls = len(sentences)
        self.last_stats.embed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return vectors

    def _gap_similarities(self, vectors: List[List[float]]) -> List[float]:
        """Similarity between the window before and the window after each sentence gap."""
        return [
            _cosine(
                _mean_vector(vectors[max(0, gap - self.window + 1) : gap + 1]),
                _mean_vector(vectors[gap + 1 : gap + 1 + self.window]),
            )
            for gap in range(len(vectors) - 1)
        ]

    def _overlap_tail(self, sentences: List[str]) -> List[str]:
        """Trailing sentences within the overlap budget, or a token tail of the last one."""
        budget = int(self._measure(join_units(sentences)) * self.overlap_ratio)
        if budget <= 0:
            return []
        tail: List[str] = []
        used = 0
        for sentence in reversed(sentences):
            cost = self._measure(sentence)
            if used + cost > budget:
                break
            tail.insert(0, sentence)
            used += cost
        if tail:
            return tail
        units = self._units(sentences[-1])
        kept: List[tuple[str, bool]] = []
        for unit in reversed(units):
            cost = self._measure(unit[0])
            if used + cost > budget:
                break
            kept.insert(0, unit)
            used += cost
        return [self._render(kept)] if kept else []

    def split(self, text: str) -> List[str]:
        self.last_stats = SemanticChunkStats()
        sentences = split_sentences(text)
        self.last_stats.sentences = len(sentences)
        if not sentences:
            return []

        # Sentences above the cap are windowed first so the cap holds for every unit.
        units: List[str] = []
        for sentence in sentences:
            units.extend(
                self._split_oversized(sentence) if self._measure(sentence) > self.config.max_tokens else [sentence]
            )
        if len(units) < 3:
            whole = join_units(units)
            return [whole] if self._measure(whole) <= self.config.max_tokens else units

        similarities = self._gap_similarities(self._embed_all(units))
        depths = _valley_depths(similarities)
        mean = sum(depths) / len(depths)
        std = math.sqrt(sum((value - mean) ** 2 for value in depths) / len(depths))
        threshold = max(mean + self.breakpoint_std * std, _MIN_VALLEY_DEPTH)
        self.last_stats.similarities = [round(value, 4) for value in similarities]
        self.last_stats.depths = [round(value, 4) for value in depths]
        self.last_stats.threshold = round(threshold, 4)

        groups: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for index, unit in enumerate(units):
            unit_len = self._measure(unit)
            # A hard cap always wins; a topic boundary only wins past the minimum size.
            if current and current_len + unit_len > self.config.max_tokens:
                groups.append(current)
                current, current_len = [], 0
            current.append(unit)
            current_len += unit_len
            is_topic_shift = 0 < index < len(depths) - 1 and depths[index] >= threshold
            if is_topic_shift and current_len >= self.min_chunk_tokens:
                self.last_stats.boundaries.append(index)
                groups.append(current)
                current, current_len = [], 0
        if current:
            groups.append(current)

        chunks = [join_units(groups[0])]
        for previous, group in zip(groups, groups[1:]):
            room = self.config.max_tokens - self._measure(join_units(group))
            tail = self._overlap_tail(previous) if room > 0 else []
            while tail and self._measure(join_units(tail)) > room:
                tail = tail[1:]
            chunks.append(join_units(tail + group))
        return chunks


_LIST_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:[-*•|]|\d+[.)]|[A-Za-z0-9_-]+\s*[:=|])")


class AdaptiveChunker(BaseChunker):
    """Routes each text: semantic where topics shift, fixed-size where they do not.

    Semantic chunking costs one embedding per sentence, so it is only used on prose
    or transcripts long enough to span topics. Short texts, lists, logs and other
    line-oriented uniform content take the cheap path. The routing itself makes no
    model calls, and ``last_decision`` records why each text went the way it did.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None, embedder: Any = None) -> None:
        super().__init__(config)
        self.semantic = SemanticChunker(self.config, embedder)
        self.fixed = SlidingWindowChunker(self.config)
        self.last_decision: Dict[str, str] = {}

    def route(self, text: str) -> Dict[str, str]:
        if self._measure(text) <= self.config.max_tokens:
            return {"strategy": "fixed", "reason": "fits in one chunk"}
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) >= 5:
            list_ratio = sum(1 for line in lines if _LIST_LINE_RE.match(line)) / len(lines)
            if list_ratio >= 0.6:
                return {"strategy": "fixed", "reason": f"uniform line structure ({list_ratio:.0%} list-like)"}
        if len(split_sentences(text)) < 6:
            return {"strategy": "fixed", "reason": "too few sentences to span topics"}
        return {"strategy": "semantic", "reason": "long free-form prose"}

    def split(self, text: str) -> List[str]:
        self.last_decision = self.route(text)
        chunker = self.semantic if self.last_decision["strategy"] == "semantic" else self.fixed
        return chunker.split(text)


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
    embedder: Any = None,
) -> BaseChunker:
    """Factory creating a validated pluggable chunker instance.

    ``embedder`` is required for the ``semantic`` and ``adaptive`` strategies.
    """
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

    if resolved_config.strategy == "semantic":
        return SemanticChunker(resolved_config, embedder)
    if resolved_config.strategy == "adaptive":
        return AdaptiveChunker(resolved_config, embedder)

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
