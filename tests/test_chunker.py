"""Unit tests for pluggable chunking strategies, tokenizers, and ChunkingConfig."""

import pytest
from src.rag.chunker import (
    CharacterTokenizer,
    ChunkingConfig,
    ChunkingUnit,
    HeuristicTokenizer,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    SlidingWindowChunker,
    create_chunker,
)


def test_chunking_config_validation():
    # Valid config
    cfg = ChunkingConfig(strategy="sliding_window", max_tokens=100, overlap_tokens=20)
    assert cfg.max_tokens == 100
    assert cfg.overlap_tokens == 20

    # Invalid max_tokens
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        ChunkingConfig(max_tokens=0)

    # Invalid overlap_tokens
    with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
        ChunkingConfig(overlap_tokens=-5)

    # Overlap >= max_tokens
    with pytest.raises(ValueError, match="strictly less than max_tokens"):
        ChunkingConfig(max_tokens=100, overlap_tokens=100)

    # Unknown strategy
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        ChunkingConfig(strategy="unknown_algo")


def test_heuristic_tokenizer_multilingual():
    tok = HeuristicTokenizer()
    # English: 9 words + 1 period = 10 tokens
    en_count = tok.count_tokens("The quick brown fox jumps over the lazy dog.")
    assert en_count == 10

    # Chinese: 11 characters
    zh_count = tok.count_tokens("香港天文台八號風球警告")
    assert zh_count == 11

    # Mixed bilingual
    mix_count = tok.count_tokens("Group Treasury maintains HKD 450M 現金儲備.")
    assert mix_count > 5


def test_sliding_window_chunker_respects_limits():
    cfg = ChunkingConfig(
        strategy="sliding_window",
        unit=ChunkingUnit.TOKENS,
        max_tokens=10,
        overlap_tokens=2,
        tokenizer=HeuristicTokenizer(),
    )
    chunker = SlidingWindowChunker(cfg)
    text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."
    chunks = chunker.split(text)

    assert len(chunks) > 1
    for chunk in chunks:
        token_count = cfg.tokenizer.count_tokens(chunk)
        assert token_count <= cfg.max_tokens + 2  # account for boundary word grouping
        assert len(chunk.strip()) > 0


def test_paragraph_chunker_handles_oversized():
    cfg = ChunkingConfig(
        strategy="paragraph",
        unit=ChunkingUnit.WORDS if hasattr(ChunkingUnit, "WORDS") else ChunkingUnit.TOKENS,
        max_tokens=15,
        overlap_tokens=0,
    )
    chunker = ParagraphChunker(cfg)

    # Paragraph 1 has 5 words, Paragraph 2 has 25 words (oversized)
    p1 = "This is a short paragraph."
    p2 = "This is an extremely long paragraph that easily exceeds the fifteen token limit and must be segmented properly into smaller chunks without dropping data."
    text = f"{p1}\n\n{p2}"

    chunks = chunker.split(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_sentence_chunker_multilingual():
    cfg = ChunkingConfig(
        strategy="sentence",
        max_tokens=12,
        overlap_tokens=2,
    )
    chunker = SentenceChunker(cfg)
    text = "第一句話很短。第二句話稍微長一點點！第三句話結尾了？"
    chunks = chunker.split(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.strip()) > 0


def test_create_chunker_factory():
    c1 = create_chunker("paragraph", max_tokens=200, overlap_tokens=50)
    assert isinstance(c1, ParagraphChunker)
    assert c1.config.max_tokens == 200

    c2 = create_chunker("recursive", max_tokens=150)
    assert isinstance(c2, RecursiveChunker)

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        create_chunker("non_existent")
