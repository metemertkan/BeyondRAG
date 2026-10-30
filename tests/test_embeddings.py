"""Unit tests for embedding provider factory and manifests."""

import pytest
from src.rag.embeddings import (
    DeterministicFakeEmbeddingProvider,
    EmbeddingManifest,
    FeatureHashEmbeddingProvider,
    create_embedding_provider,
    normalize_l2,
)


def test_embedding_manifest_validation():
    manifest = EmbeddingManifest(
        provider="test",
        model_name="test-model",
        dimensions=128,
    )
    assert manifest.dimensions == 128
    assert "test::test-model::128" in manifest.config_identifier

    with pytest.raises(ValueError, match="dimensions must be positive"):
        EmbeddingManifest(provider="test", model_name="bad", dimensions=0)


def test_feature_hash_provider_deterministic():
    provider = FeatureHashEmbeddingProvider(dimensions=128)
    assert provider.dimensions == 128
    v1 = provider.embed_query("Group Treasury Liquidity")
    v2 = provider.embed_query("Group Treasury Liquidity")
    assert v1 == v2
    assert len(v1) == 128

    # Batch embedding
    batch = provider.embed_documents(["First doc", "Second doc"])
    assert len(batch) == 2
    assert len(batch[0]) == 128
    assert len(batch[1]) == 128


def test_fake_provider_batch():
    fake = DeterministicFakeEmbeddingProvider(dimensions=64)
    assert fake.dimensions == 64
    docs = ["Doc A", "Doc B", "Doc C"]
    vecs = fake.embed_documents(docs)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 64


def test_provider_factory_validation():
    p1 = create_embedding_provider("feature_hash", dimensions=256)
    assert isinstance(p1, FeatureHashEmbeddingProvider)
    assert p1.dimensions == 256

    p2 = create_embedding_provider("fake", dimensions=32)
    assert isinstance(p2, DeterministicFakeEmbeddingProvider)
    assert p2.dimensions == 32

    # Incompatible provider name must fail loudly
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_provider("unsupported_provider_xyz")
