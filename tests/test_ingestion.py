"""Unit tests for ingestion idempotency, deterministic IDs, and rollback handling."""

import pytest
from src.rag.chunker import create_chunker
from src.rag.cross_lingual import MultilingualQueryAdapter
from src.rag.embeddings import DeterministicFakeEmbeddingProvider
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import DocumentParserRegistry
from src.rag.ingestor import PdfIngestor, generate_deterministic_doc_id
from src.rag.models import IngestionState
from src.rag.repository import DocumentRepository


class FailingExtractor:
    def extract_pages(self, filename: str, file_bytes: bytes):
        raise ValueError("Corrupt PDF byte structure")


def test_deterministic_document_id():
    id1 = generate_deterministic_doc_id("Policy_Q4.pdf", b"pdf binary content 123")
    id2 = generate_deterministic_doc_id("Policy_Q4.pdf", b"pdf binary content 123")
    id3 = generate_deterministic_doc_id("Policy_Q4.pdf", b"different content")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("doc-policy_q4-")


def test_ingestion_idempotency_avoids_duplicate_chunks():
    es_mgr = ElasticsearchIndexManager(es_url="http://non-existent-es:9200")  # forces memory chunks
    repo = DocumentRepository(es_manager=es_mgr)
    extractor = DocumentParserRegistry()
    chunker = create_chunker("sliding_window", max_tokens=100)
    cross_lingual = MultilingualQueryAdapter()
    embedder = DeterministicFakeEmbeddingProvider(dimensions=32)

    ingestor = PdfIngestor(
        extractor=extractor,
        chunker=chunker,
        cross_lingual=cross_lingual,
        embedder=embedder,
        repository=repo,
    )

    doc_bytes = b"Page 1: Group Treasury maintains liquidity. Page 2: CFO approves foreign exchange contracts."
    # 1. First ingestion
    meta1 = ingestor.ingest("Treasury_Doc.txt", doc_bytes, tags=["finance"])
    assert meta1["status"] == IngestionState.INDEXED.value
    chunks_count_pass1 = len(repo.es_manager.memory_chunks)
    assert chunks_count_pass1 > 0

    # 2. Re-ingesting the exact same document must NOT duplicate chunks
    meta2 = ingestor.ingest("Treasury_Doc.txt", doc_bytes, tags=["finance"])
    assert meta2["status"] == IngestionState.INDEXED.value
    chunks_count_pass2 = len(repo.es_manager.memory_chunks)

    assert chunks_count_pass1 == chunks_count_pass2
    # Verify staged file is preserved after ingestion
    doc_id = meta2["doc_id"]
    file_path = repo.uploads_dir / f"{doc_id}_Treasury_Doc.txt"
    assert file_path.exists(), "Uploaded file must be preserved in uploads directory after indexing"
    # Clean up test artifact
    repo.delete_document(doc_id)
    assert not file_path.exists(), "delete_document must clean up uploaded file"


def test_ingestion_failure_triggers_clean_rollback():
    es_mgr = ElasticsearchIndexManager(es_url="http://non-existent-es:9200")
    repo = DocumentRepository(es_manager=es_mgr)
    failing_extractor = FailingExtractor()
    chunker = create_chunker("sliding_window")
    cross_lingual = MultilingualQueryAdapter()
    embedder = DeterministicFakeEmbeddingProvider(dimensions=32)

    ingestor = PdfIngestor(
        extractor=failing_extractor,
        chunker=chunker,
        cross_lingual=cross_lingual,
        embedder=embedder,
        repository=repo,
    )

    doc_bytes = b"corrupted bytes"
    with pytest.raises(ValueError, match="Corrupt PDF byte structure"):
        ingestor.ingest("corrupt.pdf", doc_bytes)

    # Verify no orphan chunks left in repository
    assert len(repo.es_manager.memory_chunks) == 0
    # State recorded as FAILED
    doc_id = generate_deterministic_doc_id("corrupt.pdf", doc_bytes)
    state = repo.get_document_state(doc_id)
    assert state is not None
    assert state.status == IngestionState.FAILED
