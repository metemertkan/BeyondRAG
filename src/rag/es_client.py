"""Component 1: Elasticsearch Connection & Index Lifecycle Management.

Handles exclusively the `Elasticsearch` client connection, health ping, `int8_hnsw`
index creation, slowlog configuration, and index resets, plus shared in-memory state
when running offline in unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, Final, List

from elasticsearch import Elasticsearch

from src.config import ELASTICSEARCH_URL
from src.rag.embeddings import EMBEDDING_DIMS

KB_INDEX_NAME: Final[str] = "hk-rbac-rag-kb"


class ElasticsearchIndexManager:
    """Manages Elasticsearch 8.15 connection, health checks, and index schema lifecycle."""

    def __init__(
        self,
        es_url: str = ELASTICSEARCH_URL,
        index_name: str = KB_INDEX_NAME,
    ) -> None:
        self._es_url = es_url
        self.index_name = index_name
        self.client = Elasticsearch(self._es_url, request_timeout=5, max_retries=1)
        self.memory_chunks: List[Dict[str, Any]] = []
        self.ensure_index()

    @property
    def es_url(self) -> str:
        return self._es_url

    def is_online(self) -> bool:
        """Return True if the Elasticsearch cluster responds to ping."""
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def ensure_index(self) -> None:
        """Create the RBAC hybrid search index (`int8_hnsw` + slowlog) if it does not exist."""
        if not self.is_online():
            return
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                settings={
                    "index.search.slowlog.threshold.query.warn": "0ms",
                    "index.search.slowlog.threshold.fetch.warn": "0ms",
                },
                mappings={
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "title": {"type": "text"},
                        "page_number": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "content": {"type": "text"},
                        "is_public": {"type": "boolean"},
                        "acl_tags": {"type": "keyword"},
                        "uploaded_at": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIMS,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "int8_hnsw",
                                "m": 16,
                                "ef_construction": 100,
                            },
                        },
                    }
                },
            )

    def reset_index(self) -> None:
        """Delete and recreate the index (and clear in-memory fallback chunks)."""
        if self.is_online() and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self.memory_chunks.clear()
        self.ensure_index()
