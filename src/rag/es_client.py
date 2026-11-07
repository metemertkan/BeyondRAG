"""Component 1: Elasticsearch Connection & Index Lifecycle Management.

Handles exclusively the `Elasticsearch` client connection, health ping, `int8_hnsw`
index creation, slowlog configuration, and index resets, plus shared in-memory state
when running offline in unit tests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Final, List

from elasticsearch import Elasticsearch

from src.config import ELASTICSEARCH_URL
from src.rag.embeddings import EMBEDDING_DIMS
from src.rag.profiles import DEFAULT_INDEX_PREFIX, IndexTenantStrategy

KB_INDEX_NAME: Final[str] = DEFAULT_INDEX_PREFIX
VECTOR_INDEX_TYPES: Final[tuple[str, ...]] = ("int8_hnsw", "int4_hnsw", "hnsw")
DEFAULT_VECTOR_INDEX_TYPE: Final[str] = os.getenv("RAG_VECTOR_INDEX_TYPE", "int8_hnsw").strip().lower()

# Keeps "ACME-X200" and "ERR-1042" whole. Splits on whitespace and sentence punctuation only,
# so hyphens, slashes, dots and underscores inside a code survive. "." still splits a sentence
# end through the filter below.
CODE_PRESERVING_ANALYSIS: Final[Dict[str, Any]] = {
    "tokenizer": {
        "code_tokenizer": {
            "type": "char_group",
            "tokenize_on_chars": ["whitespace", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'", "`", "|", "=", "<", ">"],
        }
    },
    "filter": {
        "trim_code_edges": {"type": "pattern_replace", "pattern": "^[-./_]+|[-./_]+$", "replacement": ""},
        "drop_empty_tokens": {"type": "length", "min": 1},
    },
    "analyzer": {
        "code_preserving": {
            "type": "custom",
            "tokenizer": "code_tokenizer",
            "filter": ["cjk_width", "lowercase", "trim_code_edges", "drop_empty_tokens"],
        }
    },
}


class ElasticsearchIndexManager:
    """Manages Elasticsearch 8.15 connection, health checks, and configurable index/tenant schema lifecycle."""

    def __init__(
        self,
        es_url: str = ELASTICSEARCH_URL,
        index_name: str | None = None,
        tenant_strategy: IndexTenantStrategy | None = None,
        dimensions: int = EMBEDDING_DIMS,
        vector_index_type: str = DEFAULT_VECTOR_INDEX_TYPE,
    ) -> None:
        if vector_index_type not in VECTOR_INDEX_TYPES:
            raise ValueError(f"Unknown vector index type {vector_index_type!r}. Valid: {VECTOR_INDEX_TYPES}")
        self.vector_index_type = vector_index_type
        self._es_url = es_url
        self.tenant_strategy = tenant_strategy or IndexTenantStrategy()
        self.index_name = index_name or self.tenant_strategy.resolve_index_name()
        self.dimensions = dimensions
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
                    "analysis": {
                        "tokenizer": CODE_PRESERVING_ANALYSIS["tokenizer"],
                        "filter": CODE_PRESERVING_ANALYSIS["filter"],
                        "analyzer": {
                            "code_preserving": CODE_PRESERVING_ANALYSIS["analyzer"]["code_preserving"],
                            # Built-in filters only, no plugins: full-width folding, Arabic
                            # diacritic and letter-form normalization, Arabic-Indic digits
                            # to ASCII, and Han bigrams instead of isolated characters.
                            "multilingual": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "cjk_width",
                                    "lowercase",
                                    "arabic_normalization",
                                    "decimal_digit",
                                    "cjk_bigram",
                                ],
                            }
                        }
                    },
                },
                mappings={
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "multilingual"},
                        "page_number": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "content": {
                            "type": "text",
                            "analyzer": "multilingual",
                            "fields": {"codes": {"type": "text", "analyzer": "code_preserving"}},
                        },
                        # Readable context for parent-child retrieval: stored, never searched.
                        "parent_id": {"type": "keyword"},
                        "parent_content": {"type": "text", "index": False},
                        "is_public": {"type": "boolean"},
                        "acl_tags": {"type": "keyword"},
                        "uploaded_at": {"type": "keyword"},
                        # Episode 2 versioning / structure fields (explicit so ES filters work).
                        "source_key": {"type": "keyword"},
                        "content_hash": {"type": "keyword"},
                        "doc_version": {"type": "integer"},
                        "tenant": {"type": "keyword"},
                        "is_current": {"type": "boolean"},
                        "ingestion_status": {"type": "keyword"},
                        "block_type": {"type": "keyword"},
                        "embedding_manifest": {"type": "keyword"},
                        "parser_name": {"type": "keyword"},
                        # Governance and lineage (trust tiers, expiry, GDPR erasure).
                        "trust_tier": {"type": "keyword"},
                        "effective_date": {"type": "date"},
                        "expires_at": {"type": "date"},
                        "business_domain": {"type": "keyword"},
                        "source_system": {"type": "keyword"},
                        "subject_ids": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.dimensions,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": self.vector_index_type,
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
