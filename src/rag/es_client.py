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
from src.rag.tenancy import DEFAULT_TENANT, normalize_tenant

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
        # "shared" (pool): one index, every query filtered by tenant.
        # "per_tenant" (silo): one index per tenant, and the same filter as a second barrier.
        self.per_tenant = self.tenant_strategy.tenant_mode == "per_tenant"
        self.base_index_name = index_name or self.tenant_strategy.prefix
        # Kept for callers that predate tenants: the default tenant's index.
        self.index_name = self.index_for(self.tenant_strategy.default_tenant or DEFAULT_TENANT)
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

    def index_for(self, tenant: str | None) -> str:
        """The one index a tenant's data lives in. Always a concrete name, never a pattern."""
        if not self.per_tenant:
            return self.base_index_name
        return f"{self.base_index_name}-{normalize_tenant(tenant)}"

    def tenant_indices(self) -> List[str]:
        """Every index this manager owns. For operator tasks only, never for a search.

        In silo mode the name pattern alone is not trusted: an index counts only if this
        code created it for the same base name (recorded in the mapping's ``_meta``).
        """
        if not self.is_online():
            return []
        if not self.per_tenant:
            return [self.base_index_name] if self.client.indices.exists(index=self.base_index_name) else []
        found = self.client.indices.get(index=f"{self.base_index_name}-*", allow_no_indices=True)
        return sorted(
            name
            for name, body in found.items()
            if (body.get("mappings", {}).get("_meta") or {}).get("base_index") == self.base_index_name
        )

    def drop_tenant_index(self, tenant: str) -> bool:
        """Silo offboarding: remove a tenant's whole index. Refused in the pool, where it is shared."""
        if not self.per_tenant:
            raise ValueError("drop_tenant_index needs RAG_TENANT_MODE=per_tenant. The shared index holds every tenant.")
        scope = normalize_tenant(tenant)
        self.memory_chunks = [c for c in self.memory_chunks if normalize_tenant(c.get("tenant")) != scope]
        name = self.index_for(scope)
        if self.is_online() and self.client.indices.exists(index=name):
            self.client.indices.delete(index=name)
            return True
        return False

    def ensure_index(self, tenant: str | None = None) -> str:
        """Create the tenant's hybrid search index (`int8_hnsw` + slowlog) if missing. Returns its name."""
        index_name = self.index_for(tenant)
        if not self.is_online():
            return index_name
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(
                index=index_name,
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
                    "_meta": {
                        "base_index": self.base_index_name,
                        "tenant": normalize_tenant(tenant) if self.per_tenant else "*",
                    },
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
        return index_name

    def reset_index(self) -> None:
        """Delete every tenant's index and recreate the default one (and clear in-memory chunks).

        Indices are deleted one by one, by name. Elasticsearch 8 refuses wildcard deletes
        by default, and a wildcard is the wrong tool next to other people's indices anyway.
        """
        for name in self.tenant_indices():
            self.client.indices.delete(index=name)
        self.memory_chunks.clear()
        self.ensure_index()
