"""Hybrid Retrieval & Storage Adapter for Elasticsearch 8.15 (`BM25 + int8_hnsw`).

Implements `ChunkStorePort` with an automatic in-memory fallback when Elasticsearch
is offline (e.g. unit tests). All Elasticsearch-specific index mappings, slowlog
settings, Painless scripts, and pre-filtered kNN + BM25 queries live strictly here.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Final, List, Sequence

from elasticsearch import Elasticsearch

from src.config import ELASTICSEARCH_URL
from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.embeddings import EMBEDDING_DIMS, STOPWORDS, has_cjk
from src.rag.ports import HybridSearchOutcome

KB_INDEX_NAME: Final[str] = "hk-rbac-rag-kb"


class ElasticsearchHybridStore:
    """Elasticsearch 8.15 adapter implementing `ChunkStorePort` for pre-filtered hybrid RAG."""

    def __init__(
        self,
        es_url: str = ELASTICSEARCH_URL,
        index_name: str = KB_INDEX_NAME,
    ) -> None:
        self._es_url = es_url
        self.index_name = index_name
        self.es = Elasticsearch(self._es_url, request_timeout=5, max_retries=1)
        self._memory_chunks: List[Dict[str, Any]] = []
        self._ensure_index()

    @property
    def backend_url(self) -> str:
        return self._es_url

    def is_online(self) -> bool:
        try:
            return bool(self.es.ping())
        except Exception:
            return False

    def _ensure_index(self) -> None:
        if not self.is_online():
            return
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
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

    def index_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        if self.is_online():
            self._ensure_index()
            for rec in chunks:
                self.es.index(
                    index=self.index_name,
                    id=rec["chunk_id"],
                    document=rec,
                )
            self.es.indices.refresh(index=self.index_name)
        else:
            self._memory_chunks.extend(chunks)

    def list_all_chunk_metadata(self) -> List[Dict[str, Any]]:
        if self.is_online():
            self._ensure_index()
            resp = self.es.search(
                index=self.index_name,
                size=1000,
                _source=[
                    "doc_id",
                    "filename",
                    "page_number",
                    "is_public",
                    "acl_tags",
                    "uploaded_at",
                ],
                query={"match_all": {}},
            )
            return [h["_source"] for h in resp["hits"]["hits"]]
        return list(self._memory_chunks)

    def update_document_acl(
        self,
        doc_id: str,
        acl_tags: List[str],
        is_public: bool,
    ) -> None:
        if self.is_online():
            self._ensure_index()
            self.es.update_by_query(
                index=self.index_name,
                query={"term": {"doc_id": doc_id}},
                script={
                    "source": "ctx._source.acl_tags = params.acl_tags; ctx._source.is_public = params.is_public;",
                    "lang": "painless",
                    "params": {
                        "acl_tags": acl_tags,
                        "is_public": is_public,
                    },
                },
                refresh=True,
            )
        else:
            for rec in self._memory_chunks:
                if rec["doc_id"] == doc_id:
                    rec["acl_tags"] = acl_tags
                    rec["is_public"] = is_public

    def delete_document_chunks(self, doc_id: str) -> None:
        if self.is_online():
            self._ensure_index()
            self.es.delete_by_query(
                index=self.index_name,
                query={"term": {"doc_id": doc_id}},
                refresh=True,
            )
        self._memory_chunks = [
            c for c in self._memory_chunks if c.get("doc_id") != doc_id
        ]

    def reset_store(self) -> None:
        if self.is_online() and self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        self._memory_chunks.clear()
        self._ensure_index()

    def hybrid_search(
        self,
        expanded_query: str,
        query_vector: List[float],
        effective_acl_tags: List[str],
        top_k: int,
    ) -> HybridSearchOutcome:
        acl_filter_clause = {"terms": {"acl_tags": effective_acl_tags}}
        es_query_dsl = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": expanded_query, "boost": 2.0}}},
                        {"match": {"title": {"query": expanded_query, "boost": 1.2}}},
                    ],
                    "filter": [acl_filter_clause],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": "<256-dim Cross-Lingual EN<->ZH L2-normalized vector>",
                "k": top_k,
                "num_candidates": max(20, top_k * 4),
                "filter": acl_filter_clause,
            },
        }

        t0 = time.perf_counter()
        authorized_chunks: List[Dict[str, Any]] = []
        masked_restricted_docs: List[Dict[str, Any]] = []

        if self.is_online():
            self._ensure_index()
            resp = self.es.search(
                index=self.index_name,
                size=top_k,
                query={
                    "bool": {
                        "should": [
                            {"match": {"content": {"query": expanded_query, "boost": 2.0}}},
                            {"match": {"title": {"query": expanded_query, "boost": 1.2}}},
                        ],
                        "filter": [acl_filter_clause],
                    }
                },
                knn={
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": max(20, top_k * 4),
                    "filter": acl_filter_clause,
                },
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                raw_acl = src.get("acl_tags", [PUBLIC_SENTINEL_TAG])
                visible_tags = extract_visible_tags(raw_acl)
                authorized_chunks.append(
                    {
                        "chunk_id": src.get("chunk_id"),
                        "doc_id": src.get("doc_id"),
                        "filename": src.get("filename"),
                        "page_number": src.get("page_number", 1),
                        "content": src.get("content", ""),
                        "is_public": bool(src.get("is_public", len(visible_tags) == 0)),
                        "tags": visible_tags,
                        "score": round(float(hit.get("_score") or 0.0), 4),
                    }
                )

            domain_keywords = " ".join(
                w.strip("?.,!\"'()？，。！")
                for w in expanded_query.split()
                if len(w.strip("?.,!\"'()？，。！")) >= 3
                and w.lower().strip("?.,!\"'()？，。！") not in STOPWORDS
                and not has_cjk(w)
            )
            unfiltered_resp = self.es.search(
                index=self.index_name,
                size=10,
                _source=["doc_id", "filename", "page_number", "acl_tags", "is_public"],
                query={
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": domain_keywords or expanded_query,
                                        "minimum_should_match": "2<60%",
                                    }
                                }
                            }
                        ],
                        "must_not": [acl_filter_clause],
                    }
                },
            )
            seen_masked = set()
            for hit in unfiltered_resp["hits"]["hits"]:
                src = hit["_source"]
                key = (src.get("filename"), src.get("page_number"))
                if key in seen_masked:
                    continue
                seen_masked.add(key)
                masked_restricted_docs.append(
                    {
                        "filename": src.get("filename"),
                        "page_number": src.get("page_number", 1),
                        "required_any_of_tags": extract_visible_tags(src.get("acl_tags", [])),
                    }
                )
        else:
            for rec in self._memory_chunks:
                rec_tags = set(rec.get("acl_tags", [PUBLIC_SENTINEL_TAG]))
                if rec_tags.intersection(effective_acl_tags):
                    score = sum(a * b for a, b in zip(query_vector, rec["embedding"]))
                    visible_tags = extract_visible_tags(rec_tags)
                    authorized_chunks.append(
                        {
                            "chunk_id": rec["chunk_id"],
                            "doc_id": rec["doc_id"],
                            "filename": rec["filename"],
                            "page_number": rec["page_number"],
                            "content": rec["content"],
                            "is_public": bool(rec.get("is_public", len(visible_tags) == 0)),
                            "tags": visible_tags,
                            "score": round(float(score), 4),
                        }
                    )
            authorized_chunks.sort(key=lambda c: c["score"], reverse=True)
            authorized_chunks = authorized_chunks[:top_k]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return HybridSearchOutcome(
            authorized_chunks=authorized_chunks,
            masked_restricted_docs=masked_restricted_docs,
            latency_ms=elapsed_ms,
            query_dsl=es_query_dsl,
        )
