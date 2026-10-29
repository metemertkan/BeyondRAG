"""Component 6: Hybrid Retrieval (`BM25 + int8_hnsw`) & RBAC Mask Auditing.

Handles exclusively executing pre-filtered hybrid queries against Elasticsearch 8.15
(or in-memory fallback) and auditing restricted documents masked by the RBAC filter.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.embeddings import STOPWORDS, has_cjk
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.ports import HybridSearchOutcome


class HybridRetriever:
    """Executes pre-filtered hybrid (`BM25 + int8_hnsw`) retrieval and RBAC mask auditing."""

    def __init__(self, es_manager: ElasticsearchIndexManager) -> None:
        self.es_manager = es_manager

    def search(
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

        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            resp = self.es_manager.client.search(
                index=self.es_manager.index_name,
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
            unfiltered_resp = self.es_manager.client.search(
                index=self.es_manager.index_name,
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
            for rec in self.es_manager.memory_chunks:
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
