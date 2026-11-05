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
from src.rag.governance import apply_trust_ranking, is_expired
from src.rag.ports import HybridSearchOutcome
from src.rag.profiles import RetrievalProfile, get_retrieval_profile


# Fetch extra candidates so trust ranking can promote a chunk from outside the top-k.
_RERANK_OVERFETCH = 3
_GOVERNANCE_FIELDS = ("trust_tier", "effective_date", "expires_at", "business_domain", "subject_ids")


class HybridRetriever:
    """Executes pre-filtered hybrid (`BM25 + int8_hnsw`) retrieval and RBAC mask auditing."""

    def __init__(self, es_manager: ElasticsearchIndexManager) -> None:
        self.es_manager = es_manager

    def search(
        self,
        expanded_query: str,
        query_vector: List[float],
        effective_acl_tags: List[str],
        top_k: int = 4,
        profile_name: str | None = None,
        resolved_config: RetrievalProfile | None = None,
    ) -> HybridSearchOutcome:
        profile: RetrievalProfile = resolved_config or get_retrieval_profile(profile_name, top_k_override=top_k)
        effective_top_k = profile.top_k
        acl_filter_clause = {"terms": {"acl_tags": effective_acl_tags}}
        es_query_dsl = {
            "size": effective_top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "content": {
                                    "query": expanded_query,
                                    "boost": profile.content_boost,
                                    "minimum_should_match": profile.min_should_match,
                                }
                            }
                        },
                        {
                            "match": {
                                "title": {
                                    "query": expanded_query,
                                    "boost": profile.title_boost,
                                }
                            }
                        },
                    ],
                    "filter": [acl_filter_clause],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": f"<{len(query_vector)}-dim Dense Vector>",
                "k": effective_top_k,
                "num_candidates": max(20, effective_top_k * profile.num_candidates_multiplier),
                "filter": acl_filter_clause,
            },
        }

        t0 = time.perf_counter()
        authorized_chunks: List[Dict[str, Any]] = []
        masked_restricted_docs: List[Dict[str, Any]] = []

        # Exclude superseded, archived and expired content from retrieval.
        fetch_size = effective_top_k * _RERANK_OVERFETCH
        current_only_filter = {
            "bool": {
                "must_not": [
                    {"term": {"is_current": False}},
                    {"term": {"ingestion_status": "SUPERSEDED"}},
                    {"term": {"ingestion_status": "ARCHIVED"}},
                    {"range": {"expires_at": {"lt": "now/d"}}},
                ]
            }
        }

        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            resp = self.es_manager.client.search(
                index=self.es_manager.index_name,
                size=fetch_size,
                query={
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": expanded_query,
                                        "boost": profile.content_boost,
                                        "minimum_should_match": profile.min_should_match,
                                    }
                                }
                            },
                            {
                                "match": {
                                    "title": {
                                        "query": expanded_query,
                                        "boost": profile.title_boost,
                                    }
                                }
                            },
                        ],
                        "filter": [acl_filter_clause, current_only_filter],
                    }
                },
                knn={
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": fetch_size,
                    "num_candidates": max(20, fetch_size, effective_top_k * profile.num_candidates_multiplier),
                    "filter": [acl_filter_clause, current_only_filter],
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
                        "block_type": src.get("block_type"),
                        "source_key": src.get("source_key"),
                        "doc_version": src.get("doc_version"),
                        "ingestion_status": src.get("ingestion_status"),
                        "metadata": src.get("metadata") or {},
                        **{name: src.get(name) for name in _GOVERNANCE_FIELDS},
                    }
                )
            authorized_chunks = apply_trust_ranking(authorized_chunks)[:effective_top_k]

            domain_keywords = " ".join(
                w.strip("?.,!\"'()？，。！")
                for w in expanded_query.split()
                if len(w.strip("?.,!\"'()？，。！")) >= 3
                and w.lower().strip("?.,!\"'()？，。！") not in STOPWORDS
                and not has_cjk(w)
            )
            unfiltered_resp = self.es_manager.client.search(
                index=self.es_manager.index_name,
                size=profile.audit_masked_size,
                _source=["doc_id", "filename", "page_number", "acl_tags", "is_public"],
                query={
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": domain_keywords or expanded_query,
                                        "minimum_should_match": profile.min_should_match,
                                    }
                                }
                            }
                        ],
                        # The audit must not reveal retired content either.
                        "must_not": [acl_filter_clause],
                        "filter": [current_only_filter],
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
                # Soft-superseded versions stay in storage but are excluded from retrieval.
                if rec.get("is_current") is False:
                    continue
                if rec.get("ingestion_status") in {"SUPERSEDED", "ARCHIVED"}:
                    continue
                if is_expired(rec.get("expires_at")):
                    continue
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
                            "block_type": rec.get("block_type"),
                            "source_key": rec.get("source_key"),
                            "doc_version": rec.get("doc_version"),
                            "metadata": rec.get("metadata") or {},
                            **{name: rec.get(name) for name in _GOVERNANCE_FIELDS},
                        }
                    )
            authorized_chunks = apply_trust_ranking(authorized_chunks)[:top_k]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return HybridSearchOutcome(
            authorized_chunks=authorized_chunks,
            masked_restricted_docs=masked_restricted_docs,
            latency_ms=elapsed_ms,
            query_dsl=es_query_dsl,
        )
