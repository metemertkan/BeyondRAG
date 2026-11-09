"""Component 6: Hybrid Retrieval (`BM25 + int8_hnsw`) & RBAC Mask Auditing.

Runs lexical and dense searches against Elasticsearch 8.15 (or the in-memory fallback), both
pre-filtered by ACL and document lifecycle, merges them with Reciprocal Rank Fusion, and audits
restricted documents masked by the RBAC filter.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Final, List, Optional, Sequence

from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.embeddings import STOPWORDS, has_cjk
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.governance import apply_trust_ranking, is_expired
from src.rag.ports import HybridSearchOutcome
from src.rag.profiles import RetrievalProfile, get_retrieval_profile

# Fetch extra candidates so trust ranking can promote a chunk from outside the top-k.
_RERANK_OVERFETCH = 3
_GOVERNANCE_FIELDS = ("trust_tier", "effective_date", "expires_at", "business_domain", "subject_ids")
_CHUNK_FIELDS = ("block_type", "source_key", "doc_version", "ingestion_status", "parent_id", "parent_content")
# An exact code outranks an ordinary word match, but stays one lexical signal among others.
_CODE_FIELD_BOOST: Final[float] = 3.0
_MEMORY_TOKEN_RE = re.compile(r"[\w][\w./-]*[\w]|[\w]")

FUSION_MODES: Final[tuple[str, ...]] = ("rrf", "linear", "dense", "lexical")


def _memory_tokens(text: str) -> set[str]:
    return {token.lower() for token in _MEMORY_TOKEN_RE.findall(text)}


class HybridRetriever:
    """Executes pre-filtered hybrid (`BM25 + int8_hnsw`) retrieval and RBAC mask auditing."""

    def __init__(self, es_manager: ElasticsearchIndexManager, fusion: str = "rrf") -> None:
        if fusion not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {fusion!r}. Valid: {FUSION_MODES}")
        self.es_manager = es_manager
        self.fusion = fusion

    def _chunk_from_source(self, src: Dict[str, Any], score: float) -> Dict[str, Any]:
        raw_acl = src.get("acl_tags", [PUBLIC_SENTINEL_TAG])
        visible_tags = extract_visible_tags(raw_acl)
        return {
            "chunk_id": src.get("chunk_id"),
            "doc_id": src.get("doc_id"),
            "filename": src.get("filename"),
            "page_number": src.get("page_number", 1),
            "content": src.get("content", ""),
            "is_public": bool(src.get("is_public", len(visible_tags) == 0)),
            "tags": visible_tags,
            "score": round(float(score or 0.0), 6),
            "metadata": src.get("metadata") or {},
            **{name: src.get(name) for name in _CHUNK_FIELDS},
            **{name: src.get(name) for name in _GOVERNANCE_FIELDS},
        }

    def _lexical_query(self, text: str, profile: RetrievalProfile, filters: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": {
                                "query": text,
                                "boost": profile.content_boost,
                                "minimum_should_match": profile.min_should_match,
                            }
                        }
                    },
                    {"match": {"title": {"query": text, "boost": profile.title_boost}}},
                    # Whole hyphenated codes live here, so "ACME-X200" can't match "ACME-X100".
                    {"match": {"content.codes": {"query": text, "boost": _CODE_FIELD_BOOST}}},
                ],
                # With a filter present, should clauses are optional by default, and RRF would
                # rank every authorized document as a lexical hit.
                "minimum_should_match": 1,
                "filter": list(filters),
            }
        }

    def _es_candidates(
        self,
        mode: str,
        lexical_text: str,
        query_vector: Optional[List[float]],
        profile: RetrievalProfile,
        filters: Sequence[Dict[str, Any]],
        fetch_size: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        client, index = self.es_manager.client, self.es_manager.index_name
        knn = None
        if query_vector is not None:
            knn = {
                "field": "embedding",
                "query_vector": query_vector,
                "k": fetch_size,
                "num_candidates": max(20, fetch_size, profile.top_k * profile.num_candidates_multiplier),
                "filter": list(filters),
            }
        query = self._lexical_query(lexical_text, profile, filters)

        def hits_of(resp: Any) -> List[Dict[str, Any]]:
            return [self._chunk_from_source(h["_source"], h.get("_score")) for h in resp["hits"]["hits"]]

        if mode == "linear":
            # The earlier behavior, kept to compare against: BM25 and cosine scores are added.
            return hits_of(client.search(index=index, size=fetch_size, query=query, knn=knn)), {}
        lexical: List[Dict[str, Any]] = []
        dense: List[Dict[str, Any]] = []
        if mode in {"rrf", "lexical"}:
            lexical = hits_of(client.search(index=index, size=fetch_size, query=query))
        if mode in {"rrf", "dense"} and knn is not None:
            dense = hits_of(client.search(index=index, size=fetch_size, knn=knn))
        rankings = {"lexical": lexical, "dense": dense}
        if mode == "rrf":
            return reciprocal_rank_fusion([lexical, dense]), rankings
        return (lexical if mode == "lexical" else dense), rankings

    def _memory_candidates(
        self,
        mode: str,
        lexical_text: str,
        query_vector: Optional[List[float]],
        effective_acl_tags: Sequence[str],
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        query_tokens = _memory_tokens(lexical_text)
        dense: List[Dict[str, Any]] = []
        lexical: List[Dict[str, Any]] = []
        for rec in self.es_manager.memory_chunks:
            # Soft-superseded versions stay in storage but are excluded from retrieval.
            if rec.get("is_current") is False or rec.get("ingestion_status") in {"SUPERSEDED", "ARCHIVED"}:
                continue
            if is_expired(rec.get("expires_at")):
                continue
            if not set(rec.get("acl_tags", [PUBLIC_SENTINEL_TAG])).intersection(effective_acl_tags):
                continue
            if query_vector is not None and rec.get("embedding"):
                dense.append(self._chunk_from_source(rec, sum(a * b for a, b in zip(query_vector, rec["embedding"]))))
            overlap = len(query_tokens & _memory_tokens(rec.get("content", "")))
            if overlap:
                lexical.append(self._chunk_from_source(rec, float(overlap)))
        dense.sort(key=lambda c: c["score"], reverse=True)
        lexical.sort(key=lambda c: c["score"], reverse=True)
        rankings = {"lexical": lexical, "dense": dense}
        if mode == "lexical":
            return lexical, rankings
        if mode == "dense" or not lexical:
            return dense, rankings
        return reciprocal_rank_fusion([lexical, dense]), rankings

    def search(
        self,
        expanded_query: str,
        query_vector: Optional[List[float]],
        effective_acl_tags: List[str],
        top_k: int = 4,
        profile_name: str | None = None,
        resolved_config: RetrievalProfile | None = None,
        fusion: str | None = None,
        candidate_k: int | None = None,
    ) -> HybridSearchOutcome:
        """Return up to ``candidate_k or top_k`` chunks, trust-ranked, with the ACL pushed into every search."""
        profile: RetrievalProfile = resolved_config or get_retrieval_profile(profile_name, top_k_override=top_k)
        mode = fusion or self.fusion
        if mode not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {mode!r}. Valid: {FUSION_MODES}")
        effective_top_k = candidate_k or profile.top_k
        fetch_size = candidate_k or effective_top_k * _RERANK_OVERFETCH
        vector_size = len(query_vector) if query_vector is not None else 0
        acl_filter_clause = {"terms": {"acl_tags": effective_acl_tags}}
        es_query_dsl = {
            "size": fetch_size,
            "fusion": mode,
            "query": self._lexical_query(expanded_query, profile, [acl_filter_clause]),
            "knn": {
                "field": "embedding",
                "query_vector": f"<{vector_size}-dim Dense Vector>",
                "k": fetch_size,
                "num_candidates": max(20, fetch_size, profile.top_k * profile.num_candidates_multiplier),
                "filter": acl_filter_clause,
            },
        }

        t0 = time.perf_counter()
        masked_restricted_docs: List[Dict[str, Any]] = []
        # Exclude superseded, archived and expired content from retrieval.
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
            candidates, rankings = self._es_candidates(
                mode, expanded_query, query_vector, profile, [acl_filter_clause, current_only_filter], fetch_size
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            authorized_chunks = apply_trust_ranking(candidates)[:effective_top_k]
            masked_restricted_docs = self._audit_masked(
                expanded_query, profile, acl_filter_clause, current_only_filter
            )
        else:
            candidates, rankings = self._memory_candidates(mode, expanded_query, query_vector, effective_acl_tags)
            authorized_chunks = apply_trust_ranking(candidates)[:effective_top_k]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return HybridSearchOutcome(
            authorized_chunks=authorized_chunks,
            masked_restricted_docs=masked_restricted_docs,
            latency_ms=elapsed_ms,
            query_dsl=es_query_dsl,
            rankings={name: [c["chunk_id"] for c in ranking] for name, ranking in rankings.items()},
        )

    def _audit_masked(
        self,
        expanded_query: str,
        profile: RetrievalProfile,
        acl_filter_clause: Dict[str, Any],
        current_only_filter: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
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
        masked: List[Dict[str, Any]] = []
        seen_masked = set()
        for hit in unfiltered_resp["hits"]["hits"]:
            src = hit["_source"]
            key = (src.get("filename"), src.get("page_number"))
            if key in seen_masked:
                continue
            seen_masked.add(key)
            masked.append(
                {
                    "filename": src.get("filename"),
                    "page_number": src.get("page_number", 1),
                    "required_any_of_tags": extract_visible_tags(src.get("acl_tags", [])),
                }
            )
        return masked
