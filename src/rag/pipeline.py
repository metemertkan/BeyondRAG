"""The retrieval pipeline: transform, search, fuse, re-rank, expand to parents.

Each stage is optional and records what it did, so an answer can show how it was retrieved.
The caller's ACL tags go into every search. Nothing downstream can widen them.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.fusion import reciprocal_rank_fusion
from src.rag.governance import apply_trust_ranking
from src.rag.parent_child import expand_to_parents
from src.rag.ports import EmbedderPort, HybridRetrieverPort, RerankerPort
from src.rag.query_transform import LlmQueryTransformer, QueryPlan, QueryVariant

logger = logging.getLogger(__name__)

_TRACE_TOP = 8


@dataclass(frozen=True)
class PipelineOptions:
    fusion: Optional[str] = None  # None uses the retriever's default.
    query_mode: str = "off"
    rerank: bool = False
    # First-stage breadth when re-ranking or fusing queries. On SciFact, 10 and 20 scored as well
    # as 50 at a fraction of the CPU time.
    candidates: int = 20
    multi_hop: bool = False
    max_hops: int = 3
    # Trust: check the answer against its sources, and what to do with unsupported statements.
    verify: bool = False
    verify_with_llm: bool = True
    trust_mode: str = "annotate"
    # With re-ranking on: answer nothing when the best candidate scores below this.
    min_relevance: Optional[float] = None

    @classmethod
    def from_env(cls) -> "PipelineOptions":
        return cls(
            query_mode=os.getenv("RAG_QUERY_MODE", "off").strip().lower(),
            rerank=os.getenv("RAG_RERANK", "false").strip().lower() == "true",
            candidates=int(os.getenv("RAG_RERANK_CANDIDATES", "20")),
            multi_hop=os.getenv("RAG_MULTI_HOP", "false").strip().lower() == "true",
            verify=os.getenv("RAG_VERIFY", "false").strip().lower() == "true",
            verify_with_llm=os.getenv("RAG_VERIFY_WITH_LLM", "true").strip().lower() == "true",
            trust_mode=os.getenv("RAG_TRUST_MODE", "annotate").strip().lower(),
            min_relevance=float(os.environ["RAG_MIN_RELEVANCE"]) if os.getenv("RAG_MIN_RELEVANCE") else None,
        )


@dataclass
class PipelineResult:
    chunks: List[Dict[str, Any]]
    masked_restricted_docs: List[Dict[str, Any]]
    latency_ms: float
    query_dsl: Dict[str, Any]
    trace: Dict[str, Any] = field(default_factory=dict)
    # Everything the lexical stage searched for, so the relevance check understands rewrites.
    search_text: str = ""


def _summary(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"chunk_id": c.get("chunk_id"), "filename": c.get("filename"), "score": c.get("score"), "text": str(c.get("content", ""))[:160]}
        for c in chunks[:_TRACE_TOP]
    ]


class RetrievalPipeline:
    def __init__(
        self,
        retriever: HybridRetrieverPort,
        embedder: EmbedderPort,
        *,
        reranker: Optional[RerankerPort] = None,
        transformer: Optional[LlmQueryTransformer] = None,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._reranker = reranker
        self._transformer = transformer or LlmQueryTransformer()

    def _search_variant(
        self,
        variant: QueryVariant,
        acl_tags: List[str],
        top_k: int,
        profile_name: Optional[str],
        options: PipelineOptions,
        candidate_k: int,
    ) -> Any:
        fusion = options.fusion
        if not variant.lexical:
            fusion = "dense"
        vector = self._embedder.embed(variant.text) if fusion != "lexical" else None
        return self._retriever.search(
            expanded_query=variant.text,
            query_vector=vector,
            effective_acl_tags=acl_tags,
            top_k=top_k,
            profile_name=profile_name,
            fusion=fusion,
            candidate_k=candidate_k,
        )

    def _rerank(self, question: str, chunks: List[Dict[str, Any]], trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._reranker is None:
            trace["rerank"] = {"status": "skipped", "reason": "No reranker is configured. Install .[rerank]."}
            return chunks
        started = time.perf_counter()
        try:
            scores = self._reranker.score(question, [c["content"] for c in chunks])
        except Exception as exc:
            # Keep the first-stage order, and say so. A silent fallback would look re-ranked.
            logger.warning("Re-ranking failed; keeping first-stage order: %s", exc)
            trace["rerank"] = {"status": "failed", "reason": str(exc)}
            return chunks
        before = [c["chunk_id"] for c in chunks]
        scored = [
            {**chunk, "first_stage_rank": rank, "score": round(1.0 / (1.0 + math.exp(-value)), 6)}
            for rank, (chunk, value) in enumerate(zip(chunks, scores), start=1)
        ]
        ranked = apply_trust_ranking(scored)
        trace["rerank"] = {
            "status": "applied",
            "model": self._reranker.model_name,
            "pairs": len(chunks),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "moved": [
                {
                    "chunk_id": c["chunk_id"],
                    "filename": c.get("filename"),
                    "from": before.index(c["chunk_id"]) + 1,
                    "to": rank,
                    "score": c["score"],
                    "text": str(c.get("content", ""))[:160],
                }
                for rank, c in enumerate(ranked[:_TRACE_TOP], start=1)
            ],
        }
        return ranked

    def run(
        self,
        question: str,
        search_text: str,
        acl_tags: List[str],
        top_k: int = 4,
        profile_name: Optional[str] = None,
        options: Optional[PipelineOptions] = None,
    ) -> PipelineResult:
        options = options or PipelineOptions()
        started = time.perf_counter()
        trace: Dict[str, Any] = {"options": asdict(options)}
        plan: QueryPlan = self._transformer.plan(question, search_text, options.query_mode)
        trace["variants"] = [{"kind": v.kind, "text": v.text, "lexical": v.lexical, "dense": v.dense} for v in plan.variants]
        if plan.errors:
            trace["transform_errors"] = plan.errors

        broad = options.rerank or len(plan.variants) > 1
        # Without a second stage, keep three times top_k so trust ranking and parent merging have room.
        candidate_k = options.candidates if broad else top_k * 3
        outcomes = [
            self._search_variant(variant, acl_tags, top_k, profile_name, options, candidate_k)
            for variant in plan.variants
        ]
        first = outcomes[0]
        trace["first_stage"] = {
            "candidates": candidate_k,
            "rankings": {name: ids[:_TRACE_TOP] for name, ids in first.rankings.items()},
        }
        if len(outcomes) == 1:
            chunks = list(first.authorized_chunks)
        else:
            fused = reciprocal_rank_fusion([o.authorized_chunks for o in outcomes])[:candidate_k]
            chunks = apply_trust_ranking(fused)
        trace["first_stage"]["top"] = _summary(chunks)
        trace["first_stage"]["count"] = len(chunks)

        if options.rerank and chunks:
            chunks = self._rerank(question, chunks, trace)

        chunks, merged = expand_to_parents(chunks)
        parents = sum(1 for c in chunks if c.get("matched_children"))
        trace["parents"] = {"expanded": parents, "children_merged": merged}
        chunks = chunks[:top_k]
        for chunk in chunks:
            chunk.pop("parent_content", None)
        trace["final"] = _summary(chunks)

        return PipelineResult(
            chunks=chunks,
            masked_restricted_docs=first.masked_restricted_docs,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            query_dsl=first.query_dsl,
            trace=trace,
            search_text=" ".join(v.text for v in plan.variants if v.lexical),
        )
