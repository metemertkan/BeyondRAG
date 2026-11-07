"""Grounded Answer Synthesis Adapter for BeyondRAG.

Implements `AnswerSynthesizerPort` with both deterministic fast citation synthesis
and LLM-grounded synthesis (via Gemini or local Ollama).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Final, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm
from src.rag.embeddings import STOPWORDS, has_cjk
from src.rag.fusion import reorder_for_attention
from src.rag.governance import DEFAULT_TRUST_TIER, SOURCE_PRECEDENCE_RULES, detect_conflicts
from src.rag.multilingual import normalize_for_matching, split_sentences
from src.rag.tabular_store import TableQuery

logger = logging.getLogger(__name__)

_EXTRA_IGNORED_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "why",
        "policy",
        "document",
        "limit",
        "rate",
        "minimum",
        "maximum",
    }
)

_BANNER_PREFIXES: Final[tuple[str, ...]] = (
    "RESTRICTED — FINANCE & TREASURY ROLE ONLY.",
    "RESTRICTED — OPERATIONS & LOGISTICS ROLE ONLY.",
    "RESTRICTED — OPERATIONS & SUPPLY CHAIN ROLE ONLY.",
    "RESTRICTED — JOINT FINANCE & OPERATIONS ROLES.",
    "SHARED RBAC — ACCESSIBLE TO EITHER 'FINANCE' OR 'OPERATION' ROLES.",
    "RESTRICTED FINANCE DOCUMENT:",
)


def _best_snippet(text: str, tokens: set[str]) -> str:
    """Start the quote at the sentence that best matches the query, not at the top of a long parent."""
    sentences = split_sentences(text)
    if len(sentences) < 2 or not tokens:
        return text
    def overlap(sentence: str) -> int:
        lowered = normalize_for_matching(sentence).lower()
        return sum(1 for token in tokens if token in lowered)
    best = max(range(len(sentences)), key=lambda index: (overlap(sentences[index]), -index))
    return " ".join(sentences[best:]) if overlap(sentences[best]) else text


class GroundedAnswerSynthesizer:
    """Synthesizes RBAC-aware and cross-lingual grounded answers from authorized chunks."""

    def __init__(self, llm_factory: Callable[..., Any] = get_llm, *, reorder_context: bool = True) -> None:
        self._llm_factory = llm_factory
        self._reorder_context = reorder_context
        # Set per call: "llm" or "fast". Lets the eval gate detect a silent fallback.
        self.last_mode_used = "fast"
        # The exact sources the model was given, tool results and headers included.
        self.last_context_parts: List[str] = []

    def _run_table_query(
        self,
        question: str,
        table_chunks: List[Dict[str, Any]],
        table_query_runner: Callable[[TableQuery], Dict[str, Any]],
    ) -> Optional[str]:
        """Let the LLM fill in one structured table query, and format the rows as a source."""
        descriptions = "\n\n".join(chunk["content"] for chunk in table_chunks)
        planner = self._llm_factory(role="manager", temperature=0.0).with_structured_output(TableQuery)
        plan = planner.invoke(
            "Fill in a structured query that answers the question from one of these tables. "
            "Column names must match exactly. Use an aggregate for totals or averages.\n\n"
            f"Tables:\n{descriptions}\n\nQuestion: {question}"
        )
        result = table_query_runner(plan)
        return (
            f"Source: table {result['name']} (structured query result, authoritative for this table)\n"
            f"Content: {json.dumps(result, ensure_ascii=False, default=str)}"
        )

    def synthesize(
        self,
        question: str,
        expanded_query: str,
        matched_concepts: List[str],
        is_cjk_query: bool,
        user_roles: List[str],
        authorized_chunks: List[Dict[str, Any]],
        masked_restricted_docs: List[Dict[str, Any]],
        synthesis_mode: str = "fast",
        table_query_runner: Optional[Callable[[TableQuery], Dict[str, Any]]] = None,
        ranked_by_model: bool = False,
    ) -> str:
        """``ranked_by_model`` means a cross-encoder or planner chose these chunks, so a chunk
        that shares no words with the question is still relevant and must not be filtered out."""
        self.last_mode_used = "fast"
        self.last_context_parts = []
        role_display = (
            ", ".join(f"`{r}`" for r in user_roles)
            if user_roles
            else "`None (Public Knowledge Base Only)`"
        )

        q_tokens = {
            w.lower().strip("?.,!\"'()？，。！")
            for w in expanded_query.split()
            if len(w.strip("?.,!\"'()？，。！")) >= 3
            and w.lower().strip("?.,!\"'()？，。！") not in STOPWORDS
            and w.lower().strip("?.,!\"'()？，。！") not in _EXTRA_IGNORED_TOKENS
            and not has_cjk(w)
        }

        q_tokens = {normalize_for_matching(tok) for tok in q_tokens}
        relevant_chunks = [
            c
            for c in authorized_chunks
            if ranked_by_model
            or not q_tokens
            or sum(
                1
                for tok in q_tokens
                if tok in normalize_for_matching(c["content"]).lower() or tok in c["filename"].lower()
            )
            >= max(1, min(2, len(q_tokens)))
        ]

        if not relevant_chunks:
            if masked_restricted_docs:
                locked_doc = masked_restricted_docs[0]["filename"]
                req_role_str = ", ".join(masked_restricted_docs[0]["required_any_of_tags"])
                return (
                    f"🔒 **Access Blocked by RBAC Pre-Filter**\n"
                    f"📄 `{locked_doc}` — Required Role: **`{req_role_str}`**"
                )
            return f"No matching documents for role {role_display}."

        if synthesis_mode == "llm":
            try:
                # Models attend most to the start and end of a long context, so the
                # strongest chunks go at the edges and the weakest in the middle.
                ordered = reorder_for_attention(relevant_chunks) if self._reorder_context else relevant_chunks
                context_parts = [
                    f"Source: {c['filename']} (Page {c['page_number']}, Tags: {c['tags'] or ['PUBLIC']}, "
                    f"Trust tier: {c.get('trust_tier') or DEFAULT_TRUST_TIER}, "
                    f"Effective: {c.get('effective_date') or 'unknown'})\n"
                    f"Content: {c['content']}"
                    for c in ordered
                ]
                table_chunks = [c for c in relevant_chunks if c.get("block_type") == "table_summary"]
                if table_chunks and table_query_runner is not None:
                    try:
                        table_source = self._run_table_query(question, table_chunks, table_query_runner)
                        if table_source:
                            context_parts.insert(0, table_source)
                    except Exception as exc:
                        # The summary chunk is still in context, so the answer can proceed.
                        logger.warning("Structured table query failed: %s", exc)
                self.last_context_parts = list(context_parts)
                context_str = "\n\n".join(context_parts)
                # Language choice is query-driven — no hardcoded language-pair instruction map.
                lang_instruction = (
                    "Respond in the same language as the user's question in 2 concise "
                    "bullet points citing [filename.pdf, p. 1]."
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are the BeyondRAG enterprise assistant. Answer concisely using ONLY "
                            f"the authorized chunks. {lang_instruction}\n\n{SOURCE_PRECEDENCE_RULES}",
                        ),
                        (
                            "human",
                            "Active User Roles: {roles}\n\n"
                            "Authorized Context:\n{context}\n\n"
                            "Question: {question}",
                        ),
                    ]
                )
                llm = self._llm_factory(role="architect", temperature=0.1)
                chain = prompt | llm
                resp = chain.invoke(
                    {
                        "roles": ", ".join(user_roles) if user_roles else "Public Only",
                        "context": context_str,
                        "question": question,
                    }
                )
                content = str(getattr(resp, "content", resp)).strip()
                if content:
                    self.last_mode_used = "llm"
                    return content
            except Exception as exc:
                logger.warning("LLM synthesis failed; using the fast citation answer: %s", exc)

        conflicts = detect_conflicts(relevant_chunks)
        top = relevant_chunks[0]
        # When sources disagree, the fast answer must quote the one that wins on precedence.
        for conflict in conflicts:
            winner = conflict["winner"]
            if winner:
                top = next((c for c in relevant_chunks if c.get("chunk_id") == winner["chunk_id"]), top)
                break
        tag_label = "PUBLIC" if top["is_public"] else ", ".join(top["tags"])
        raw_text = top["content"]
        if "(Page 1)" in raw_text:
            raw_text = raw_text.split("(Page 1)", 1)[-1].strip()
        for prefix in _BANNER_PREFIXES:
            raw_text = raw_text.replace(prefix, "").strip()

        banner = (
            "✅ **Cross-lingual match retrieved**"
            if is_cjk_query
            else "✅ **Authorized Match Retrieved**"
        )
        answer = (
            f"{banner}\n"
            f"📄 `{top['filename']}` *(Role: `{tag_label}`)*\n"
            f"💡 {_best_snippet(raw_text, q_tokens)[:155]}..."
        )
        for conflict in conflicts:
            # Only conflicts about the source being quoted belong in this answer.
            if top["filename"] not in {s["filename"] for s in conflict["sources"]}:
                continue
            sources = " vs ".join(f"`{s['filename']}`" for s in conflict["sources"])
            outcome = (
                f"using `{conflict['winner']['filename']}`: {conflict['reason']}"
                if conflict["winner"]
                else f"unresolved: {conflict['reason']}"
            )
            answer += f"\n⚠️ **Conflicting sources** ({sources}), {outcome}"
        return answer
