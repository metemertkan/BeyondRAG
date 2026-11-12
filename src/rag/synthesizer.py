"""Grounded Answer Synthesis Adapter for BeyondRAG.

Implements `AnswerSynthesizerPort` with both deterministic fast citation synthesis
and LLM-grounded synthesis (via Gemini or local Ollama).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm
from src.rag.routing import cascade_first_model, cascade_first_tier, tier_models
from src.rag import telemetry
from src.rag.embeddings import STOPWORDS, has_cjk
from src.rag.fusion import reorder_for_attention
from src.rag.governance import DEFAULT_TRUST_TIER, SOURCE_PRECEDENCE_RULES, detect_conflicts
from src.rag.multilingual import normalize_for_matching, split_sentences
from src.rag.tabular_store import TableQuery
from src.rag.trust import ABSTAIN_MESSAGE, CITATION_RULES, AnswerVerifier, is_not_found, render_citations, without_not_found

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


@dataclass(frozen=True)
class PromptPolicy:
    """Token optimization for the generation prompt. All off by default: each changes the prompt."""

    compact_sources: bool = False  # "[S1] file p.1" instead of the full header; trust details only on conflict
    context_budget: int = 0  # about this many context tokens in total, by keeping the best sentences; 0 = all
    max_answer_tokens: int = 0  # cap on generated tokens; 0 = the model's default

    @classmethod
    def from_env(cls) -> "PromptPolicy":
        return cls(
            compact_sources=os.getenv("RAG_COMPACT_SOURCES", "off").strip().lower() == "on",
            context_budget=int(os.getenv("RAG_CONTEXT_BUDGET", "0")),
            max_answer_tokens=int(os.getenv("RAG_ANSWER_MAX_TOKENS", "0")),
        )


def compress(text: str, tokens: set[str], budget_chars: int) -> str:
    """Keep the sentences that share the most words with the question, in their original order,
    up to ``budget_chars``. Numbers stay with their sentence, so a value is never cut from its subject."""
    if len(text) <= budget_chars:
        return text
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text[:budget_chars]

    def score(sentence: str) -> float:
        lowered = normalize_for_matching(sentence).lower()
        return sum(1 for token in tokens if token in lowered) + (0.5 if any(ch.isdigit() for ch in sentence) else 0.0)

    ranked = sorted(range(len(sentences)), key=lambda i: (-score(sentences[i]), i))
    keep: List[int] = []
    used = 0
    for index in ranked:
        if keep and used + len(sentences[index]) > budget_chars:
            continue
        keep.append(index)
        used += len(sentences[index]) + 1
    return " ".join(sentences[i] for i in sorted(keep))


class GroundedAnswerSynthesizer:
    """Synthesizes RBAC-aware and cross-lingual grounded answers from authorized chunks."""

    def __init__(
        self,
        llm_factory: Callable[..., Any] = get_llm,
        *,
        reorder_context: bool = True,
        cite_sources: bool = True,
        prompt_policy: Optional["PromptPolicy"] = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._prompt_policy = prompt_policy or PromptPolicy.from_env()
        # Off reproduces the earlier prompt, which cited file names and had no way to say "not found".
        self._cite_sources = cite_sources
        self._reorder_context = reorder_context
        # Set per call: "llm" or "fast". Lets the eval gate detect a silent fallback.
        self.last_mode_used = "fast"
        # The exact sources the model was given, tool results and headers included.
        self.last_context_parts: List[str] = []
        # Source id ("S1") -> the text the model saw, and the answer before citations were rendered.
        self.last_sources: Dict[str, Dict[str, Any]] = {}
        self.last_raw_answer = ""

    def _cascade_problem(self, content: str) -> str:
        """Why a small model's answer should go to the large model, or "" to keep it."""
        if not content:
            return "the small model returned nothing"
        if is_not_found(content):
            return "the small model said the sources don't answer it"
        if not self._cite_sources:
            return ""
        verdict = AnswerVerifier(numeric=True).report(content, self.last_sources, use_llm=False).verdict
        if verdict in {"has_unsupported_claims", "citation_problems", "empty"}:
            return f"cheap checks: {verdict}"
        return ""

    def _run_table_query(
        self,
        question: str,
        table_chunks: List[Dict[str, Any]],
        table_query_runner: Callable[[TableQuery], Dict[str, Any]],
    ) -> Optional[str]:
        """Let the LLM fill in one structured table query, and format the rows as a source."""
        descriptions = "\n\n".join(chunk["content"] for chunk in table_chunks)
        planner = self._llm_factory(role="manager", temperature=0.0, task="table").with_structured_output(TableQuery)
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
        self.last_sources = {}
        self.last_raw_answer = ""
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
                telemetry.note("access_blocked", "the only matches are documents this user's roles cannot read")
                return (
                    f"🔒 **Access Blocked by RBAC Pre-Filter**\n"
                    f"📄 `{locked_doc}` — Required Role: **`{req_role_str}`**"
                )
            telemetry.note("no_documents", "no retrieved chunk matches the question")
            return f"No matching documents for role {role_display}."

        if synthesis_mode == "llm":
            try:
                # Models attend most to the start and end of a long context, so the
                # strongest chunks go at the edges and the weakest in the middle.
                ordered = reorder_for_attention(relevant_chunks) if self._reorder_context else relevant_chunks
                sources: List[Dict[str, Any]] = []
                table_chunks = [c for c in relevant_chunks if c.get("block_type") == "table_summary"]
                if table_chunks and table_query_runner is not None:
                    try:
                        with telemetry.stage("table_query"):
                            table_source = self._run_table_query(question, table_chunks, table_query_runner)
                        if table_source:
                            sources.append({"filename": table_chunks[0]["filename"], "page_number": table_chunks[0]["page_number"], "text": table_source})
                    except Exception as exc:
                        # The summary chunk is still in context, so the answer can proceed.
                        logger.warning("Structured table query failed: %s", exc)
                        telemetry.note("table_query_failed", f"{type(exc).__name__}: {exc}")
                policy = self._prompt_policy
                # Trust tier and dates only matter when sources disagree (Episode 2 precedence).
                conflicting = bool(detect_conflicts(relevant_chunks)) if policy.compact_sources else True
                per_chunk = policy.context_budget * 4 // max(1, len(ordered)) if policy.context_budget else 0
                for c in ordered:
                    content_text = compress(c["content"], q_tokens, per_chunk) if per_chunk else c["content"]
                    if policy.compact_sources:
                        header = f"{c['filename']} p.{c['page_number']}"
                        if conflicting:
                            header += f", trust tier {c.get('trust_tier') or DEFAULT_TRUST_TIER}, effective {c.get('effective_date') or 'unknown'}"
                        text = f"{header}\n{content_text}"
                    else:
                        text = (
                            f"Source: {c['filename']} (Page {c['page_number']}, Tags: {c['tags'] or ['PUBLIC']}, "
                            f"Trust tier: {c.get('trust_tier') or DEFAULT_TRUST_TIER}, "
                            f"Effective: {c.get('effective_date') or 'unknown'})\n"
                            f"Content: {content_text}"
                        )
                    sources.append(
                        {"filename": c["filename"], "page_number": c["page_number"], "chunk_id": c.get("chunk_id"), "text": text}
                    )
                # Short ids let the model cite per sentence, and let the answer be checked against them.
                self.last_sources = {f"S{i}": s for i, s in enumerate(sources, start=1)}
                context_parts = (
                    [f"[{sid}] {s['text']}" for sid, s in self.last_sources.items()]
                    if self._cite_sources
                    else [s["text"] for s in sources]
                )
                self.last_context_parts = list(context_parts)
                context_str = "\n\n".join(context_parts)
                # Language choice is query-driven — no hardcoded language-pair instruction map.
                lang_instruction = (
                    "Respond in the same language as the user's question in 2 concise bullet points. "
                    f"{CITATION_RULES}"
                    if self._cite_sources
                    else "Respond in the same language as the user's question in 2 concise bullet points citing [filename.pdf, p. 1]."
                )
                precedence = f"\n\n{SOURCE_PRECEDENCE_RULES}" if conflicting else ""
                # Static text first and the same every time: Ollama reuses the work on a shared prefix.
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are the BeyondRAG enterprise assistant. Answer concisely using ONLY "
                            f"the authorized chunks. {lang_instruction}{precedence}",
                        ),
                        (
                            "human",
                            "Active User Roles: {roles}\n\n"
                            "Authorized Context:\n{context}\n\n"
                            "Question: {question}",
                        ),
                    ]
                )
                inputs = {
                    "roles": ", ".join(user_roles) if user_roles else "Public Only",
                    "context": context_str,
                    "question": question,
                }

                def generate(model: Optional[str] = None) -> str:
                    llm = self._llm_factory(
                        role="architect", temperature=0.1, task="generate", model=model,
                        max_tokens=policy.max_answer_tokens or None,
                    )
                    resp = (prompt | llm).invoke(inputs)
                    return str(getattr(resp, "content", resp)).strip()

                first = cascade_first_model()
                if first:
                    telemetry.route_decision("generate", cascade_first_tier(), first)
                    content = generate(first)
                    problem = self._cascade_problem(content)
                    if problem:
                        # The cheap model's answer failed a cheap check: pay for the large one.
                        telemetry.note("route_escalated", problem)
                        large = tier_models()["large"]
                        telemetry.route_decision("generate", "large", large)
                        content = generate(large)
                else:
                    content = generate()
                if content:
                    self.last_mode_used = "llm"
                    self.last_raw_answer = content
                    if not self._cite_sources:
                        return content
                    if is_not_found(content):
                        telemetry.note("model_abstained", "the model said the sources do not answer the question")
                        return ABSTAIN_MESSAGE
                    return render_citations(without_not_found(content), self.last_sources)
                # An empty answer used to fall through to the fast answer with no log line at all.
                logger.warning("The model returned an empty answer; using the fast citation answer")
                telemetry.note("llm_fallback", "the model returned an empty answer")
            except Exception as exc:
                logger.warning("LLM synthesis failed; using the fast citation answer: %s", exc)
                telemetry.note("llm_fallback", f"{type(exc).__name__}: {exc}")

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
