"""Grounded Answer Synthesis Adapter for BeyondRAG.

Implements `AnswerSynthesizerPort` with both deterministic fast citation synthesis
and LLM-grounded synthesis (via Gemini or local Ollama).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Final, List

from langchain_core.prompts import ChatPromptTemplate

from src.config import get_llm
from src.rag.embeddings import STOPWORDS, has_cjk

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


class GroundedAnswerSynthesizer:
    """Synthesizes RBAC-aware and cross-lingual grounded answers from authorized chunks."""

    def __init__(self, llm_factory: Callable[..., Any] = get_llm) -> None:
        self._llm_factory = llm_factory

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
    ) -> str:
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

        relevant_chunks = [
            c
            for c in authorized_chunks
            if not q_tokens
            or sum(
                1
                for tok in q_tokens
                if tok in c["content"].lower() or tok in c["filename"].lower()
            )
            >= max(1, min(2, len(q_tokens)))
        ]

        if not relevant_chunks:
            if masked_restricted_docs:
                locked_doc = masked_restricted_docs[0]["filename"]
                req_role_str = ", ".join(masked_restricted_docs[0]["required_any_of_tags"])
                if is_cjk_query:
                    return (
                        f"🔒 **RBAC 權限攔截 (Access Blocked)**\n"
                        f"📄 `{locked_doc}` — Required Role: **`{req_role_str}`**"
                    )
                return (
                    f"🔒 **Access Blocked by RBAC Pre-Filter**\n"
                    f"📄 `{locked_doc}` — Required Role: **`{req_role_str}`**"
                )
            return f"No matching documents for role {role_display}."

        if synthesis_mode == "llm":
            try:
                context_str = "\n\n".join(
                    f"Source: {c['filename']} (Page {c['page_number']}, Tags: {c['tags'] or ['PUBLIC']})\n"
                    f"Content: {c['content']}"
                    for c in relevant_chunks
                )
                lang_instruction = (
                    "Respond in Traditional Chinese in 2 concise bullet points citing [filename.pdf, p. 1]."
                    if is_cjk_query
                    else "Respond in English in 2 concise bullet points citing [filename.pdf, p. 1]."
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are the BeyondRAG enterprise assistant. Answer concisely using ONLY "
                            f"the authorized chunks. {lang_instruction}",
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
                    return content
            except Exception:
                pass

        top = relevant_chunks[0]
        tag_label = "PUBLIC" if top["is_public"] else ", ".join(top["tags"])
        raw_text = top["content"]
        if "(Page 1)" in raw_text:
            raw_text = raw_text.split("(Page 1)", 1)[-1].strip()
        for prefix in _BANNER_PREFIXES:
            raw_text = raw_text.replace(prefix, "").strip()

        if is_cjk_query:
            return (
                f"✅ **跨語言檢索成功 (ZH → EN PDF Unlocked)**\n"
                f"📄 `{top['filename']}` *(Role: `{tag_label}`)*\n"
                f"💡 {raw_text[:155]}..."
            )

        return (
            f"✅ **Authorized Match Retrieved**\n"
            f"📄 `{top['filename']}` *(Role: `{tag_label}`)*\n"
            f"💡 {raw_text[:155]}..."
        )
