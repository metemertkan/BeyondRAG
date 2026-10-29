"""Cross-Lingual Query Handling & Dynamic Concept Extraction Adapter.

Implements `CrossLingualPort` by bridging English and Traditional/Simplified Chinese
via dynamic Ollama ingestion-time concept extraction (`embeddings.py`) and LLM
fallback query translation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from src.config import get_llm
from src.rag.embeddings import (
    embed_text,
    expand_cross_lingual_query,
    extract_bilingual_concepts_at_ingestion,
    has_cjk,
)


class DefaultCrossLingualAdapter:
    """Adapter implementing `CrossLingualPort` for EN <-> ZH query handling and embedding."""

    def __init__(self, llm_factory: Callable[..., Any] = get_llm) -> None:
        self._llm_factory = llm_factory

    def is_cross_lingual(self, text: str) -> bool:
        return has_cjk(text)

    def extract_concepts_at_ingestion(
        self,
        filename: str,
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        return extract_bilingual_concepts_at_ingestion(
            filename=filename,
            full_text=full_text,
        )

    def _translate_unseen_chinese_query(self, question: str) -> str:
        """Use the configured LLM router to translate unseen Chinese queries into English keywords."""
        try:
            llm = self._llm_factory(role="manager", temperature=0.0)
            resp = llm.invoke(
                f"Translate this Chinese query into concise English search keywords only (no explanation): {question}"
            )
            translated = str(getattr(resp, "content", resp)).strip()
            return f"{question} {translated}" if translated else question
        except Exception:
            return question

    def prepare_query(
        self,
        question: str,
        allow_llm_fallback: bool = False,
    ) -> Tuple[str, List[str]]:
        expanded_query, matched_concepts = expand_cross_lingual_query(question)
        if has_cjk(question) and not matched_concepts and allow_llm_fallback:
            expanded_query = self._translate_unseen_chinese_query(question)
        return expanded_query, matched_concepts

    def embed(self, text: str) -> List[float]:
        return embed_text(text)
