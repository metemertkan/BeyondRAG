"""Language-Agnostic Multilingual Retrieval & Dynamic Concept Adapter.

Implements language-agnostic cross-lingual querying, allowing documents in any corpus language
(configured via RAG_CORPUS_LANGUAGE, default: English) to be queried in non-corpus languages
(Chinese, Japanese, Korean, Spanish, French, German, etc.) with automatic translation and
concept expansion.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Final, List, Tuple

from src.config import get_llm
from src.rag.embeddings import (
    embed_text,
    expand_cross_lingual_query,
    extract_bilingual_concepts_at_ingestion,
    has_cjk,
)

CORPUS_PRIMARY_LANGUAGE: Final[str] = os.getenv("RAG_CORPUS_LANGUAGE", "English").strip()
SUPPORTED_LANGUAGES: Final[str] = os.getenv("RAG_SUPPORTED_LANGUAGES", "Traditional/Simplified Chinese, Japanese, Korean, Spanish, French, German").strip()


def is_non_ascii_or_multilingual(text: str) -> bool:
    """Return True if the text contains non-ASCII characters or CJK ideographs."""
    if has_cjk(text):
        return True
    return any(ord(c) > 127 for c in text)


class DefaultCrossLingualAdapter:
    """Language-agnostic cross-lingual query adapter."""

    def __init__(
        self,
        llm_factory: Callable[..., Any] = get_llm,
        corpus_language: str = CORPUS_PRIMARY_LANGUAGE,
        supported_languages: str = SUPPORTED_LANGUAGES,
    ) -> None:
        self._llm_factory = llm_factory
        self.corpus_language = corpus_language
        self.supported_languages = supported_languages

    def is_cross_lingual(self, text: str) -> bool:
        return is_non_ascii_or_multilingual(text)

    def extract_concepts_at_ingestion(
        self,
        filename: str,
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        return extract_bilingual_concepts_at_ingestion(
            filename=filename,
            full_text=full_text,
        )

    def _translate_unseen_query(self, question: str) -> str:
        """Use LLM router to translate any non-corpus language query into corpus search keywords."""
        try:
            llm = self._llm_factory(role="manager", temperature=0.0)
            prompt = (
                f"The target document corpus is in {self.corpus_language}.\n"
                f"Translate the following user search query into concise {self.corpus_language} search keywords only (no explanation):\n"
                f"Query: {question}"
            )
            resp = llm.invoke(prompt)
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
        if self.is_cross_lingual(question) and not matched_concepts and allow_llm_fallback:
            expanded_query = self._translate_unseen_query(question)
        return expanded_query, matched_concepts

    def embed(self, text: str) -> List[float]:
        return embed_text(text)


# Alias
MultilingualQueryHandler = DefaultCrossLingualAdapter
