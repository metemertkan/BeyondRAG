"""Component 9: Language-Independent Multilingual Retrieval, Concept Extraction & Translation.

Provides independent contracts:
  - LanguageDetector: Detects text language without confusing encoding with language
  - QueryTranslator: Translates queries across supported languages with fallback & logging
  - ConceptExtractor: Extracts domain concepts independently of embedding models
  - MultilingualQueryAdapter: Orchestrator implementing CrossLingualPort
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Tuple

import httpx

from src.config import OLLAMA_BASE_URL, WORKSPACE_DIR, _detect_host_ollama_models, get_llm
from src.rag.models import Concept

logger = logging.getLogger(__name__)

_CJK_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\uffff]")
_HIRAGANA_KATAKANA_RE: Final[re.Pattern[str]] = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE: Final[re.Pattern[str]] = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC_RE: Final[re.Pattern[str]] = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE: Final[re.Pattern[str]] = re.compile(r"[\u0600-\u06ff]")

DYNAMIC_CONCEPTS_PATH: Final[Path] = WORKSPACE_DIR / "dynamic_concepts.json"

_GENERIC_CJK_BIGRAMS: Final[set[str]] = {
    "政策", "規定", "规定", "什麼", "什么", "請問", "请问", "公司", "香港", "要求",
    "多少", "時間", "时间", "工作", "管理", "系統", "系统", "標準", "标准", "部門", "部门",
}


class LanguageDetector(Protocol):
    """Protocol for detecting natural language of texts."""

    def detect(self, text: str) -> str:
        """Return ISO 639-1 language code (e.g. 'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de')."""
        ...


class QueryTranslator(Protocol):
    """Protocol for translating user queries into corpus language."""

    def translate(
        self,
        query: str,
        source_language: str,
        target_language: str,
    ) -> str:
        ...


class ConceptExtractor(Protocol):
    """Protocol for extracting semantic concepts from documents."""

    def extract(
        self,
        text: str,
        language: str,
    ) -> List[Concept]:
        ...


class RuleBasedLanguageDetector:
    """Robust rule-based language detector distinguishing scripts and common vocabulary."""

    def detect(self, text: str) -> str:
        if not text or not text.strip():
            return "en"

        cleaned = text.strip()

        # Script-based detections
        if _HIRAGANA_KATAKANA_RE.search(cleaned):
            return "ja"
        if _HANGUL_RE.search(cleaned):
            return "ko"
        if _CYRILLIC_RE.search(cleaned):
            return "ru"
        if _ARABIC_RE.search(cleaned):
            return "ar"
        # Japanese specific markers
        if any(c in cleaned for c in ["号", "警報", "円", "の", "は", "に", "を"]):
            if any(ord(c) in range(0x3040, 0x30FF) for c in cleaned) or "の" in cleaned:
                return "ja"
        if _CJK_CHAR_RE.search(cleaned):
            return "zh"

        # Latin text: check for language-specific marker words / accents without misclassifying generic English
        lowered = f" {cleaned.lower()} "
        if any(w in lowered for w in [" quelle ", " quel ", " est ", " du ", " des ", " dans ", " avec ", " télétravail "]):
            return "fr"
        if any(w in lowered for w in [" cuál ", " el ", " los ", " las ", " por ", " para ", " cómo ", "¿"]):
            return "es"
        if any(w in lowered for w in [" der ", " die ", " das ", " und ", " nicht ", " mit ", " wie "]):
            return "de"

        return "en"


def _matches_zh_triggers(text: str, zh_triggers: List[str]) -> bool:
    """Return True if text matches a Chinese trigger phrase or shares a domain CJK 2-gram."""
    if not _CJK_CHAR_RE.search(text):
        return False
    for zt in zh_triggers:
        if not zt:
            continue
        if zt in text:
            return True
        if len(zt) >= 2:
            for i in range(len(zt) - 1):
                bg = zt[i : i + 2]
                if len(_CJK_CHAR_RE.findall(bg)) == 2 and bg not in _GENERIC_CJK_BIGRAMS:
                    if bg in text:
                        return True
    return False


class DynamicConceptRegistry:
    """Manages dynamically extracted concept mappings persisted in dynamic_concepts.json."""

    def __init__(self, storage_path: Path = DYNAMIC_CONCEPTS_PATH) -> None:
        self.storage_path = storage_path
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._query_cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._documents = raw.get("documents", {})
                self._query_cache = raw.get("query_cache", {})
        except Exception:
            self._documents = {}
            self._query_cache = {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self._documents,
            "query_cache": self._query_cache,
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all_concepts(self) -> List[Tuple[str, List[str], List[str], str]]:
        concepts: List[Tuple[str, List[str], List[str], str]] = []
        seen_ids: set[str] = set()
        for doc_entry in self._documents.values():
            for item in doc_entry.get("concepts", []):
                cid = str(item.get("concept_id", "")).strip()
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                en_t = [str(x).lower().strip() for x in item.get("en_triggers", []) if str(x).strip()]
                zh_t = [str(x).strip() for x in item.get("zh_triggers", []) if str(x).strip()]
                en_exp = str(item.get("en_expansion", "")).strip()
                if en_t or zh_t:
                    concepts.append((cid, en_t, zh_t, en_exp))
        return concepts

    def get_document_concepts(self, content_hash: str) -> List[Dict[str, Any]] | None:
        entry = self._documents.get(content_hash)
        if entry and isinstance(entry.get("concepts"), list) and entry["concepts"]:
            return entry["concepts"]
        return None

    def register_document_concepts(
        self,
        content_hash: str,
        filename: str,
        concepts: List[Dict[str, Any]],
        model_used: str,
    ) -> None:
        self._documents[content_hash] = {
            "filename": filename,
            "model_used": model_used,
            "concepts": concepts,
        }
        self._save()

    def get_cached_query_translation(self, query: str) -> str | None:
        return self._query_cache.get(query.strip())

    def cache_query_translation(self, query: str, english_keywords: str) -> None:
        self._query_cache[query.strip()] = english_keywords.strip()
        self._save()


_REGISTRY: DynamicConceptRegistry | None = None


def get_concept_registry() -> DynamicConceptRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DynamicConceptRegistry()
    return _REGISTRY


def _select_ollama_model() -> str:
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    available = _detect_host_ollama_models(OLLAMA_BASE_URL)
    # Prefer lightweight low-resource models first so experiments run fast on CPU/weak GPU
    for preferred in ("qwen2.5:1.5b", "qwen2.5:0.5b", "llama3.2:1b", "qwen2.5:7b", "qwen3:8b"):
        if preferred in available:
            return preferred
    return available[0] if available else "qwen2.5:1.5b"


def extract_bilingual_concepts_at_ingestion(
    filename: str,
    full_text: str,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract or retrieve cached concept mappings for uploaded documents."""
    registry = get_concept_registry()
    content_hash = hashlib.sha256(full_text.strip().encode("utf-8")).hexdigest()[:16]

    if not force_refresh:
        cached = registry.get_document_concepts(content_hash)
        if cached:
            model_name = registry._documents.get(content_hash, {}).get("model_used", "ollama-cached")
            return cached, model_name

    model_name = _select_ollama_model()
    prompt_text = (
        f"Document: {filename}\n"
        f"Text:\n{full_text[:2500]}\n\n"
        "Extract 3 distinct domain concepts from the text so users can search this English PDF using "
        "Traditional Chinese (Hong Kong), Simplified Chinese, or English.\n"
        "Return ONLY JSON:\n"
        '{"concepts": [{"concept_id": "concept:unique_name", "en_triggers": ["kw1"], "zh_triggers": ["繁體", "简体"], "en_expansion": "english keywords"}]}'
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Extract bilingual EN<->ZH search concepts as compact JSON."},
            {"role": "user", "content": prompt_text},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 480},
    }

    extracted_concepts: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            parsed = json.loads(resp.json().get("message", {}).get("content", "{}"))
            raw_list = parsed.get("concepts", [])
            if isinstance(raw_list, list):
                for idx, item in enumerate(raw_list, start=1):
                    if not isinstance(item, dict):
                        continue
                    cid = f"concept:{content_hash[:6]}_{idx}"
                    en_t = [str(x).lower().strip() for x in item.get("en_triggers", []) if str(x).strip()]
                    zh_t = [str(x).strip() for x in item.get("zh_triggers", []) if str(x).strip()]
                    en_exp = str(item.get("en_expansion") or " ".join(en_t)).strip()
                    if en_t or zh_t:
                        extracted_concepts.append(
                            {
                                "concept_id": cid,
                                "en_triggers": en_t,
                                "zh_triggers": zh_t,
                                "en_expansion": en_exp,
                            }
                        )
    except Exception:
        extracted_concepts = []

    if extracted_concepts:
        registry.register_document_concepts(
            content_hash=content_hash,
            filename=filename,
            concepts=extracted_concepts,
            model_used=f"ollama/{model_name}",
        )

    return extracted_concepts, f"ollama/{model_name}"


def expand_cross_lingual_query(query: str) -> Tuple[str, List[str]]:
    """Expand query with registered bilingual triggers."""
    lowered = query.lower()
    matched_concepts: List[str] = []
    expansions: List[str] = []

    dynamic_concepts = get_concept_registry().get_all_concepts()
    for concept_id, en_triggers, zh_triggers, en_expansion in dynamic_concepts:
        if _matches_zh_triggers(query, zh_triggers) or any(et in lowered for et in en_triggers):
            matched_concepts.append(concept_id)
            if _CJK_CHAR_RE.search(query) and en_expansion:
                expansions.append(en_expansion)

    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
    else:
        expanded = query

    return expanded, matched_concepts


class LLMQueryTranslator:
    """LLM-backed query translator with explicit fallback, error logging, and no silent failure."""

    def __init__(self, llm_factory: Callable[..., Any] = get_llm, enabled: bool = True) -> None:
        self._llm_factory = llm_factory
        self.enabled = enabled

    def translate(
        self,
        query: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if not self.enabled or source_language.lower() == target_language.lower():
            return query

        try:
            llm = self._llm_factory(role="manager", temperature=0.0)
            prompt = (
                f"Translate this {source_language} search query into concise {target_language} "
                f"search keywords only (output strictly the translated search query without commentary):\n"
                f"Query: {query}"
            )
            resp = llm.invoke(prompt)
            translated = str(getattr(resp, "content", resp)).strip()
            if translated:
                logger.info(
                    "Translated query from %s to %s: '%s' -> '%s'",
                    source_language,
                    target_language,
                    query,
                    translated,
                )
                return translated
        except Exception as exc:
            logger.warning(
                "Query translation failed (%s -> %s) for '%s': %s. Using original query.",
                source_language,
                target_language,
                query,
                exc,
            )
        return query


class DynamicConceptExtractor:
    """Extracts domain concepts using registered concepts and heuristic detection."""

    def __init__(self, llm_factory: Callable[..., Any] = get_llm) -> None:
        self._llm_factory = llm_factory
        self._cache: Dict[str, List[Concept]] = {}

    def extract(self, text: str, language: str) -> List[Concept]:
        cache_key = f"{language}:{hash(text[:1000])}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        concepts: List[Concept] = []
        terms = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
        unique_terms = sorted(list(set(terms)))[:6]
        for idx, term in enumerate(unique_terms):
            cid = f"concept:{term.lower().replace(' ', '_')}"
            concepts.append(
                Concept(
                    concept_id=cid,
                    triggers=[term.lower(), term],
                    expansion=term,
                    source_language=language,
                )
            )

        self._cache[cache_key] = concepts
        return concepts


class MultilingualQueryAdapter:
    """Language-independent cross-lingual query orchestrator implementing CrossLingualPort."""

    def __init__(
        self,
        detector: Optional[LanguageDetector] = None,
        translator: Optional[QueryTranslator] = None,
        extractor: Optional[ConceptExtractor] = None,
        corpus_language: str = "en",
        supported_languages: Optional[List[str]] = None,
    ) -> None:
        self.detector = detector or RuleBasedLanguageDetector()
        self.translator = translator or LLMQueryTranslator()
        self.extractor = extractor or DynamicConceptExtractor()
        self.corpus_language = corpus_language
        self.supported_languages = supported_languages or [
            "en", "zh", "ja", "ko", "es", "fr", "de", "ru", "ar"
        ]

    def is_cross_lingual(self, text: str) -> bool:
        """Return True if text language differs from corpus language."""
        detected = self.detector.detect(text)
        return detected != self.corpus_language

    def extract_concepts_at_ingestion(
        self,
        filename: str,
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        # Uses registered bilingual concept table backed by dynamic_concepts.json
        return extract_bilingual_concepts_at_ingestion(filename, full_text)

    def prepare_query(
        self,
        question: str,
        allow_llm_fallback: bool = False,
    ) -> Tuple[str, List[str]]:
        """Prepare search query with language detection, concept expansion, and fallback translation."""
        source_lang = self.detector.detect(question)
        expanded_query, matched_concepts = expand_cross_lingual_query(question)

        if matched_concepts or source_lang == self.corpus_language:
            return expanded_query, matched_concepts

        if allow_llm_fallback:
            translated = self.translator.translate(
                query=question,
                source_language=source_lang,
                target_language=self.corpus_language,
            )
            if translated and translated.strip().lower() != question.strip().lower():
                expanded_query = f"{question} {translated}"

        return expanded_query, matched_concepts


# Backward-compatible alias
DefaultCrossLingualAdapter = MultilingualQueryAdapter
