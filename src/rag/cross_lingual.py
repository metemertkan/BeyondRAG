"""Component 9: Language-Independent Multilingual Retrieval, Concept Extraction & Translation.

Provides independent contracts:
  - LanguageDetector: Detects text language without confusing encoding with language
  - QueryTranslator: Translates queries across supported languages with fallback & logging
  - ConceptExtractor: Extracts domain concepts independently of embedding models
  - MultilingualQueryAdapter: Orchestrator implementing CrossLingualPort

Product language lists, trigger field names, and extraction prompts live in
``language_packs`` / JSON config — not as hardcoded constants here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Tuple

import httpx

from src.config import OLLAMA_BASE_URL, WORKSPACE_DIR, _detect_host_ollama_models, get_llm
from src.rag.language_packs import (
    LanguagePack,
    LanguagePackRegistry,
    get_language_pack_registry,
)
from src.rag.models import Concept

# Re-export for Episode 2 acceptance / data-driven language hooks.
__all__ = [
    "LanguagePack",
    "LanguagePackRegistry",
    "load_language_pack",
    "MultilingualQueryAdapter",
    "RuleBasedLanguageDetector",
    "DefaultCrossLingualAdapter",
]

logger = logging.getLogger(__name__)

DYNAMIC_CONCEPTS_PATH: Final[Path] = WORKSPACE_DIR / "dynamic_concepts.json"


def load_language_pack(path: str | Path | None = None) -> LanguagePackRegistry:
    """Load language packs from JSON (config/data-driven defaults live in pack data)."""
    packs_path = Path(path) if path else None
    return LanguagePackRegistry(packs_path=packs_path)


class LanguageDetector(Protocol):
    """Protocol for detecting natural language of texts."""

    def detect(self, text: str) -> str:
        """Return a language code from the configured pack registry."""
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
    """Data-driven language detector backed by :class:`LanguagePackRegistry`."""

    def __init__(self, pack_registry: Optional[LanguagePackRegistry] = None) -> None:
        self.pack_registry = pack_registry or get_language_pack_registry()

    def detect(self, text: str) -> str:
        corpus = self.pack_registry.corpus_language
        if not text or not text.strip():
            return corpus

        cleaned = text.strip()
        for code in self.pack_registry.ordered_detection_codes():
            pack = self.pack_registry.get(code)
            if pack is None:
                continue
            for pattern in pack.compiled_scripts():
                if pattern.search(cleaned):
                    return code
            if pack.marker_words:
                # Strip punctuation so accented / inverted-mark queries still match markers.
                normalized = re.sub(r"[^\w\s]", " ", cleaned.lower(), flags=re.UNICODE)
                padded = f" {normalized} "
                if any(f" {w} " in padded for w in pack.marker_words):
                    return code

        return corpus


def _pack_script_matches(text: str, pack: LanguagePack) -> bool:
    return any(p.search(text) for p in pack.compiled_scripts())


def _matches_pack_triggers(
    text: str,
    triggers: List[str],
    *,
    pack: Optional[LanguagePack] = None,
) -> bool:
    """Return True if text matches a trigger phrase (script-aware bigrams from pack data)."""
    if not triggers:
        return False
    blocked = set(pack.generic_bigrams) if pack else set()
    use_bigrams = bool(pack and pack.script_patterns and pack.generic_bigrams)
    for trigger in triggers:
        if not trigger:
            continue
        if trigger in text:
            return True
        if use_bigrams and pack is not None and _pack_script_matches(trigger, pack):
            for i in range(len(trigger) - 1):
                bg = trigger[i : i + 2]
                if bg in blocked:
                    continue
                if _pack_script_matches(bg, pack) and bg in text:
                    return True
    return False


def _matches_script_triggers(
    text: str,
    triggers: List[str],
    *,
    pack: Optional[LanguagePack] = None,
) -> bool:
    """Match triggers using pack script/bigram config; latin packs casefold."""
    if pack is not None and pack.script_patterns:
        if not _pack_script_matches(text, pack):
            return any(t and t in text for t in triggers)
        return _matches_pack_triggers(text, triggers, pack=pack)
    lowered = text.lower()
    return any(t and t.lower() in lowered for t in triggers)


def _should_casefold_field(field_name: str, packs: LanguagePackRegistry) -> bool:
    """Casefold triggers for packs without script_patterns (configured, not hardcoded)."""
    for pack in packs._packs.values():  # noqa: SLF001
        if pack.concept_trigger_field == field_name:
            return not bool(pack.script_patterns)
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

    def get_all_concepts(
        self,
        *,
        pack_registry: Optional[LanguagePackRegistry] = None,
    ) -> List[Tuple[str, Dict[str, List[str]], str]]:
        """Return (concept_id, triggers_by_field, expansion) using pack-configured field names."""
        packs = pack_registry or get_language_pack_registry()
        expansion_field = packs.expansion_field()
        configured = set(packs.trigger_fields())
        concepts: List[Tuple[str, Dict[str, List[str]], str]] = []
        seen_ids: set[str] = set()
        for doc_entry in self._documents.values():
            for item in doc_entry.get("concepts", []):
                cid = str(item.get("concept_id", "")).strip()
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                triggers_by_field: Dict[str, List[str]] = {}
                for key, value in item.items():
                    field_name = str(key)
                    if configured and field_name not in configured:
                        continue
                    if not field_name.endswith("_triggers"):
                        continue
                    if not isinstance(value, list):
                        continue
                    cleaned = [str(x).strip() for x in value if str(x).strip()]
                    if cleaned:
                        if _should_casefold_field(field_name, packs):
                            cleaned = [t.lower() for t in cleaned]
                        triggers_by_field[field_name] = cleaned
                expansion = str(item.get(expansion_field, "")).strip()
                if triggers_by_field:
                    concepts.append((cid, triggers_by_field, expansion))
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
    env_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    available = _detect_host_ollama_models(OLLAMA_BASE_URL)
    if available and env_model in available:
        return env_model
    # Prefer lightweight low-resource models first so experiments run fast on CPU/weak GPU
    for preferred in ("qwen2.5:1.5b", "qwen2.5:0.5b", "llama3.2:1b", "qwen2.5:7b", "qwen3:8b"):
        if preferred in available:
            return preferred
    return env_model


def extract_concepts_at_ingestion(
    filename: str,
    full_text: str,
    force_refresh: bool = False,
    pack_registry: Optional[LanguagePackRegistry] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract or retrieve cached concept mappings for uploaded documents."""
    registry = get_concept_registry()
    packs = pack_registry or get_language_pack_registry()
    content_hash = hashlib.sha256(full_text.strip().encode("utf-8")).hexdigest()[:16]

    if not force_refresh:
        cached = registry.get_document_concepts(content_hash)
        if cached:
            model_name = registry._documents.get(content_hash, {}).get("model_used", "ollama-cached")
            return cached, model_name

    model_name = _select_ollama_model()
    system_prompt, user_prompt = packs.concept_extraction_prompts(
        filename=filename,
        text=full_text,
    )
    trigger_fields = packs.trigger_fields()
    expansion_field = packs.expansion_field()
    corpus_field = None
    corpus_pack = packs.get(packs.corpus_language)
    if corpus_pack is not None:
        corpus_field = corpus_pack.concept_trigger_field

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
                    concept: Dict[str, Any] = {"concept_id": cid}
                    any_triggers = False
                    for field_name in trigger_fields:
                        raw_triggers = item.get(field_name, [])
                        if not isinstance(raw_triggers, list):
                            continue
                        cleaned = [str(x).strip() for x in raw_triggers if str(x).strip()]
                        if _should_casefold_field(field_name, packs):
                            cleaned = [t.lower() for t in cleaned]
                        if cleaned:
                            concept[field_name] = cleaned
                            any_triggers = True
                    fallback_triggers = (
                        concept.get(corpus_field, []) if corpus_field else []
                    )
                    expansion = str(
                        item.get(expansion_field) or " ".join(fallback_triggers)
                    ).strip()
                    concept[expansion_field] = expansion
                    if any_triggers:
                        extracted_concepts.append(concept)
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


# Backward-compatible alias (name is historical; behaviour is pack-driven).
extract_bilingual_concepts_at_ingestion = extract_concepts_at_ingestion


def expand_cross_lingual_query(
    query: str,
    pack_registry: Optional[LanguagePackRegistry] = None,
) -> Tuple[str, List[str]]:
    """Expand query with registered multilingual triggers from language packs."""
    packs = pack_registry or get_language_pack_registry()
    lowered = query.lower()
    matched_concepts: List[str] = []
    expansions: List[str] = []

    # Map trigger field → pack (for bigram / script-aware matching).
    field_to_pack: Dict[str, LanguagePack] = {}
    for pack in packs._packs.values():  # noqa: SLF001 — pack iteration
        if pack.concept_trigger_field:
            field_to_pack[pack.concept_trigger_field] = pack

    corpus_pack = packs.get(packs.corpus_language)
    corpus_field = corpus_pack.concept_trigger_field if corpus_pack else None
    configured_fields = set(packs.trigger_fields())

    for concept_id, triggers_by_field, expansion in get_concept_registry().get_all_concepts(
        pack_registry=packs
    ):
        hit = False
        matched_via_non_corpus = False
        for field_name, triggers in triggers_by_field.items():
            if field_name not in configured_fields:
                continue
            pack = field_to_pack.get(field_name)
            if pack is not None:
                matched = _matches_script_triggers(query, triggers, pack=pack)
            else:
                matched = any(t.lower() in lowered for t in triggers)
            if matched:
                hit = True
                if corpus_field is None or field_name != corpus_field:
                    matched_via_non_corpus = True
        if hit:
            matched_concepts.append(concept_id)
            if matched_via_non_corpus and expansion:
                expansions.append(expansion)

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
        corpus_language: Optional[str] = None,
        supported_languages: Optional[List[str]] = None,
        language_pack: Optional[LanguagePackRegistry] = None,
    ) -> None:
        self.language_pack = language_pack or get_language_pack_registry()
        self.detector = detector or RuleBasedLanguageDetector(self.language_pack)
        self.translator = translator or LLMQueryTranslator()
        self.extractor = extractor or DynamicConceptExtractor()
        self.corpus_language = (
            corpus_language
            or self.language_pack.corpus_language
            or os.getenv("RAG_CORPUS_LANGUAGE", "").strip()
            or (self.language_pack.supported_languages[0] if self.language_pack.supported_languages else "")
        )
        self.supported_languages = supported_languages or list(self.language_pack.supported_languages)

    def is_cross_lingual(self, text: str) -> bool:
        """Return True if text language differs from corpus language."""
        detected = self.detector.detect(text)
        return detected != self.corpus_language

    def extract_concepts_at_ingestion(
        self,
        filename: str,
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        return extract_concepts_at_ingestion(
            filename,
            full_text,
            pack_registry=self.language_pack,
        )

    def prepare_query(
        self,
        question: str,
        allow_llm_fallback: bool = False,
    ) -> Tuple[str, List[str]]:
        """Prepare search query with language detection, concept expansion, and fallback translation."""
        source_lang = self.detector.detect(question)
        expanded_query, matched_concepts = expand_cross_lingual_query(
            question,
            pack_registry=self.language_pack,
        )

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
