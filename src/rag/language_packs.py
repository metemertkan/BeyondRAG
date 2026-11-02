"""Config/data-driven language packs for multilingual retrieval.

Episode 2 rule: no hardcoded product languages in orchestration code.
EN and ZH ship as the default pack data; additional languages are added by
loading another JSON pack (or extending ``workspace/language_packs.json``).
Detection priority, concept-trigger field names, and extraction prompts live
here (or in the JSON override) — not in ``cross_lingual.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

from src.config import WORKSPACE_DIR

DEFAULT_PACKS_PATH: Final[Path] = WORKSPACE_DIR / "language_packs.json"

# Built-in seed data. Not imported as constants into detectors — loaded through
# LanguagePackRegistry so tests can swap packs (including removing EN/ZH).
_DEFAULT_CONCEPT_EXTRACTION: Final[Dict[str, Any]] = {
    "system_prompt": (
        "Extract multilingual search concepts as compact JSON. "
        "Use the trigger field names supplied in the user prompt for each language."
    ),
    "user_prompt_template": (
        "Document: {filename}\n"
        "Text:\n{text}\n\n"
        "Extract 3 distinct domain concepts from the text so users can search this document using "
        "any of the configured languages ({language_names}).\n"
        "Return ONLY JSON:\n"
        '{{"concepts": [{{"concept_id": "concept:unique_name", {trigger_schema}, '
        '"en_expansion": "english keywords"}}]}}'
    ),
    "expansion_field": "en_expansion",
}

_BUILTIN_DEFAULT_PACKS: Final[Dict[str, Any]] = {
    "corpus_language": "en",
    # Script-first order so Han-overlap languages (JA before ZH) win on mixed CJK.
    "detection_priority": ["ja", "ko", "ru", "ar", "zh", "fr", "es", "de", "tr"],
    "concept_extraction": dict(_DEFAULT_CONCEPT_EXTRACTION),
    "packs": {
        "en": {
            "language_code": "en",
            "display_name": "English",
            "script_patterns": [],
            "marker_words": [],
            "concept_trigger_field": "en_triggers",
        },
        "zh": {
            "language_code": "zh",
            "display_name": "Chinese",
            "script_patterns": [r"[\u4e00-\uffff]"],
            "marker_words": [],
            "concept_trigger_field": "zh_triggers",
            "generic_bigrams": [
                "政策", "規定", "规定", "什麼", "什么", "請問", "请问", "公司", "香港",
                "要求", "多少", "時間", "时间", "工作", "管理", "系統", "系统",
                "標準", "标准", "部門", "部门",
            ],
        },
        "ja": {
            "language_code": "ja",
            "display_name": "Japanese",
            "script_patterns": [r"[\u3040-\u30ff]"],
            "marker_words": [],
            "concept_trigger_field": "ja_triggers",
        },
        "ko": {
            "language_code": "ko",
            "display_name": "Korean",
            "script_patterns": [r"[\uac00-\ud7af]"],
            "marker_words": [],
            "concept_trigger_field": "ko_triggers",
        },
        "fr": {
            "language_code": "fr",
            "display_name": "French",
            "script_patterns": [],
            "marker_words": ["quelle", "quel", "est", "du", "des", "dans", "avec", "télétravail"],
            "concept_trigger_field": "fr_triggers",
        },
        "es": {
            "language_code": "es",
            "display_name": "Spanish",
            "script_patterns": [],
            "marker_words": ["cuál", "el", "los", "las", "por", "para", "cómo"],
            "concept_trigger_field": "es_triggers",
        },
        "de": {
            "language_code": "de",
            "display_name": "German",
            "script_patterns": [],
            "marker_words": ["der", "die", "das", "und", "nicht", "mit", "wie"],
            "concept_trigger_field": "de_triggers",
        },
        "ru": {
            "language_code": "ru",
            "display_name": "Russian",
            "script_patterns": [r"[\u0400-\u04ff]"],
            "marker_words": [],
            "concept_trigger_field": "ru_triggers",
        },
        "ar": {
            "language_code": "ar",
            "display_name": "Arabic",
            "script_patterns": [r"[\u0600-\u06ff]"],
            "marker_words": [],
            "concept_trigger_field": "ar_triggers",
        },
        "tr": {
            "language_code": "tr",
            "display_name": "Turkish",
            "script_patterns": [],
            "marker_words": ["nedir", "nasıl", "için", "veya", "değil", "politikası", "nakit"],
            "concept_trigger_field": "tr_triggers",
        },
    },
}


@dataclass(frozen=True)
class LanguagePack:
    """One language's detection + concept-trigger configuration."""

    language_code: str
    display_name: str
    script_patterns: List[str] = field(default_factory=list)
    marker_words: List[str] = field(default_factory=list)
    concept_trigger_field: str = ""
    generic_bigrams: List[str] = field(default_factory=list)

    def compiled_scripts(self) -> List[re.Pattern[str]]:
        return [re.compile(p) for p in self.script_patterns if p]


class LanguagePackRegistry:
    """Loads language packs from JSON (workspace override or built-in defaults)."""

    def __init__(
        self,
        packs_path: Optional[Path] = None,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.packs_path = packs_path or Path(
            os.getenv("RAG_LANGUAGE_PACKS_PATH", str(DEFAULT_PACKS_PATH))
        )
        if raw is not None:
            payload = raw
        elif self.packs_path.exists():
            payload = json.loads(self.packs_path.read_text(encoding="utf-8"))
        else:
            payload = _BUILTIN_DEFAULT_PACKS

        self._packs: Dict[str, LanguagePack] = {}
        for code, entry in (payload.get("packs") or {}).items():
            if not isinstance(entry, dict):
                continue
            lang = str(entry.get("language_code") or code).strip().lower()
            self._packs[lang] = LanguagePack(
                language_code=lang,
                display_name=str(entry.get("display_name") or lang),
                script_patterns=[str(p) for p in entry.get("script_patterns") or []],
                marker_words=[str(w).lower() for w in entry.get("marker_words") or []],
                concept_trigger_field=str(
                    entry.get("concept_trigger_field") or f"{lang}_triggers"
                ),
                generic_bigrams=[str(b) for b in entry.get("generic_bigrams") or []],
            )

        # Corpus language comes from pack JSON / env / first loaded pack — never a code default.
        configured_corpus = str(
            payload.get("corpus_language") or os.getenv("RAG_CORPUS_LANGUAGE", "")
        ).strip().lower()
        if configured_corpus:
            self.corpus_language: str = configured_corpus
        elif self._packs:
            self.corpus_language = next(iter(self._packs.keys()))
        else:
            self.corpus_language = ""

        configured_priority = payload.get("detection_priority")
        if isinstance(configured_priority, list) and configured_priority:
            self.detection_priority: List[str] = [
                str(c).strip().lower() for c in configured_priority if str(c).strip()
            ]
        else:
            # Fall back to pack insertion order excluding corpus language.
            self.detection_priority = [
                c for c in self._packs.keys() if c != self.corpus_language
            ]

        extraction = payload.get("concept_extraction")
        if isinstance(extraction, dict) and extraction:
            merged = dict(_DEFAULT_CONCEPT_EXTRACTION)
            merged.update(extraction)
            self.concept_extraction: Dict[str, Any] = merged
        else:
            self.concept_extraction = dict(_DEFAULT_CONCEPT_EXTRACTION)

    @property
    def supported_languages(self) -> List[str]:
        return sorted(self._packs.keys())

    def get(self, language_code: str) -> Optional[LanguagePack]:
        return self._packs.get(language_code.strip().lower())

    def ordered_detection_codes(self) -> List[str]:
        """Return detection order: configured priority (present packs), then remaining."""
        ordered = [c for c in self.detection_priority if c in self._packs]
        ordered.extend(
            c
            for c in self.supported_languages
            if c not in ordered and c != self.corpus_language
        )
        return ordered

    def trigger_fields(self) -> List[str]:
        """Concept JSON field names derived from loaded packs (not hardcoded)."""
        fields: List[str] = []
        seen: set[str] = set()
        for code in self.ordered_detection_codes() + [self.corpus_language]:
            pack = self.get(code)
            if pack is None:
                continue
            field_name = pack.concept_trigger_field
            if field_name and field_name not in seen:
                seen.add(field_name)
                fields.append(field_name)
        for pack in self._packs.values():
            if pack.concept_trigger_field and pack.concept_trigger_field not in seen:
                seen.add(pack.concept_trigger_field)
                fields.append(pack.concept_trigger_field)
        return fields

    def concept_extraction_prompts(
        self,
        *,
        filename: str,
        text: str,
    ) -> tuple[str, str]:
        """Build system + user prompts from pack config (no EN/ZH literals in callers)."""
        system = str(
            self.concept_extraction.get("system_prompt")
            or _DEFAULT_CONCEPT_EXTRACTION["system_prompt"]
        )
        template = str(
            self.concept_extraction.get("user_prompt_template")
            or _DEFAULT_CONCEPT_EXTRACTION["user_prompt_template"]
        )
        names = ", ".join(
            (self.get(c).display_name if self.get(c) else c)
            for c in self.supported_languages
        ) or "configured languages"
        trigger_schema = ", ".join(
            f'"{field}": ["kw"]' for field in self.trigger_fields()
        ) or '"triggers": ["kw"]'
        user = template.format(
            filename=filename,
            text=text[:2500],
            language_names=names,
            trigger_schema=trigger_schema,
        )
        return system, user

    def expansion_field(self) -> str:
        return str(
            self.concept_extraction.get("expansion_field")
            or _DEFAULT_CONCEPT_EXTRACTION["expansion_field"]
        )

    def ensure_workspace_defaults(self) -> Path:
        """Write default packs to workspace if missing (demo / local runs)."""
        if not self.packs_path.exists():
            self.packs_path.parent.mkdir(parents=True, exist_ok=True)
            self.packs_path.write_text(
                json.dumps(_BUILTIN_DEFAULT_PACKS, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return self.packs_path


_PACK_REGISTRY: LanguagePackRegistry | None = None


def get_language_pack_registry(
    *,
    reload: bool = False,
    raw: Optional[Dict[str, Any]] = None,
) -> LanguagePackRegistry:
    global _PACK_REGISTRY
    if reload or _PACK_REGISTRY is None or raw is not None:
        _PACK_REGISTRY = LanguagePackRegistry(raw=raw)
    return _PACK_REGISTRY
