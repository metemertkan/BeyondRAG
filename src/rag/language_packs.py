"""Config/data-driven language packs for multilingual retrieval.

Episode 2 rule: no hardcoded product languages in orchestration code.
Default packs ship as runtime-loaded JSON under ``config/language_packs.defaults.json``
(workspace override: ``workspace/language_packs.json`` or ``RAG_LANGUAGE_PACKS_PATH``).
Detection priority, concept-trigger field names, and extraction prompts live in
those data files — not as Python dicts here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

from src.config import WORKSPACE_DIR

logger = logging.getLogger(__name__)

# Repo-root defaults file (checked into git). Workspace override is preferred when present.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
BUNDLED_DEFAULTS_PATH: Final[Path] = _REPO_ROOT / "config" / "language_packs.defaults.json"
DEFAULT_PACKS_PATH: Final[Path] = WORKSPACE_DIR / "language_packs.json"


def _load_json_payload(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Language pack file must be a JSON object: {path}")
    return raw


def resolve_language_packs_path(
    packs_path: Optional[Path] = None,
) -> Path:
    """Prefer explicit path → env → workspace override → bundled config defaults."""
    if packs_path is not None:
        return Path(packs_path)
    env = os.getenv("RAG_LANGUAGE_PACKS_PATH", "").strip()
    if env:
        return Path(env)
    if DEFAULT_PACKS_PATH.exists():
        return DEFAULT_PACKS_PATH
    return BUNDLED_DEFAULTS_PATH


def _inherit_missing_detection_fields(
    payload: Dict[str, Any],
    loaded_from: Path,
) -> Dict[str, Any]:
    """If a workspace override omits detection_priority, inherit from bundled defaults.

    Tester failure mode: stale workspace/language_packs.json without detection_priority
    used pack insertion order (zh before ja) while zh Han range matched mixed JP text.
    """
    priority = payload.get("detection_priority")
    if isinstance(priority, list) and priority:
        return payload
    if not BUNDLED_DEFAULTS_PATH.exists():
        return payload
    if loaded_from.resolve() == BUNDLED_DEFAULTS_PATH.resolve():
        return payload
    try:
        bundled = _load_json_payload(BUNDLED_DEFAULTS_PATH)
    except Exception:
        return payload
    bundled_priority = bundled.get("detection_priority")
    if not (isinstance(bundled_priority, list) and bundled_priority):
        return payload
    merged = dict(payload)
    merged["detection_priority"] = list(bundled_priority)
    logger.info(
        "Inherited detection_priority from %s into incomplete pack file %s",
        BUNDLED_DEFAULTS_PATH,
        loaded_from,
    )
    return merged


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
    """Loads language packs from JSON (workspace override or bundled config defaults)."""

    def __init__(
        self,
        packs_path: Optional[Path] = None,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.packs_path = resolve_language_packs_path(packs_path)
        if raw is not None:
            payload = raw
        else:
            if not self.packs_path.exists():
                raise FileNotFoundError(
                    f"Language pack config not found at {self.packs_path}. "
                    "Ship config/language_packs.defaults.json or set RAG_LANGUAGE_PACKS_PATH."
                )
            payload = _load_json_payload(self.packs_path)
            # Incomplete workspace overrides (missing detection_priority) inherit from bundled JSON.
            payload = _inherit_missing_detection_fields(payload, self.packs_path)

        self._packs: Dict[str, LanguagePack] = {}
        for code, entry in (payload.get("packs") or {}).items():
            if not isinstance(entry, dict):
                continue
            lang = str(entry.get("language_code") or code).strip().lower()
            trigger_field = str(entry.get("concept_trigger_field") or "").strip()
            if not trigger_field:
                trigger_field = f"{lang}_triggers"
            self._packs[lang] = LanguagePack(
                language_code=lang,
                display_name=str(entry.get("display_name") or lang),
                script_patterns=[str(p) for p in entry.get("script_patterns") or []],
                marker_words=[str(w).lower() for w in entry.get("marker_words") or []],
                concept_trigger_field=trigger_field,
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
            # No hardcoded product order — insertion order excluding corpus only.
            self.detection_priority = [
                c for c in self._packs.keys() if c != self.corpus_language
            ]

        extraction = payload.get("concept_extraction")
        self.concept_extraction: Dict[str, Any] = (
            dict(extraction) if isinstance(extraction, dict) else {}
        )

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
        """Build system + user prompts from pack config JSON."""
        system = str(self.concept_extraction.get("system_prompt") or "").strip()
        template = str(self.concept_extraction.get("user_prompt_template") or "").strip()
        if not template:
            template = (
                "Document: {filename}\nText:\n{text}\n\n"
                "Extract domain concepts for languages ({language_names}). "
                "Return ONLY JSON with fields: {trigger_schema}."
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
        configured = str(self.concept_extraction.get("expansion_field") or "").strip()
        return configured or "expansion"

    def ensure_workspace_defaults(self) -> Path:
        """Ensure workspace override exists and has detection_priority from bundled JSON."""
        target = DEFAULT_PACKS_PATH
        source = BUNDLED_DEFAULTS_PATH
        if not source.exists():
            raise FileNotFoundError(
                f"Bundled language pack defaults missing: {source}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Wrote workspace language packs from %s", source)
            return target

        try:
            current = _load_json_payload(target)
        except Exception:
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            logger.warning("Replaced unreadable workspace language packs from %s", source)
            return target

        priority = current.get("detection_priority")
        if isinstance(priority, list) and priority:
            return target

        bundled = _load_json_payload(source)
        current["detection_priority"] = list(bundled.get("detection_priority") or [])
        # Keep Han-only zh scripts from bundled when present (avoid stale broad ranges).
        bundled_zh = (bundled.get("packs") or {}).get("zh") or {}
        if isinstance(bundled_zh, dict) and bundled_zh.get("script_patterns"):
            packs = current.setdefault("packs", {})
            if isinstance(packs, dict) and isinstance(packs.get("zh"), dict):
                packs["zh"]["script_patterns"] = list(bundled_zh["script_patterns"])
        target.write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Upgraded workspace language packs detection_priority from %s",
            source,
        )
        return target


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
