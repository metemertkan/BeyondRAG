"""Episode 2 content-acceptance amendments (O1 languages FIX, O2 parser, O3 supersede).

Claim-matrix anchors — keep names stable for Manager reporting:
  - test_ja_only_pack_runtime_drives_detection_and_prompts
  - test_src_rag_has_no_hardcoded_language_priority_lists
  - test_concept_trigger_field_from_pack_drives_expansion
  - test_parser_allow_degraded_true_records_fallback
  - test_parser_allow_degraded_false_fails_loud
  - test_same_bytes_retry_refuses_silent_flat_over_tables
  - test_stage_then_publish_v2_indexed_before_v1_superseded
  - test_stage_then_publish_v2_fail_leaves_v1_current
  - test_stage_then_publish_always_exactly_one_current
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.rag.chunker import create_chunker
from src.rag.cross_lingual import (
    DynamicConceptRegistry,
    MultilingualQueryAdapter,
    RuleBasedLanguageDetector,
    expand_cross_lingual_query,
    get_concept_registry,
)
from src.rag.embeddings import DeterministicFakeEmbeddingProvider
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import (
    DocumentParserRegistry,
    ParserDegradedError,
    PdfPlumberDocumentParser,
    PyPdfDocumentParser,
)
from src.rag.ingestor import PdfIngestor
from src.rag.language_packs import LanguagePackRegistry, get_language_pack_registry
from src.rag.models import IngestionState, ParsedDocument, ParsedBlock
from src.rag.repository import DocumentRepository

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"
POLICY_V1 = FIXTURES / "policy_v1_liquidity_table.pdf"
POLICY_V2 = FIXTURES / "policy_v2_liquidity_table.pdf"
POLICY_MISPAIRED = FIXTURES / "policy_mispaired_limits_table.pdf"
SOURCE_KEY = "hk-treasury-fx-liquidity-q4"

SRC_RAG = Path(__file__).resolve().parents[1] / "src" / "rag"


def _memory_repo() -> DocumentRepository:
    return DocumentRepository(
        es_manager=ElasticsearchIndexManager(es_url="http://non-existent-es:9200")
    )


def _ingestor(repo: DocumentRepository, *, extractor: Any = None) -> PdfIngestor:
    return PdfIngestor(
        extractor=extractor or DocumentParserRegistry(),
        chunker=create_chunker("sliding_window", max_tokens=120),
        cross_lingual=MultilingualQueryAdapter(),
        embedder=DeterministicFakeEmbeddingProvider(dimensions=32),
        repository=repo,
    )


def _ja_only_pack_raw() -> Dict[str, Any]:
    """Runtime pack with NO en/zh — proves packs drive behaviour without code edits."""
    return {
        "corpus_language": "ja",
        "detection_priority": ["ja", "ko"],
        "concept_extraction": {
            "system_prompt": "JA-ONLY-SYSTEM-MARKER extract concepts as JSON",
            "user_prompt_template": (
                "DOC={filename}; LANGS={language_names}; SCHEMA={trigger_schema}; TEXT={text}"
            ),
            "expansion_field": "en_expansion",
        },
        "packs": {
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
        },
    }


# ---------------------------------------------------------------------------
# O1 — Languages FIX (behavioural, not string-scoped-down)
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_ja_only_pack_runtime_drives_detection_and_prompts():
    """JA-only pack loaded at runtime drives detect + concept prompts; no EN/ZH in config."""
    packs = LanguagePackRegistry(raw=_ja_only_pack_raw())
    assert "en" not in packs.supported_languages
    assert "zh" not in packs.supported_languages
    assert packs.trigger_fields() == ["ja_triggers", "ko_triggers"]

    adapter = MultilingualQueryAdapter(language_pack=packs, corpus_language="ja")
    assert adapter.detector.detect("台風警報が出ました") == "ja"
    assert adapter.detector.detect("태풍 경보") == "ko"
    # Latin/English text falls back to pack corpus_language (ja), not a hardcoded "en".
    assert adapter.detector.detect("What is the cash reserve?") == "ja"

    query, _ = adapter.prepare_query("流動性カバレッジ比率の下限は？", allow_llm_fallback=False)
    assert isinstance(query, str) and query

    system, user = packs.concept_extraction_prompts(filename="note.pdf", text="流動性")
    assert "JA-ONLY-SYSTEM-MARKER" in system
    assert "ja_triggers" in user
    assert "ko_triggers" in user
    assert "zh_triggers" not in user
    assert "en_triggers" not in user
    assert "Japanese" in user


# ISO-ish codes that must not appear as product mapping tables in src/.
_LANG_CODE = r"(?:en|zh|ja|ko|fr|es|de|ru|ar|tr|nl|pt|it|vi|th|hi|id)"

_FORBIDDEN_LANG_HARDCODE = re.compile(
    rf"""
    priority\s*=\s*\[[^\]]*["']{_LANG_CODE}["']
    |Extract\s+bilingual\s+EN\s*<[-=>]+\s*ZH
    |_matches_zh_triggers
    |zh_triggers
    |en_triggers
    |Traditional\s+Chinese
    |ZH\s*→\s*EN
    |EN\s*<->\s*ZH
    |\bor\s+["']en["']
    |startswith\(\s*["']en["']\s*\)
    |_BUILTIN_DEFAULT_PACKS
    |\[\s*["']{_LANG_CODE}["']\s*,\s*["']{_LANG_CODE}["']
    |["']{_LANG_CODE}["']\s*:\s*["'](?:{_LANG_CODE}|[a-z_]*triggers)["']
    """,
    re.VERBOSE,
)

_FORBIDDEN_HTML_JS = re.compile(
    r"""
    <html[^>]*\slang\s*=\s*["'][A-Za-z-]{2,}["']   # hardcoded document lang
    |runQuickPrompt\s*\(\s*['\"][^'\"]{4,}          # inline demo prompt strings
    |八號風球|最低港元現金儲備|葵青貨櫃碼頭             # known demo fixture copy in product HTML
    |Chinese\s*/\s*English
    |ask\s+in\s+Chinese
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _scan_src_for_language_hardcodes() -> List[str]:
    """Return offender strings ``relpath:line:snippet`` for Mete no-hardcode rule."""
    src_root = SRC_RAG.parent  # src/
    offenders: List[str] = []
    for path in src_root.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".html", ".js", ".css"}:
            continue
        # No allowlist — language_packs.py loader must also stay free of pack dicts.
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        patterns = [_FORBIDDEN_LANG_HARDCODE]
        if path.suffix.lower() in {".html", ".js"}:
            patterns.append(_FORBIDDEN_HTML_JS)
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                line_no = text[: match.start()].count("\n") + 1
                rel = path.relative_to(src_root.parent)
                offenders.append(f"{rel}:{line_no}:{match.group(0)[:48].replace(chr(10), ' ')}")
                break
    return offenders


@pytest.mark.ep2
def test_src_has_no_hardcoded_language_codes_or_maps():
    """Guard: language codes / pair maps / HTML lang+prompts must not hardcode in src/.

    Pack + UI demo data live in config/*.json. No Python allowlist (incl. language_packs.py).
    """
    offenders = _scan_src_for_language_hardcodes()
    assert offenders == [], "hardcoded language/mapping leftovers:\n" + "\n".join(offenders)


@pytest.mark.ep2
def test_language_hardcode_guard_fails_on_planted_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Negative self-test: guard must FAIL when a language-code list is planted under src/."""
    planted = SRC_RAG / "_guard_plant_language_map.py"
    planted.write_text(
        'SUPPORTED_LANGS = ["ja", "ko", "ru", "ar", "zh"]\n'
        'PAIR_MAP = {"zh": "en", "ja": "en"}\n',
        encoding="utf-8",
    )
    try:
        offenders = _scan_src_for_language_hardcodes()
        assert any("_guard_plant_language_map.py" in o for o in offenders), offenders
        assert any("ja" in o and "ko" in o or "zh" in o for o in offenders), offenders
    finally:
        planted.unlink(missing_ok=True)
    # Clean tree must still be green after removal.
    assert _scan_src_for_language_hardcodes() == []


@pytest.mark.ep2
def test_language_packs_load_from_config_json_not_python_dicts():
    """Defaults must come from config/language_packs.defaults.json at runtime."""
    from src.rag.language_packs import BUNDLED_DEFAULTS_PATH, LanguagePackRegistry

    assert BUNDLED_DEFAULTS_PATH.exists(), BUNDLED_DEFAULTS_PATH
    assert BUNDLED_DEFAULTS_PATH.suffix == ".json"
    # Loader module must not embed builtins pack dicts / language lists.
    loader_src = (SRC_RAG / "language_packs.py").read_text(encoding="utf-8")
    assert "_BUILTIN_DEFAULT_PACKS" not in loader_src
    assert not re.search(r'["\']packs["\']\s*:\s*\{', loader_src)
    assert not re.search(
        r'detection_priority["\']?\s*:\s*\[["\']ja["\']',
        loader_src,
    )

    packs = LanguagePackRegistry(packs_path=BUNDLED_DEFAULTS_PATH)
    assert packs.get("ja") is not None
    assert "ja_triggers" in packs.trigger_fields()


@pytest.mark.ep2
def test_ui_demo_prompts_come_from_config_not_html():
    """Demo prompts / html lang live in config/ui_demo.json — not hardcoded in index.html."""
    html = (Path(__file__).resolve().parents[1] / "src" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "八號風球" not in html
    assert "runQuickPrompt('" not in html
    assert not re.search(r'<html[^>]*\slang\s*=\s*["\'][A-Za-z]{2,}["\']', html, re.I)
    assert "/api/ui-demo-config" in html
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "ui_demo.json"
    assert cfg_path.exists()
    import json

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg.get("demo_prompts"), "ui_demo.json must list demo prompts"
    assert any("八號風球" in str(p.get("query", "")) for p in cfg["demo_prompts"])
    assert str(cfg.get("html_lang") or "").strip()


@pytest.mark.ep2
def test_default_packs_detect_japanese_before_chinese_han_overlap():
    """O1a: default/bundled packs + detection_priority must not let zh Han range eat kana."""
    from src.rag.cross_lingual import RuleBasedLanguageDetector
    from src.rag.language_packs import BUNDLED_DEFAULTS_PATH, LanguagePackRegistry

    packs = LanguagePackRegistry(packs_path=BUNDLED_DEFAULTS_PATH)
    assert packs.detection_priority[0] == "ja"
    # Stale workspace shape (no priority, zh before ja) must still inherit bundled priority.
    stale = {
        "corpus_language": "en",
        "packs": {
            "zh": {
                "language_code": "zh",
                "display_name": "Chinese",
                "script_patterns": [r"[\u4e00-\uffff]"],
                "concept_trigger_field": "zh_triggers",
            },
            "ja": {
                "language_code": "ja",
                "display_name": "Japanese",
                "script_patterns": [r"[\u3040-\u30ff]"],
                "concept_trigger_field": "ja_triggers",
            },
            "en": {
                "language_code": "en",
                "display_name": "English",
                "script_patterns": [],
                "concept_trigger_field": "en_triggers",
            },
        },
    }
    # Simulate incomplete workspace file load path.
    import src.rag.language_packs as lp

    merged = lp._inherit_missing_detection_fields(stale, lp.DEFAULT_PACKS_PATH)
    assert merged["detection_priority"][0] == "ja"
    reg = LanguagePackRegistry(raw=merged)
    det = RuleBasedLanguageDetector(reg)
    assert det.detect("台風第8号の警報が出ました") == "ja"
    assert det.detect("流動性カバレッジ比率の下限は？") == "ja"
    assert det.detect("香港天文台八號風球警告") == "zh"


@pytest.mark.ep2
def test_concept_trigger_field_from_pack_drives_expansion(tmp_path: Path):
    """concept_trigger_field is wired: custom field name routes expansion; zh_triggers ignored."""
    raw = {
        "corpus_language": "en",
        "detection_priority": ["ja"],
        "concept_extraction": {
            "system_prompt": "sys",
            "user_prompt_template": "{filename} {text} {language_names} {trigger_schema}",
            "expansion_field": "en_expansion",
        },
        "packs": {
            "en": {
                "language_code": "en",
                "display_name": "English",
                "script_patterns": [],
                "marker_words": [],
                "concept_trigger_field": "en_triggers",
            },
            "ja": {
                "language_code": "ja",
                "display_name": "Japanese",
                "script_patterns": [r"[\u3040-\u30ff]"],
                "marker_words": [],
                # Renamed field — proves pack config, not zh_triggers hardcode.
                "concept_trigger_field": "liquidity_ja_triggers",
            },
        },
    }
    packs = LanguagePackRegistry(raw=raw)
    assert "liquidity_ja_triggers" in packs.trigger_fields()
    assert "zh_triggers" not in packs.trigger_fields()

    storage = tmp_path / "dynamic_concepts.json"
    reg = DynamicConceptRegistry(storage_path=storage)
    reg.register_document_concepts(
        content_hash="abc123deadbeef",
        filename="policy.pdf",
        concepts=[
            {
                "concept_id": "concept:lcr",
                "en_triggers": ["liquidity coverage"],
                "liquidity_ja_triggers": ["流動性カバレッジ"],
                "zh_triggers": ["流動性覆蓋"],  # leftover key — must be ignored by pack
                "en_expansion": "liquidity coverage ratio floor",
            }
        ],
        model_used="test",
    )

    # Swap global concept registry for this assertion.
    import src.rag.cross_lingual as cl

    previous = cl._REGISTRY
    cl._REGISTRY = reg
    try:
        expanded, matched = expand_cross_lingual_query(
            "流動性カバレッジの下限は？",
            pack_registry=packs,
        )
        assert "concept:lcr" in matched
        assert "liquidity coverage ratio floor" in expanded

        # zh_triggers leftover alone must NOT match when ZH pack / field is absent.
        expanded_zh, matched_zh = expand_cross_lingual_query(
            "流動性覆蓋是多少",
            pack_registry=packs,
        )
        assert matched_zh == []
        assert expanded_zh == "流動性覆蓋是多少"
    finally:
        cl._REGISTRY = previous


# ---------------------------------------------------------------------------
# O2 — Explicit parser fallback policy
# ---------------------------------------------------------------------------


class _BoomPlumber(PdfPlumberDocumentParser):
    """Layout-aware stub that always fails so registry degraded policy is exercised."""

    def extract_pages(self, file_bytes: bytes):
        raise RuntimeError("pdfplumber boom")

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        raise RuntimeError("pdfplumber boom")


@pytest.mark.ep2
def test_parser_allow_degraded_true_records_fallback(caplog: pytest.LogCaptureFixture):
    registry = DocumentParserRegistry(prefer_layout_aware=True, allow_degraded=True)
    # Force primary to fail while keeping PyPDF registered.
    registry._parsers = [_BoomPlumber(), PyPdfDocumentParser()]  # noqa: SLF001
    with caplog.at_level(logging.WARNING):
        doc = registry.extract_document("policy.pdf", POLICY_V1.read_bytes())
    assert doc.parser_name == "pypdf"
    assert doc.parser_degraded is True
    assert "Layout-aware parse failed" in doc.parser_fallback_reason
    assert any("falling back to PyPDF" in r.message for r in caplog.records)


@pytest.mark.ep2
def test_parser_allow_degraded_false_fails_loud(caplog: pytest.LogCaptureFixture):
    registry = DocumentParserRegistry(prefer_layout_aware=True, allow_degraded=False)
    registry._parsers = [_BoomPlumber(), PyPdfDocumentParser()]  # noqa: SLF001
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ParserDegradedError, match="degraded fallback disabled|ALLOW_DEGRADED"):
            registry.extract_document("policy.pdf", POLICY_V1.read_bytes())
    assert any("degraded fallback disabled" in r.message for r in caplog.records)


@pytest.mark.ep2
def test_same_bytes_retry_refuses_silent_flat_over_tables():
    """Same bytes already INDEXED via pdfplumber must not become flat PyPDF while INDEXED."""
    if not DocumentParserRegistry().pdfplumber_available:
        pytest.skip("pdfplumber required for layout-aware first pass")

    repo = _memory_repo()
    first = _ingestor(repo, extractor=DocumentParserRegistry(allow_degraded=True))
    meta = first.ingest(
        POLICY_MISPAIRED.name,
        POLICY_MISPAIRED.read_bytes(),
        tags=["finance"],
        source_key=SOURCE_KEY,
    )
    state = repo.get_document_state(meta["doc_id"])
    assert state is not None
    assert state.parser_name == "pdfplumber"
    assert state.status == IngestionState.INDEXED

    class _FlatOnRetry(DocumentParserRegistry):
        def extract_document(self, filename: str, file_bytes: bytes) -> ParsedDocument:
            # Simulate layout primary failing on retry → degraded flat result.
            flat = PyPdfDocumentParser().extract_document(file_bytes)
            return ParsedDocument(
                page_count=flat.page_count,
                pages=list(flat.pages),
                blocks=list(flat.blocks),
                parser_name="pypdf",
                parser_degraded=True,
                parser_fallback_reason="Layout-aware parse failed (simulated)",
            )

    retry = _ingestor(repo, extractor=_FlatOnRetry(allow_degraded=True))
    with pytest.raises(ParserDegradedError, match="Same-bytes retry"):
        retry.ingest(
            POLICY_MISPAIRED.name,
            POLICY_MISPAIRED.read_bytes(),
            tags=["finance"],
            source_key=SOURCE_KEY,
        )
    # Prior layout-aware INDEXED state retained.
    kept = repo.get_document_state(meta["doc_id"])
    assert kept is not None
    assert kept.status == IngestionState.INDEXED
    assert kept.parser_name == "pdfplumber"
    assert any(c.get("block_type") == "table" for c in repo.es_manager.memory_chunks)


# ---------------------------------------------------------------------------
# O3 — Stage-then-publish supersede (exactly one current)
# ---------------------------------------------------------------------------


def _count_current_for_source(repo: DocumentRepository, source_key: str) -> int:
    docs = [
        r
        for r in repo.list_registry_documents()
        if r.source_key == source_key
        and r.is_current
        and r.status == IngestionState.INDEXED
    ]
    chunk_docs = {
        c.get("doc_id")
        for c in repo.es_manager.memory_chunks
        if c.get("source_key") == source_key
        and c.get("is_current") is not False
        and c.get("ingestion_status") != IngestionState.SUPERSEDED.value
    }
    # Registry currents must be exactly the searchable chunk docs when both exist.
    return len(docs), len(chunk_docs)  # type: ignore[return-value]


@pytest.mark.ep2
def test_stage_then_publish_v2_indexed_before_v1_superseded():
    """Observes publish order: v2 reaches INDEXED current in the same atomic flip as v1 SUPERSEDED."""
    repo = _memory_repo()
    events: List[str] = []
    original_publish = repo.publish_new_current
    original_mark = repo.mark_superseded

    def tracking_publish(report, previous_doc_id=None):  # type: ignore[no-untyped-def]
        events.append(f"publish:{report.doc_id}:before")
        # Before flip: previous must still be the sole current.
        assert repo.get_current_doc_id(SOURCE_KEY) == previous_doc_id
        original_publish(report, previous_doc_id=previous_doc_id)
        events.append(f"publish:{report.doc_id}:after")
        assert repo.get_current_doc_id(SOURCE_KEY) == report.doc_id

    def tracking_mark(doc_id: str) -> None:
        events.append(f"mark_superseded:{doc_id}")
        original_mark(doc_id)

    repo.publish_new_current = tracking_publish  # type: ignore[method-assign]
    repo.mark_superseded = tracking_mark  # type: ignore[method-assign]

    ingestor = _ingestor(repo)
    m1 = ingestor.ingest(
        POLICY_V1.name, POLICY_V1.read_bytes(), tags=["finance"], source_key=SOURCE_KEY
    )
    m2 = ingestor.ingest(
        POLICY_V2.name, POLICY_V2.read_bytes(), tags=["finance"], source_key=SOURCE_KEY
    )

    assert any(e.startswith(f"publish:{m2['doc_id']}") for e in events)
    # Standalone mark_superseded must not run before publish (old supersede-before-write).
    early_marks = [
        e for e in events if e.startswith("mark_superseded:") and f"publish:{m2['doc_id']}:before" not in events[: events.index(e) + 1]
    ]
    # Prefer: no separate mark_superseded at all when publish_new_current handles it.
    assert not any(e == f"mark_superseded:{m1['doc_id']}" for e in events) or (
        events.index(f"publish:{m2['doc_id']}:before")
        < events.index(f"mark_superseded:{m1['doc_id']}")
    )

    s1 = repo.get_document_state(m1["doc_id"])
    s2 = repo.get_document_state(m2["doc_id"])
    assert s1 is not None and s1.status == IngestionState.SUPERSEDED and s1.is_current is False
    assert s2 is not None and s2.status == IngestionState.INDEXED and s2.is_current is True


@pytest.mark.ep2
def test_stage_then_publish_v2_fail_leaves_v1_current():
    repo = _memory_repo()
    ingestor = _ingestor(repo)
    m1 = ingestor.ingest(
        POLICY_V1.name, POLICY_V1.read_bytes(), tags=["finance"], source_key=SOURCE_KEY
    )

    def boom(_chunks):  # type: ignore[no-untyped-def]
        raise RuntimeError("bulk indexing failed")

    repo.save_chunks = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="bulk indexing failed"):
        ingestor.ingest(
            POLICY_V2.name, POLICY_V2.read_bytes(), tags=["finance"], source_key=SOURCE_KEY
        )

    kept = repo.get_document_state(m1["doc_id"])
    assert kept is not None
    assert kept.status == IngestionState.INDEXED
    assert kept.is_current is True
    assert repo.get_current_doc_id(SOURCE_KEY) == m1["doc_id"]


@pytest.mark.ep2
def test_stage_then_publish_always_exactly_one_current():
    """Once a source_key has an INDEXED current, every mutation keeps exactly one."""
    repo = _memory_repo()
    observations: List[int] = []
    armed = {"on": False}

    def observe() -> None:
        if not armed["on"]:
            return
        currents = [
            r
            for r in repo._doc_states.values()  # noqa: SLF001
            if r.source_key == SOURCE_KEY
            and r.is_current
            and r.status == IngestionState.INDEXED
        ]
        observations.append(len(currents))

    original_set = repo.set_document_state
    original_publish = repo.publish_new_current
    original_mark = repo.mark_superseded
    original_save = repo.save_chunks

    def set_state(report):  # type: ignore[no-untyped-def]
        original_set(report)
        observe()

    def publish(report, previous_doc_id=None):  # type: ignore[no-untyped-def]
        original_publish(report, previous_doc_id=previous_doc_id)
        observe()

    def mark(doc_id: str) -> None:
        original_mark(doc_id)
        observe()

    def save(chunks):  # type: ignore[no-untyped-def]
        original_save(chunks)
        observe()

    repo.set_document_state = set_state  # type: ignore[method-assign]
    repo.publish_new_current = publish  # type: ignore[method-assign]
    repo.mark_superseded = mark  # type: ignore[method-assign]
    repo.save_chunks = save  # type: ignore[method-assign]

    ingestor = _ingestor(repo)
    ingestor.ingest(POLICY_V1.name, POLICY_V1.read_bytes(), tags=["finance"], source_key=SOURCE_KEY)
    assert repo.get_current_doc_id(SOURCE_KEY) is not None
    armed["on"] = True
    observations.clear()

    ingestor.ingest(POLICY_V2.name, POLICY_V2.read_bytes(), tags=["finance"], source_key=SOURCE_KEY)
    assert observations, "expected revision observations"
    assert all(n == 1 for n in observations), observations
