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


# Allowlisted pack/config data modules (language codes may appear as DATA here only).
_LANG_HARDCODE_ALLOWLIST = frozenset(
    {
        "language_packs.py",
    }
)

_FORBIDDEN_LANG_HARDCODE = re.compile(
    r"""
    priority\s*=\s*\[[^\]]*(?:["']ja["']|["']zh["']|["']en["'])  # detector lists
    |Extract\s+bilingual\s+EN\s*<[-=>]+\s*ZH
    |_matches_zh_triggers
    |zh_triggers
    |en_triggers
    |Traditional\s+Chinese
    |ZH\s*→\s*EN
    |EN\s*<->\s*ZH
    |["']en["']\s*,\s*["']zh["']
    |["']zh["']\s*,\s*["']en["']
    |\bor\s+["']en["']
    |startswith\(\s*["']en["']\s*\)
    """,
    re.VERBOSE,
)


@pytest.mark.ep2
def test_src_has_no_hardcoded_language_codes_or_maps():
    """Guard: language codes / pair maps / trigger field names must not hardcode in src/.

    Allowlist: pack/config data modules only. Behavioural coverage is separate
    (JA-only runtime pack + concept_trigger_field tests).
    """
    src_root = SRC_RAG.parent  # src/
    offenders: List[str] = []
    for path in src_root.rglob("*.py"):
        if path.name in _LANG_HARDCODE_ALLOWLIST:
            continue
        # Skip __pycache__
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        match = _FORBIDDEN_LANG_HARDCODE.search(text)
        if match:
            # Find line number for claim-matrix reporting.
            line_no = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(src_root.parent)}:{line_no}:{match.group(0)[:40]}")
    assert offenders == [], "hardcoded language/mapping leftovers:\n" + "\n".join(offenders)


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
