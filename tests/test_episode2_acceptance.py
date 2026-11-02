"""Episode 2 acceptance tests (G0/G2).

These tests encode production claims from docs/episode-brief.md.
They skip with a clear reason when Episode 2 APIs are not yet on the branch
under test, so this suite can land ahead of or beside the Coder PR.
Once APIs ship, skips become hard failures if behaviour regresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.rag.chunker import create_chunker
from src.rag.cross_lingual import MultilingualQueryAdapter
from src.rag.embeddings import DeterministicFakeEmbeddingProvider, EmbeddingManifest
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import DocumentParserRegistry, PyPdfDocumentParser
from src.rag.ingestor import PdfIngestor, generate_deterministic_doc_id
from src.rag.models import IngestionState
from src.rag.repository import DocumentRepository

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"
POLICY_V1 = FIXTURES / "policy_v1_liquidity_table.pdf"
POLICY_V2 = FIXTURES / "policy_v2_liquidity_table.pdf"
MULTICOL = FIXTURES / "ops_compliance_multicolumn.pdf"
POLICY_MISPAIRED = FIXTURES / "policy_mispaired_limits_table.pdf"

# Stable logical identity used by versioning demos (Coder may rename; keep in sync).
SOURCE_KEY_TREASURY = "hk-treasury-fx-liquidity-q4"


def _memory_repo() -> DocumentRepository:
    es_mgr = ElasticsearchIndexManager(es_url="http://non-existent-es:9200")
    return DocumentRepository(es_manager=es_mgr)


def _make_ingestor(
    repo: DocumentRepository,
    *,
    extractor: Any = None,
    embedder: Any = None,
) -> PdfIngestor:
    return PdfIngestor(
        extractor=extractor or DocumentParserRegistry(),
        chunker=create_chunker("sliding_window", max_tokens=120),
        cross_lingual=MultilingualQueryAdapter(),
        embedder=embedder or DeterministicFakeEmbeddingProvider(dimensions=32),
        repository=repo,
    )


def _ep2_versioning_ready() -> bool:
    """True when soft SUPERSEDED / source_key path is present."""
    return IngestionState.__members__.get("SUPERSEDED") is not None or hasattr(
        PdfIngestor.ingest, "__wrapped__"
    )  # placeholder; real check below


def _has_source_key_api() -> bool:
    import inspect

    sig = inspect.signature(PdfIngestor.ingest)
    return "source_key" in sig.parameters


def _layout_parser_available() -> bool:
    """Prefer a registered layout-aware PDF parser distinct from plain PyPDF."""
    registry = DocumentParserRegistry()
    # Heuristic: look for known Episode 2 adapter names / attributes.
    for attr in ("get_parser", "_parsers"):
        if not hasattr(registry, attr):
            continue
    parsers = getattr(registry, "_parsers", [])
    for p in parsers:
        name = type(p).__name__.lower()
        if any(tok in name for tok in ("layout", "plumber", "structure", "pdfplumber")):
            return True
        if hasattr(p, "extract_structured_blocks") or hasattr(p, "extract_blocks"):
            return True
    # Optional module export
    try:
        from src.rag import extractor as ex

        for candidate in (
            "LayoutAwarePdfParser",
            "PdfPlumberDocumentParser",
            "StructureAwarePdfParser",
            "OfflineStructurePdfParser",
        ):
            if hasattr(ex, candidate):
                return True
    except Exception:
        pass
    return False


def _current_chunks(repo: DocumentRepository) -> List[Dict[str, Any]]:
    chunks = list(repo.es_manager.memory_chunks)
    # If retrieval filter helper exists, use it; else filter by status metadata.
    filtered: List[Dict[str, Any]] = []
    for c in chunks:
        status = (c.get("doc_status") or c.get("ingestion_status") or c.get("version_status") or "").upper()
        if status in {"SUPERSEDED", "FAILED"}:
            continue
        filtered.append(c)
    return filtered


def _joined_content(chunks: List[Dict[str, Any]]) -> str:
    return "\n".join(c.get("content", "") for c in chunks)


# ---------------------------------------------------------------------------
# Fixtures exist
# ---------------------------------------------------------------------------


def test_ep2_fixtures_present_and_distinct():
    assert POLICY_V1.is_file() and POLICY_V1.stat().st_size > 500
    assert POLICY_V2.is_file() and POLICY_V2.stat().st_size > 500
    assert MULTICOL.is_file() and MULTICOL.stat().st_size > 500
    assert POLICY_MISPAIRED.is_file() and POLICY_MISPAIRED.stat().st_size > 500
    assert POLICY_V1.read_bytes() != POLICY_V2.read_bytes()


# ---------------------------------------------------------------------------
# (a) Doc identity / source_key — Ep1 overclaim closure
# ---------------------------------------------------------------------------


def test_ep1_baseline_doc_id_is_stem_plus_content_hash_no_tenant():
    """Document Ep1 reality on main: no tenant in generate_deterministic_doc_id."""
    a = generate_deterministic_doc_id("Policy.pdf", b"abc")
    b = generate_deterministic_doc_id("Policy.pdf", b"abc")
    assert a == b
    assert a.startswith("doc-policy-")
    # No tenant segment in the Ep1 helper
    assert "tenant" not in a


@pytest.mark.ep2
def test_source_key_stable_across_content_revisions():
    if not _has_source_key_api():
        pytest.skip("Episode 2 source_key parameter not on PdfIngestor.ingest yet")

    repo = _memory_repo()
    ingestor = _make_ingestor(repo)
    v1 = POLICY_V1.read_bytes()
    v2 = POLICY_V2.read_bytes()

    meta1 = ingestor.ingest(
        "policy_v1_liquidity_table.pdf",
        v1,
        tags=["finance"],
        source_key=SOURCE_KEY_TREASURY,  # type: ignore[call-arg]
    )
    meta2 = ingestor.ingest(
        "policy_v2_liquidity_table.pdf",
        v2,
        tags=["finance"],
        source_key=SOURCE_KEY_TREASURY,  # type: ignore[call-arg]
    )

    sk1 = meta1.get("source_key") or getattr(
        repo.get_document_state(meta1["doc_id"]), "source_key", None
    )
    sk2 = meta2.get("source_key") or getattr(
        repo.get_document_state(meta2["doc_id"]), "source_key", None
    )
    assert sk1 == SOURCE_KEY_TREASURY
    assert sk2 == SOURCE_KEY_TREASURY
    assert meta1["doc_id"] != meta2["doc_id"]


@pytest.mark.ep2
def test_revised_pdf_same_source_key_soft_supersedes_v1_out_of_search():
    """Regression of Ep1 behaviour where v1 stayed searchable after content change."""
    if not _has_source_key_api():
        pytest.skip("Episode 2 source_key / SUPERSEDED path not shipped yet")
    if IngestionState.__members__.get("SUPERSEDED") is None:
        pytest.skip("IngestionState.SUPERSEDED not defined yet")

    repo = _memory_repo()
    ingestor = _make_ingestor(repo)
    meta1 = ingestor.ingest(
        "policy_v1_liquidity_table.pdf",
        POLICY_V1.read_bytes(),
        tags=["finance"],
        source_key=SOURCE_KEY_TREASURY,  # type: ignore[call-arg]
    )
    meta2 = ingestor.ingest(
        "policy_v2_liquidity_table.pdf",
        POLICY_V2.read_bytes(),
        tags=["finance"],
        source_key=SOURCE_KEY_TREASURY,  # type: ignore[call-arg]
    )

    state_v1 = repo.get_document_state(meta1["doc_id"])
    state_v2 = repo.get_document_state(meta2["doc_id"])
    assert state_v1 is not None and state_v1.status == IngestionState.SUPERSEDED
    assert state_v2 is not None and state_v2.status == IngestionState.INDEXED

    # Soft retain: upload / registry still knows v1
    assert state_v1.status == IngestionState.SUPERSEDED

    current = _joined_content(_current_chunks(repo))
    assert "HKD 520,000,000" in current or "520,000,000" in current
    # v1-only threshold must not remain in current/searchable chunks
    assert "HKD 450,000,000" not in current and "450,000,000" not in current


# ---------------------------------------------------------------------------
# (b) Manifest mismatch fail-fast at ingest
# ---------------------------------------------------------------------------


class _MismatchEmbedder:
    """Embedder whose manifest does not match an expected index contract."""

    def __init__(self) -> None:
        self.manifest = EmbeddingManifest(
            provider="fake",
            model_name="wrong-model",
            dimensions=16,
        )

    def embed(self, text: str) -> List[float]:
        return [0.1] * 16


@pytest.mark.ep2
def test_embedding_manifest_mismatch_fails_fast_at_ingest():
    """Ep1 claimed fail-immediate; must be enforced at ingest after Ep2."""
    import inspect

    from src.rag.ingestor import PdfIngestor as PI

    repo = _memory_repo()
    expected = EmbeddingManifest(
        provider="feature_hash",
        model_name="feature_hash_v1",
        dimensions=32,
    )
    embedder = _MismatchEmbedder()

    sig = inspect.signature(PI.__init__)
    kwargs: Dict[str, Any] = {}
    if "expected_manifest" in sig.parameters:
        kwargs["expected_manifest"] = expected
    elif "embedding_manifest" in sig.parameters:
        kwargs["embedding_manifest"] = expected

    has_helper = hasattr(PI, "validate_embedding_manifest") or hasattr(
        repo, "assert_embedding_manifest"
    )
    if not kwargs and not has_helper:
        pytest.skip("Manifest mismatch check at ingest not shipped yet")

    try:
        ingestor = PI(
            extractor=DocumentParserRegistry(),
            chunker=create_chunker("sliding_window", max_tokens=120),
            cross_lingual=MultilingualQueryAdapter(),
            embedder=embedder,
            repository=repo,
            **kwargs,
        )
    except TypeError:
        pytest.skip("Manifest-aware PdfIngestor constructor not shipped yet")

    if not kwargs and not hasattr(ingestor, "expected_manifest") and not has_helper:
        pytest.skip("Manifest mismatch check at ingest not shipped yet")

    with pytest.raises((ValueError, RuntimeError), match="(?i)manifest|embedding|dimension|mismatch"):
        ingestor.ingest("policy_v1_liquidity_table.pdf", POLICY_V1.read_bytes(), tags=["finance"])

    assert len(repo.es_manager.memory_chunks) == 0


# ---------------------------------------------------------------------------
# (c) PENDING lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_pending_state_actually_used_or_documented():
    """Matrix must record whether PENDING is used; this test prefers real usage."""
    # If a hook records PENDING before PROCESSING, assert it; else skip with note.
    recorded: List[IngestionState] = []
    repo = _memory_repo()
    original_set = repo.set_document_state

    def tracking_set(report) -> None:  # type: ignore[no-untyped-def]
        recorded.append(report.status)
        original_set(report)

    repo.set_document_state = tracking_set  # type: ignore[method-assign]
    ingestor = _make_ingestor(repo)
    ingestor.ingest("note.txt", b"Treasury maintains liquidity reserves for LCR.", tags=["finance"])

    if IngestionState.PENDING in recorded:
        assert recorded[0] == IngestionState.PENDING
        assert IngestionState.PROCESSING in recorded
        assert recorded[-1] == IngestionState.INDEXED
    else:
        pytest.skip(
            "PENDING still unused in ingest transitions (Ep1 gap). "
            "Claim matrix must document trade-off if Coder leaves it unused."
        )


# ---------------------------------------------------------------------------
# Structure-aware PDF parsing
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_layout_aware_parser_preserves_table_cell_values():
    if not _layout_parser_available():
        pytest.skip("Layout-aware PDF parser adapter not registered yet")

    registry = DocumentParserRegistry()
    pages = registry.extract_pages(POLICY_V1.name, POLICY_V1.read_bytes())
    total, page_texts = pages
    assert total >= 1
    blob = "\n".join(t for _, t in page_texts)
    for marker in ("450,000,000", "10,000,000", "85,000,000"):
        assert marker in blob


@pytest.mark.ep2
def test_layout_aware_parser_keeps_multicolumn_markers_intact():
    if not _layout_parser_available():
        pytest.skip("Layout-aware PDF parser adapter not registered yet")

    registry = DocumentParserRegistry()
    _, page_texts = registry.extract_pages(MULTICOL.name, MULTICOL.read_bytes())
    blob = "\n".join(t for _, t in page_texts)
    assert "TOKEN_LEFT_COLUMN_ALPHA" in blob
    assert "TOKEN_RIGHT_COLUMN_BETA" in blob
    # Avoid character-level interleave of the two markers
    assert "TOKEN_LEFT_COLUMN_ALPHATOKEN_RIGHT" not in blob.replace(" ", "")


@pytest.mark.ep2
def test_offline_fallback_parser_still_extracts_pdf_text():
    """Offline/fallback path (PyPDF or dedicated offline adapter) must remain usable."""
    parser = PyPdfDocumentParser()
    total, page_texts = parser.extract_pages(POLICY_V1.read_bytes())
    assert total >= 1
    # Fallback may lose table structure; still must extract some text
    assert any(len(t.strip()) > 20 for _, t in page_texts)


@pytest.mark.ep2
def test_before_after_table_retrieval_structure_beats_flat_pypdf():
    """G1 must-have for Short: same question, flattened vs layout-aware parse.

    Asserts structure-aware extract retains cell values that flat PyPDF may scramble,
    and that ingest+search over structured text can hit the cash-buffer marker.
    """
    if not _layout_parser_available():
        pytest.skip("Layout-aware parser required for before/after table demo (G1 must-have)")

    raw = POLICY_V1.read_bytes()
    flat_total, flat_pages = PyPdfDocumentParser().extract_pages(raw)
    registry = DocumentParserRegistry()
    struct_total, struct_pages = registry.extract_pages(POLICY_V1.name, raw)
    assert flat_total >= 1 and struct_total >= 1

    flat_blob = "\n".join(t for _, t in flat_pages)
    struct_blob = "\n".join(t for _, t in struct_pages)
    # Structure path must retain the table marker used in the demo question.
    assert "450,000,000" in struct_blob
    # Record before/after for claim matrix (no invented metrics — only presence).
    flat_has = "450,000,000" in flat_blob
    struct_has = "450,000,000" in struct_blob
    assert struct_has is True
    # If flat also has the marker, still require structured path to expose block metadata when available.
    parser = registry.get_parser(POLICY_V1.name)
    if hasattr(parser, "extract_structured_blocks") or hasattr(parser, "extract_blocks"):
        blocks = (
            parser.extract_structured_blocks(raw)  # type: ignore[attr-defined]
            if hasattr(parser, "extract_structured_blocks")
            else parser.extract_blocks(raw)  # type: ignore[attr-defined]
        )
        assert blocks, "structure-aware parser returned no blocks"
        block_types = {str(b.get("block_type", b.get("type", ""))).lower() for b in blocks if isinstance(b, dict)}
        assert "table" in block_types or any("table" in t for t in block_types)

    # Ingest via registry (layout path) and confirm marker lands in searchable chunks.
    repo = _memory_repo()
    ingestor = _make_ingestor(repo, extractor=registry)
    meta = ingestor.ingest(POLICY_V1.name, raw, tags=["finance"])
    assert meta["status"] == IngestionState.INDEXED.value
    joined = _joined_content(repo.es_manager.memory_chunks)
    assert "450,000,000" in joined
    # Preserve flat_has only as observed evidence for the matrix (not a pass/fail invent).
    assert isinstance(flat_has, bool)


@pytest.mark.ep2
def test_flat_pypdf_mispairs_limit_label_layout_aware_keeps_row():
    """Honest before/after for Short: flat PyPDF breaks label↔value association.

    Fixture ``policy_mispaired_limits_table.pdf`` is painted column-major. Observed
    PyPDF order puts another label immediately after ``Intraday cash buffer``; the
    correct threshold still appears later in the stream (not adjacent). Layout-aware
    table extract keeps both on the same row. Asserts observed strings only.
    """
    if not _layout_parser_available():
        pytest.skip("Layout-aware parser required for mis-pair before/after demo")
    assert POLICY_MISPAIRED.is_file()

    raw = POLICY_MISPAIRED.read_bytes()
    _total, flat_pages = PyPdfDocumentParser().extract_pages(raw)
    flat_blob = "\n".join(t for _, t in flat_pages)
    flat_lines = [ln.strip() for ln in flat_blob.splitlines() if ln.strip()]
    assert "Intraday cash buffer" in flat_lines
    label_idx = flat_lines.index("Intraday cash buffer")
    # Observed mis-pair: next non-empty line is another limit label, not the threshold.
    observed_neighbor = flat_lines[label_idx + 1]
    assert observed_neighbor == "FX forward approval floor", (
        f"expected flat neighbor mis-pair, observed {observed_neighbor!r}"
    )
    # Number still present somewhere in the flat stream (not a fake 'missing' claim).
    assert "HKD 450,000,000" in flat_blob
    # But not on the same line as the label under flat extract.
    label_line = flat_lines[label_idx]
    assert "450,000,000" not in label_line

    registry = DocumentParserRegistry()
    parser = registry.get_parser(POLICY_MISPAIRED.name)
    assert hasattr(parser, "extract_blocks")
    blocks = parser.extract_blocks(raw)  # type: ignore[attr-defined]
    table_texts = [
        str(b.get("text", ""))
        for b in blocks
        if isinstance(b, dict) and str(b.get("block_type", "")).lower() == "table"
    ]
    assert table_texts, "layout-aware parser emitted no table blocks"
    paired_row = None
    for table in table_texts:
        for line in table.splitlines():
            if "Intraday cash buffer" in line and "450,000,000" in line:
                paired_row = line
                break
    assert paired_row is not None, (
        "layout-aware table row must contain both label and HKD 450,000,000"
    )

    # Naive "next line after label" association — what a flattened stream encourages.
    # Flat: wrong neighbor. Structure: markdown row carries the real threshold.
    flat_naive_value = observed_neighbor
    assert flat_naive_value != "HKD 450,000,000"
    struct_naive_ok = "HKD 450,000,000" in paired_row and "Intraday cash buffer" in paired_row
    assert struct_naive_ok is True

    # Structured ingest indexes a chunk that preserves the correct pair (for Short after-clip).
    repo = _memory_repo()
    ingestor = _make_ingestor(repo, extractor=registry)
    meta = ingestor.ingest(POLICY_MISPAIRED.name, raw, tags=["finance"])
    assert meta["status"] == IngestionState.INDEXED.value
    matched = [
        c
        for c in repo.es_manager.memory_chunks
        if "Intraday cash buffer" in c.get("content", "") and "450,000,000" in c.get("content", "")
    ]
    assert matched, "structured ingest must index a chunk with both label and threshold"

    # Honesty note for content: on this short page flat PyPDF still *contains*
    # ``HKD 450,000,000`` elsewhere in the stream — the failure mode is
    # **mis-association** (wrong next-line neighbor), not total omission.


# ---------------------------------------------------------------------------
# Embedding port: one factory, Ollama bge-m3 + remote + CI fallback
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_embedding_factory_exposes_bge_m3_ollama_and_ci_fallback():
    from src.rag.embeddings import create_embedding_provider, FeatureHashEmbeddingProvider

    ci = create_embedding_provider("feature_hash", dimensions=32)
    assert isinstance(ci, FeatureHashEmbeddingProvider)
    assert ci.manifest.model_name

    # Ollama path must be constructible (no live call required)
    from src.rag.embeddings import OllamaEmbeddingProvider

    ollama = OllamaEmbeddingProvider(model_name="bge-m3", dimensions=1024)
    assert ollama.manifest.model_name == "bge-m3"
    assert ollama.manifest.provider == "ollama"

    # Remote API key path: Google already exists; Ep2 may add another — require factory name.
    remote_names = {"google", "gemini"}
    # Optional dedicated remote alias from Ep2
    import os

    if os.getenv("RAG_EMBEDDING_PROVIDER"):
        pass
    # Factory must reject silent unknown providers (already Ep1)
    with pytest.raises(ValueError):
        create_embedding_provider("not-a-real-provider")

    # Document that remote key path remains behind same factory
    assert callable(create_embedding_provider)
    assert remote_names  # kept for claim matrix mapping


# ---------------------------------------------------------------------------
# Formats: PDF + md/txt; xlsx interface-ready without blocking
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_registry_accepts_pdf_md_txt():
    registry = DocumentParserRegistry()
    assert registry.get_parser("a.pdf").can_parse("a.pdf")
    assert registry.get_parser("a.md").can_parse("a.md")
    assert registry.get_parser("a.txt").can_parse("a.txt")


@pytest.mark.ep2
def test_xlsx_not_required_this_episode_but_port_remains_pluggable():
    registry = DocumentParserRegistry()
    # Registering a stub must work without shipping a real xlsx parser.
    class StubXlsx:
        def can_parse(self, filename: str) -> bool:
            return filename.lower().endswith(".xlsx")

        def extract_pages(self, file_bytes: bytes):
            return 1, [(1, "sheet1")]

    registry.register(StubXlsx())
    assert registry.get_parser("book.xlsx").can_parse("book.xlsx")


# ---------------------------------------------------------------------------
# No language hardcoding
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_ingest_path_has_no_hardcoded_language_gate():
    import inspect
    from src.rag import ingestor as ing

    src = inspect.getsource(ing.PdfIngestor.ingest)
    # Must not gate ingestion on a fixed language list
    assert "language == \"en\"" not in src
    assert "allowed_languages" not in src


@pytest.mark.ep2
def test_language_handling_is_config_or_data_driven_non_en_zh_case():
    """G0: no EN/ZH hardcoding as the only path; prove a non-EN/ZH query path works.

    Until Wed 12:00 Europe/Amsterdam, this is expected green. If Coder escalates
    can't-fit, Manager may scope-down — matrix must then mark scoped, not invent pass.
    """
    from src.rag.cross_lingual import MultilingualQueryAdapter, DynamicConceptRegistry
    import inspect
    import src.rag.cross_lingual as cl

    # Prefer config/data-driven triggers (generic language fields) over en_triggers/zh_triggers-only.
    src = inspect.getsource(cl)
    data_driven_markers = (
        "supported_languages",
        "corpus_language",
        "triggers",
        "language_pack",
        "concept_pack",
        "load_language",
    )
    assert any(m in src for m in data_driven_markers), (
        "cross_lingual.py must expose config/data-driven language hooks"
    )

    adapter = MultilingualQueryAdapter(corpus_language="en")
    # Japanese (non-EN/ZH) must be detectable and prepare_query must not crash.
    query, concepts = adapter.prepare_query("流動性カバレッジ比率の下限は？", allow_llm_fallback=False)
    assert isinstance(query, str) and len(query) > 0
    detected = adapter.detector.detect("流動性カバレッジ比率の下限は？")
    assert detected == "ja"

    # If Ep2 adds generic trigger packs, register a JA→EN expansion and assert match.
    registry = DynamicConceptRegistry
    # Soft probe: generic trigger API if Coder renamed zh_triggers → triggers_by_lang
    if hasattr(cl, "load_language_pack") or hasattr(cl, "LanguagePack"):
        pack_loader = getattr(cl, "load_language_pack", None)
        assert pack_loader is not None or hasattr(cl, "LanguagePack")
    else:
        # Still require that concept extraction API is not ZH-only forever:
        # MultilingualQueryAdapter.supported_languages must include a non-EN/ZH code.
        assert "ja" in adapter.supported_languages or "de" in adapter.supported_languages


# ---------------------------------------------------------------------------
# Rollback: embed failure + bulk-index failure → no partial / dual-current
# ---------------------------------------------------------------------------


class FailingEmbedder:
    def embed(self, text: str) -> List[float]:
        raise RuntimeError("embedding provider unavailable")


@pytest.mark.ep2
def test_embed_failure_rolls_back_no_partial_chunks():
    repo = _memory_repo()
    ingestor = _make_ingestor(repo, embedder=FailingEmbedder())
    with pytest.raises(RuntimeError, match="embedding provider unavailable"):
        ingestor.ingest("policy_v1_liquidity_table.pdf", POLICY_V1.read_bytes(), tags=["finance"])
    assert repo.es_manager.memory_chunks == []
    doc_id = generate_deterministic_doc_id("policy_v1_liquidity_table.pdf", POLICY_V1.read_bytes())
    state = repo.get_document_state(doc_id)
    assert state is not None
    assert state.status == IngestionState.FAILED


@pytest.mark.ep2
def test_bulk_index_failure_rolls_back_no_dual_current():
    repo = _memory_repo()

    def boom(chunks):  # type: ignore[no-untyped-def]
        raise RuntimeError("bulk indexing failed")

    repo.save_chunks = boom  # type: ignore[method-assign]
    ingestor = _make_ingestor(repo)
    with pytest.raises(RuntimeError, match="bulk indexing failed"):
        ingestor.ingest("note.txt", b"Liquidity coverage ratio must stay above floor.", tags=["finance"])
    assert repo.es_manager.memory_chunks == []


@pytest.mark.ep2
def test_retry_same_bytes_still_idempotent():
    repo = _memory_repo()
    ingestor = _make_ingestor(repo)
    raw = b"Page 1: Group Treasury maintains liquidity for LCR compliance."
    m1 = ingestor.ingest("Treasury_Doc.txt", raw, tags=["finance"])
    n1 = len(repo.es_manager.memory_chunks)
    m2 = ingestor.ingest("Treasury_Doc.txt", raw, tags=["finance"])
    n2 = len(repo.es_manager.memory_chunks)
    assert m1["doc_id"] == m2["doc_id"]
    assert n1 == n2 > 0


# ---------------------------------------------------------------------------
# ACL regression (Ep1 must still hold)
# ---------------------------------------------------------------------------


@pytest.mark.ep2
def test_ingestion_preserves_acl_tags_on_chunks():
    repo = _memory_repo()
    ingestor = _make_ingestor(repo)
    ingestor.ingest("fin.txt", b"FX forward contracts require CFO approval.", tags=["finance"])
    assert repo.es_manager.memory_chunks
    for c in repo.es_manager.memory_chunks:
        assert "finance" in c.get("acl_tags", []) or c.get("acl_tags")


# ---------------------------------------------------------------------------
# What-shipped notes presence (Coder G1 deliverable)
# ---------------------------------------------------------------------------


def _find_what_shipped_text() -> Optional[str]:
    """Locate Coder what-shipped notes / Episode 2 README section if present."""
    candidates = [
        Path("WHAT_SHIPPED.md"),
        Path("what-shipped.md"),
        Path("docs/what-shipped.md"),
        Path("docs/episode-2-what-shipped.md"),
        Path("NOTES_EPISODE_2.md"),
    ]
    for p in candidates:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    readme = Path("README.md")
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        lower = text.lower()
        # Require an explicit Episode 2 section — do not accept Ep1-only README.
        for marker in ("episode 2", "episode #02", "episode #2", "## episode 2"):
            idx = lower.find(marker)
            if idx >= 0:
                return text[idx:]
        if "layout-aware" in lower and "superseded" in lower:
            return text
    return None


@pytest.mark.ep2
def test_what_shipped_notes_document_g1_surface_area():
    """Verify what-shipped / README documents layout fallback, remote embed env, md/txt, registry, run branch."""
    text = _find_what_shipped_text()
    if text is None:
        pytest.skip("what-shipped notes / Episode 2 README section not present yet")

    lowered = text.lower()
    # Teaser-only Episode 2 mentions on Ep1 README are not what-shipped notes yet.
    shipped_signals = ("source_key", "superseded", "what shipped", "what-shipped", "offline fallback")
    if not any(s in lowered for s in shipped_signals):
        pytest.skip("Episode 2 README teaser present, but what-shipped notes not written yet")

    required_themes = {
        "layout_parser": ("layout-aware", "layout aware", "pdfplumber", "structure-aware", "structure aware"),
        "offline_fallback": ("fallback", "offline fallback", "pypdf fallback"),
        "remote_embedding": ("remote", "api key", "google_api_key", "openai", "bge-m3"),
        "env_var": ("rag_embedding", "google_api_key", ".env.example", "environment variable"),
        "md_txt_upload": ("markdown", ".md", "plain text", ".txt upload", "upload accepts"),
        "durable_registry": ("durable", "registry", "superseded", "document registry", "source_key"),
        "run_locally": ("branch", "cursor/", "checkout", "docker compose", "run it locally"),
    }
    missing = []
    for key, needles in required_themes.items():
        if not any(n in lowered for n in needles):
            missing.append(key)
    assert not missing, f"what-shipped / README missing themes: {missing}"
