"""Episode 2: one test group per production ingestion challenge.

Tests run offline against the in-memory backend. The ``live_es`` group repeats the
behaviours that depend on Elasticsearch itself, and skips when no cluster is reachable.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import pytest
from fastapi.testclient import TestClient

from src.eval.harness import (
    EvalPipeline,
    FAILED,
    INCONCLUSIVE,
    PASSED,
    NoopCrossLingual,
    build_report,
    CaseResult,
    context_precision,
    contains_value,
    run_evaluation,
)
from src.eval.judge import ClaimVerdict, JudgeUnavailable, LlmJudge
from src.rag.chunker import (
    AdaptiveChunker,
    ChunkingConfig,
    SemanticChunker,
    SlidingWindowChunker,
    create_chunker,
)
from src.rag.embeddings import CrossLingualEmbedder, FeatureHashEmbeddingProvider, has_cjk
from src.rag.erasure import ErasedSourceError, ErasureService, SourceSyncHandler
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import DocumentParserRegistry, PdfPlumberDocumentParser, XlsxDocumentParser
from src.rag.governance import (
    SOURCE_PRECEDENCE_RULES,
    DocumentGovernance,
    apply_trust_ranking,
    detect_conflicts,
    is_expired,
)
from src.rag.ingestor import PdfIngestor, detect_subject_ids
from src.rag.models import IngestionState
from src.rag.multilingual import (
    MultilingualTokenizer,
    normalize_for_matching,
    repair_rtl_line,
    split_sentences,
)
from src.rag.ocr import OcrLine, order_ocr_lines
from src.rag.repository import DocumentRepository
from src.rag.retriever import HybridRetriever
from src.rag.synthesizer import GroundedAnswerSynthesizer
from src.rag.tabular_store import TableFilter, TableQuery

FIXTURES = Path(__file__).parent / "fixtures"
XLSX = FIXTURES / "xlsx" / "regional_budget.xlsx"
ARABIC_PDF = FIXTURES / "pdfs" / "arabic_mixed_policy.pdf"
ARABIC_SCAN = FIXTURES / "pdfs" / "arabic_scanned_policy.pdf"
POLICY_V1 = FIXTURES / "pdfs" / "policy_v1_liquidity_table.pdf"
POLICY_V2 = FIXTURES / "pdfs" / "policy_v2_liquidity_table.pdf"

ARABIC_GROUND_TRUTH = [
    "سياسة السيولة النقدية لعام 2026",
    "الحد الأدنى للسيولة النقدية هو HKD 520,000,000 يوميا.",
    "يجب على قسم الخزينة مراجعة هذا الحد كل 90 يوما.",
    "تتم الموافقة على العقود الآجلة من قبل المدير المالي.",
]
PERSON = "maya.chen@example.com"


class Stack:
    """Real ingestor, repository and retriever on an offline or live index."""

    def __init__(self, tmp_path: Path, es_url: str = "http://non-existent-es:9200", index_name: str | None = None) -> None:
        self.embedder = CrossLingualEmbedder(FeatureHashEmbeddingProvider(dimensions=256))
        self.es = ElasticsearchIndexManager(es_url=es_url, index_name=index_name, dimensions=256)
        self.repo = DocumentRepository(
            es_manager=self.es,
            uploads_dir=tmp_path / "uploads",
            registry_path=tmp_path / "registry.json",
        )
        self.ingestor = PdfIngestor(
            extractor=DocumentParserRegistry(),
            chunker=create_chunker("sliding_window", max_tokens=200, overlap_tokens=30),
            cross_lingual=NoopCrossLingual(),
            embedder=self.embedder,
            repository=self.repo,
        )
        self.retriever = HybridRetriever(self.es)

    def ingest(self, name: str, body: str | bytes, **kwargs: Any) -> Dict[str, Any]:
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        return self.ingestor.ingest(name, raw, **kwargs)

    def search(self, query: str, roles: Sequence[str] = ("__public__",), top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.search(query, self.embedder.embed(query), list(roles), top_k=top_k).authorized_chunks


@pytest.fixture
def stack(tmp_path: Path) -> Stack:
    return Stack(tmp_path)


# --- Complex formats: Excel ---------------------------------------------------


@pytest.fixture(scope="module")
def workbook():
    return XlsxDocumentParser().extract_document(XLSX.read_bytes())


def _budget_block(workbook) -> str:
    return next(b.text for b in workbook.blocks if b.metadata.get("sheet") == "Budget")


def test_excel_merged_cell_value_reaches_every_row(workbook):
    budget = _budget_block(workbook)
    assert "| Asia | Hong Kong |" in budget
    assert "| Asia | Singapore |" in budget  # Merged A4:A5: flat parsers leave this blank.


def test_excel_spanning_header_becomes_compound_column_names(workbook):
    assert "Q1 USD Budget" in _budget_block(workbook)
    assert "Q1 HKD Budget" in _budget_block(workbook)


def test_excel_indexes_formula_results_not_formula_text(workbook):
    row = next(line for line in _budget_block(workbook).splitlines() if "Singapore" in line)
    assert "390000" in row
    assert "=C5" not in row


def test_excel_records_cross_sheet_lineage(workbook):
    assert "D5 `=C5*Rates!B2` depends on Rates" in _budget_block(workbook)


def test_excel_uncalculated_formula_is_flagged_not_blank(tmp_path: Path):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(["Item", "Total"])
    sheet.append(["A", "=1+1"])  # openpyxl writes no cached value, like some ETL exports.
    path = tmp_path / "uncached.xlsx"
    book.save(path)
    text = XlsxDocumentParser().extract_document(path.read_bytes()).blocks[0].text
    assert "=1+1" in text
    assert "never calculated" in text


def test_excel_hidden_sheet_is_not_indexed(workbook):
    everything = "\n".join(b.text for b in workbook.blocks)
    assert "Salaries_Confidential" not in everything
    assert "999999" not in everything


def test_excel_large_sheet_becomes_one_summary_not_thousands_of_chunks(workbook):
    summaries = [b for b in workbook.blocks if b.block_type == "table_summary"]
    assert len(summaries) == 1
    assert "2,000 rows" in summaries[0].text
    assert "Amount USD" in summaries[0].text
    assert len(workbook.tables) == 1 and len(workbook.tables[0].rows) == 2000


def test_excel_table_blocks_repeat_header_per_batch(tmp_path: Path):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(["Code", "Value"])
    for number in range(30):
        sheet.append([f"C{number}", number])
    path = tmp_path / "wide.xlsx"
    book.save(path)
    blocks = XlsxDocumentParser(rows_per_block=10).extract_document(path.read_bytes()).blocks
    assert len(blocks) == 3
    assert all("| Code | Value |" in block.text for block in blocks)


def test_large_table_structured_query_aggregates_and_checks_acl(stack: Stack):
    report = stack.ingest("regional_budget.xlsx", XLSX.read_bytes(), tags=["finance"])
    table_id = report["table_ids"][0]
    store = stack.repo.tabular_store

    total = store.query(
        TableQuery(table_id=table_id, aggregate="sum", aggregate_column="Amount USD"),
        ["finance"],
    )
    assert total["rows"] == [[sum(row[3] for row in XlsxDocumentParser().extract_document(XLSX.read_bytes()).tables[0].rows)]]

    grouped = store.query(
        TableQuery(
            table_id=table_id,
            aggregate="count",
            group_by="Region",
            filters=[TableFilter(column="Category", op="eq", value="Travel")],
        ),
        ["finance"],
    )
    assert {row[0] for row in grouped["rows"]} == {"Asia", "Europe", "Americas"}

    with pytest.raises(PermissionError):
        store.query(TableQuery(table_id=table_id), ["__public__", "operation"])


@pytest.mark.parametrize(
    "bad_column",
    ['Amount USD"; DROP TABLE table_catalog; --', "c0", "*"],
)
def test_table_query_rejects_identifiers_outside_the_schema(stack: Stack, bad_column: str):
    report = stack.ingest("regional_budget.xlsx", XLSX.read_bytes(), tags=["finance"])
    with pytest.raises(ValueError, match="Unknown column"):
        stack.repo.tabular_store.query(
            TableQuery(table_id=report["table_ids"][0], select=[bad_column]), ["finance"]
        )


def test_table_filter_value_is_parameterized_not_interpolated(stack: Stack):
    report = stack.ingest("regional_budget.xlsx", XLSX.read_bytes(), tags=["finance"])
    table_id = report["table_ids"][0]
    injected = "x' OR '1'='1"
    result = stack.repo.tabular_store.query(
        TableQuery(table_id=table_id, filters=[TableFilter(column="Region", value=injected)]), ["finance"]
    )
    assert result["rows"] == []


class _PlanningLlm:
    """Stands in for the router: returns a fixed plan, then echoes the prompt's table result."""

    def __init__(self, plan: TableQuery) -> None:
        self.plan = plan
        self.prompts: List[str] = []

    def with_structured_output(self, schema: Any) -> Any:
        outer = self

        class _Planner:
            def invoke(self, prompt: str) -> TableQuery:
                outer.prompts.append(prompt)
                return outer.plan

        return _Planner()

    def invoke(self, prompt_value: Any) -> str:
        return str(prompt_value)

    def __call__(self, prompt_value: Any) -> str:
        return str(prompt_value)


def test_llm_answer_receives_structured_query_result_as_a_source(stack: Stack):
    report = stack.ingest("regional_budget.xlsx", XLSX.read_bytes(), tags=["finance"])
    table_id = report["table_ids"][0]
    llm = _PlanningLlm(TableQuery(table_id=table_id, aggregate="sum", aggregate_column="Amount USD"))
    synthesizer = GroundedAnswerSynthesizer(llm_factory=lambda **_: llm)
    chunks = stack.search("total Amount USD transactions", roles=["finance"], top_k=4)
    assert any(chunk["block_type"] == "table_summary" for chunk in chunks)

    answer = synthesizer.synthesize(
        question="What is the total Amount USD across all transactions?",
        expanded_query="total Amount USD across all transactions",
        matched_concepts=[],
        is_cjk_query=False,
        user_roles=["finance"],
        authorized_chunks=chunks,
        masked_restricted_docs=[],
        synthesis_mode="llm",
        table_query_runner=lambda query: stack.repo.tabular_store.query(query, ["finance"]),
    )
    assert synthesizer.last_mode_used == "llm"
    assert "structured query result" in answer
    assert "sum(Amount USD)" in answer
    # The judge must see this too, or a correct tool-based answer scores as invented.
    assert "sum(Amount USD)" in synthesizer.last_context_parts[0]


# --- Multilingual processing ---------------------------------------------------


def test_arabic_pdf_extraction_matches_source_text_exactly():
    parsed = PdfPlumberDocumentParser().extract_document(ARABIC_PDF.read_bytes())
    assert len(parsed.blocks) == 1, "right-aligned Arabic must not be read as two columns"
    assert parsed.blocks[0].text.split("\n") == ARABIC_GROUND_TRUTH
    assert parsed.blocks[0].metadata["direction"] == "rtl"


def test_rtl_repair_keeps_embedded_ltr_run_and_leaves_latin_alone():
    visual = ".ايموي HKD 520,000,000 وه"
    assert repair_rtl_line(visual) == "هو HKD 520,000,000 يوميا."
    assert repair_rtl_line("Plain English 450,000") == "Plain English 450,000"


def test_rtl_repair_handles_lam_alef_ligature_before_folding():
    # Reversal has to come first: NFKC expands U+FEFB to two letters in logical order.
    assert normalize_for_matching(repair_rtl_line("ﻻ")) == "لا"


def test_two_column_rtl_page_reads_right_column_first():
    from src.rag.extractor import _cluster_columns

    def word(text: str, x0: float, top: float) -> Dict[str, object]:
        return {"text": text, "x0": x0, "x1": x0 + 40, "top": top, "bottom": top + 10}

    words = [word("يمين", 400 + i * 45, 100 + j * 20) for i in range(2) for j in range(3)]
    words += [word("يسار", 40 + i * 45, 100 + j * 20) for i in range(2) for j in range(3)]
    blocks = _cluster_columns(words, 600.0)
    assert [block.metadata["column"] for block in blocks] == ["right", "left"]


def test_tashkeel_and_presentation_forms_normalize_to_the_same_form():
    assert normalize_for_matching("الْحَدُّ") == "الحد"
    assert normalize_for_matching("ﻟﺎ") == normalize_for_matching("لا")[:0] + "لا"[0] + "ا"
    assert normalize_for_matching("٤٥٠") == "450"


def test_presentation_forms_are_not_classified_as_cjk():
    assert not has_cjk("ﺎﻠﻟ")
    assert has_cjk("现金")


def test_tokenizer_segments_chinese_and_keeps_arabic_words_whole():
    tokens = MultilingualTokenizer().split_into_tokens("最低现金储备 الْحَدُّ الأدنى؟ HKD 450,000,000")
    assert "现金" in tokens and "储备" in tokens
    assert "الْحَدُّ" in tokens, "diacritics must not fragment the word"
    assert "الأدنى" in tokens and "؟" in tokens
    assert "450,000,000" in tokens


def test_sentence_split_handles_stops_without_following_space():
    assert split_sentences("第一句。第二句！هل هذا صحيح؟ نعم. Rate is 3.5 today.") == [
        "第一句。",
        "第二句！",
        "هل هذا صحيح؟",
        "نعم.",
        "Rate is 3.5 today.",
    ]


def test_unspaced_chinese_respects_max_tokens_and_adds_no_spaces():
    config = ChunkingConfig(max_tokens=100, overlap_tokens=15)
    chunks = SlidingWindowChunker(config).split("本政策规定了公司的现金管理要求。" * 60)
    tokenizer = MultilingualTokenizer()
    assert len(chunks) > 1, "this text used to come back as one oversized chunk"
    assert all(tokenizer.count_tokens(chunk) <= 100 for chunk in chunks)
    assert all(" " not in chunk for chunk in chunks)


def test_arabic_gets_a_real_vector_and_tashkeel_does_not_change_it():
    embedder = CrossLingualEmbedder(FeatureHashEmbeddingProvider(dimensions=256))
    plain, with_marks = embedder.embed("الحد الأدنى"), embedder.embed("الْحَدُّ الأَدْنَى")
    assert sum(1 for value in plain if value) > 1, "Arabic used to embed to a placeholder vector"
    assert plain == with_marks


def test_chinese_query_finds_chinese_chunk_by_shared_terms(stack: Stack):
    stack.ingest("travel_zh.md", "员工出差住宿费用上限为每晚 HKD 1,800。".encode("utf-8"))
    stack.ingest("noise.md", "Unrelated English content about office plants and watering.".encode("utf-8"))
    assert stack.search("住宿费用上限", top_k=1)[0]["filename"] == "travel_zh.md"


def test_scanned_page_goes_through_ocr_engine():
    class FakeEngine:
        name = "fake-ocr"

        def recognize(self, image: Any) -> List[OcrLine]:
            assert image.width > 0
            return [
                OcrLine("السطر الثاني", 0.9, (10, 60, 200, 80)),
                OcrLine("السطر الأول", 0.8, (10, 20, 200, 40)),
            ]

    parsed = PdfPlumberDocumentParser(ocr_engine=FakeEngine()).extract_document(ARABIC_SCAN.read_bytes())
    block = parsed.blocks[0]
    assert block.block_type == "ocr_text"
    assert block.text == "السطر الأول\nالسطر الثاني"
    assert block.metadata["ocr_engine"] == "fake-ocr"
    assert block.metadata["ocr_mean_confidence"] == pytest.approx(0.85)


def test_scanned_page_without_engine_warns_instead_of_vanishing_silently(caplog):
    PdfPlumberDocumentParser().extract_document(ARABIC_SCAN.read_bytes())
    assert "NOT indexed" in caplog.text


def test_ocr_fragments_on_one_row_read_right_to_left_on_rtl_pages():
    lines = [
        OcrLine("هذا", 0.9, (300, 10, 400, 30)),
        OcrLine("سطر", 0.9, (150, 12, 280, 32)),
    ]
    assert order_ocr_lines(lines) == "هذا سطر"


# --- Semantic chunking ---------------------------------------------------------


class TopicEmbedder:
    """Sentences about one topic get one direction, so boundaries are known exactly."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> List[float]:
        self.calls += 1
        if "cash" in text or "liquidity" in text:
            return [1.0, 0.05, 0.0]
        if "password" in text or "login" in text:
            return [0.0, 1.0, 0.05]
        return [0.05, 0.0, 1.0]


CASH = [f"The cash buffer rule number {n} sets a liquidity limit." for n in range(6)]
LOGIN = [f"The password rule number {n} changes each login flow." for n in range(6)]
TRAVEL = [f"The hotel rule number {n} applies on every trip abroad." for n in range(6)]


def _semantic(max_tokens: int = 200, **kwargs: Any) -> SemanticChunker:
    config = ChunkingConfig(strategy="semantic", max_tokens=max_tokens, overlap_tokens=20)
    return SemanticChunker(config, TopicEmbedder(), **kwargs)


def test_semantic_chunker_splits_at_topic_boundaries():
    chunker = _semantic(overlap_ratio=0.0)
    chunks = chunker.split(" ".join(CASH + LOGIN + TRAVEL))
    assert len(chunks) == 3
    assert "password" not in chunks[0] and "hotel" not in chunks[0]
    assert "cash" not in chunks[1] and "hotel" not in chunks[1]
    assert "cash" not in chunks[2] and "password" not in chunks[2]


def test_semantic_chunker_overlap_carries_previous_context_within_budget():
    chunks = _semantic(overlap_ratio=0.2).split(" ".join(CASH + LOGIN))
    assert len(chunks) == 2
    assert CASH[-1] in chunks[1], "the last sentence before the boundary should carry over"
    assert CASH[0] not in chunks[1]
    assert chunks[1].endswith(LOGIN[-1])


def test_semantic_chunker_does_not_fragment_a_single_long_topic():
    # With one topic, similarity is flat: there are no valleys to split at.
    chunks = _semantic().split(" ".join(CASH * 3))
    assert len(chunks) == 1


def test_semantic_chunker_min_size_stops_noise_fragmentation():
    # Every other sentence switches topic, which would split at nearly every gap.
    mixed = [sentence for pair in zip(CASH, LOGIN) for sentence in pair]
    chunks = _semantic(max_tokens=200, window=1, min_chunk_ratio=0.5).split(" ".join(mixed))
    tokenizer = MultilingualTokenizer()
    assert all(tokenizer.count_tokens(chunk) >= 100 for chunk in chunks[:-1])


def test_semantic_chunker_enforces_hard_cap_on_chunk_size():
    chunker = _semantic(max_tokens=40)
    chunks = chunker.split(" ".join(CASH * 4))
    tokenizer = MultilingualTokenizer()
    assert len(chunks) > 1
    assert all(tokenizer.count_tokens(chunk) <= 40 for chunk in chunks)


def test_semantic_chunker_reports_its_latency_cost():
    chunker = _semantic()
    chunker.split(" ".join(CASH + LOGIN))
    assert chunker.last_stats.embed_calls == 12, "one inference per sentence is the cost being traded"
    assert chunker.last_stats.boundaries == [5]
    assert len(chunker.last_stats.similarities) == 11


def test_valley_depth_finds_a_shift_a_global_threshold_would_miss():
    from src.rag.chunker import _valley_depths

    # A shift at index 2 inside a tight topic, and a lower but gently sloping region later.
    similarities = [0.80, 0.82, 0.70, 0.82, 0.80, 0.62, 0.60, 0.61, 0.62]
    depths = _valley_depths(similarities)
    assert depths[2] == pytest.approx(0.24)
    assert depths[2] > depths[7], "index 7 has a lower similarity but is not a real valley"


def test_semantic_chunker_requires_an_embedder():
    with pytest.raises(ValueError, match="requires an embedder"):
        create_chunker("semantic", max_tokens=100, overlap_tokens=10)


def test_semantic_chunker_handles_chinese_without_spaces():
    embedder = TopicEmbedder()
    text = "现金cash规则第一条。现金cash规则第二条。现金cash规则第三条。酒店规则第一条。酒店规则第二条。酒店规则第三条。"
    config = ChunkingConfig(strategy="semantic", max_tokens=100, overlap_tokens=10)
    chunks = SemanticChunker(config, embedder, min_chunk_ratio=0.01, overlap_ratio=0.0).split(text)
    assert len(chunks) == 2
    assert all(" " not in chunk for chunk in chunks)


def test_adaptive_router_saves_inference_on_uniform_documents():
    embedder = TopicEmbedder()
    chunker = AdaptiveChunker(ChunkingConfig(strategy="adaptive", max_tokens=60, overlap_tokens=10), embedder)
    log_file = "\n".join(f"item-{n}: value {n} recorded by the nightly job and stored safely." for n in range(40))
    chunker.split(log_file)
    assert chunker.last_decision["strategy"] == "fixed"
    assert "list-like" in chunker.last_decision["reason"]
    assert embedder.calls == 0, "the cheap path must not touch the model"


def test_adaptive_router_uses_semantic_for_long_prose():
    embedder = TopicEmbedder()
    chunker = AdaptiveChunker(ChunkingConfig(strategy="adaptive", max_tokens=60, overlap_tokens=10), embedder)
    chunker.split(" ".join(CASH + LOGIN))
    assert chunker.last_decision["strategy"] == "semantic"
    assert embedder.calls == 12


def test_adaptive_routing_decision_is_recorded_on_chunks(tmp_path: Path):
    stack = Stack(tmp_path)
    stack.ingestor._chunker = AdaptiveChunker(
        ChunkingConfig(strategy="adaptive", max_tokens=60, overlap_tokens=10), stack.embedder
    )
    stack.ingest("mixed.md", " ".join(CASH + LOGIN))
    assert stack.es.memory_chunks[0]["metadata"]["chunking"]["strategy"] == "semantic"


# --- Versioning and ground-truth decay -----------------------------------------


def _policy(year: int, days: int) -> str:
    return f"IT Security Policy {year}. Employees must rotate their network password every {days} days."


def test_every_chunk_carries_lifecycle_metadata(stack: Stack):
    stack.ingest(
        "policy.md",
        _policy(2026, 180),
        governance=DocumentGovernance(
            trust_tier="official_policy",
            effective_date="2026-01-01",
            expires_at="2099-01-01",
            business_domain="it",
        ),
    )
    chunk = stack.es.memory_chunks[0]
    assert chunk["trust_tier"] == "official_policy"
    assert chunk["effective_date"] == "2026-01-01"
    assert chunk["expires_at"] == "2099-01-01"
    assert chunk["business_domain"] == "it"
    assert chunk["source_key"] == "default:policy.md"


def test_expired_document_is_not_retrieved(stack: Stack):
    stack.ingest("old.md", _policy(2023, 90), governance=DocumentGovernance(expires_at="2024-12-31"))
    stack.ingest("new.md", _policy(2026, 180))
    assert [chunk["filename"] for chunk in stack.search("rotate network password")] == ["new.md"]


def test_document_stays_valid_through_its_expiry_date():
    assert not is_expired("2026-11-05", on="2026-11-05")
    assert is_expired("2026-11-05", on="2026-11-06")
    assert not is_expired(None)


def test_superseded_version_stays_stored_but_out_of_retrieval(stack: Stack):
    stack.ingest("a.pdf", POLICY_V1.read_bytes(), source_key="treasury")
    stack.ingest("b.pdf", POLICY_V2.read_bytes(), source_key="treasury")
    assert any(chunk["ingestion_status"] == "SUPERSEDED" for chunk in stack.es.memory_chunks)
    blob = " ".join(chunk["content"] for chunk in stack.search("intraday cash buffer", top_k=10))
    assert "520,000,000" in blob and "450,000,000" not in blob


def test_trust_tier_outranks_a_better_text_match():
    ranked = apply_trust_ranking(
        [
            {"score": 1.0, "trust_tier": "draft_notes", "filename": "draft"},
            {"score": 0.8, "trust_tier": "official_policy", "filename": "official"},
        ]
    )
    assert ranked[0]["filename"] == "official"
    assert ranked[0]["raw_score"] == 0.8


def test_equal_score_tie_goes_to_later_effective_date():
    ranked = apply_trust_ranking(
        [
            {"score": 1.0, "trust_tier": "reviewed", "effective_date": "2023-01-01", "filename": "old"},
            {"score": 1.0, "trust_tier": "reviewed", "effective_date": "2026-01-01", "filename": "new"},
        ]
    )
    assert ranked[0]["filename"] == "new"


def _chunk(name: str, content: str, tier: str, effective: str) -> Dict[str, Any]:
    return {
        "chunk_id": name,
        "filename": name,
        "source_key": name,
        "content": content,
        "trust_tier": tier,
        "effective_date": effective,
        "page_number": 1,
        "tags": [],
        "is_public": True,
        "score": 1.0,
    }


def test_conflict_between_sources_is_resolved_by_trust_tier():
    conflicts = detect_conflicts(
        [
            _chunk("draft.md", _policy(2026, 30), "draft_notes", "2026-09-01"),
            _chunk("official.md", _policy(2026, 180), "official_policy", "2026-01-01"),
        ]
    )
    assert len(conflicts) == 1
    assert conflicts[0]["winner"]["filename"] == "official.md", "tier beats a newer date"
    assert conflicts[0]["reason"].startswith("higher trust tier")


def test_conflict_at_equal_tier_is_resolved_by_date():
    conflicts = detect_conflicts(
        [
            _chunk("old.md", _policy(2023, 90), "official_policy", "2023-01-01"),
            _chunk("new.md", _policy(2026, 180), "official_policy", "2026-01-01"),
        ]
    )
    assert conflicts[0]["winner"]["filename"] == "new.md"
    assert "later effective date" in conflicts[0]["reason"]


def test_unresolvable_conflict_names_no_winner():
    conflicts = detect_conflicts(
        [
            _chunk("a.md", _policy(2026, 90), "reviewed", "2026-01-01"),
            _chunk("b.md", _policy(2026, 180), "reviewed", "2026-01-01"),
        ]
    )
    assert conflicts[0]["winner"] is None


def test_no_conflict_when_topics_differ_or_values_agree():
    assert not detect_conflicts(
        [
            _chunk("a.md", "Hotel cap is HKD 1,800 per night on business trips.", "reviewed", "2026-01-01"),
            _chunk("b.md", _policy(2026, 180), "reviewed", "2026-01-01"),
        ]
    )
    assert not detect_conflicts(
        [
            _chunk("a.md", _policy(2026, 180), "reviewed", "2026-01-01"),
            _chunk("b.md", _policy(2025, 180), "official_policy", "2025-01-01"),
        ]
    )


def test_two_versions_of_one_source_are_not_reported_as_a_conflict():
    first, second = _chunk("a.md", _policy(2023, 90), "reviewed", "2023-01-01"), _chunk("a.md", _policy(2026, 180), "reviewed", "2026-01-01")
    second["chunk_id"] = "a.md#2"
    assert not detect_conflicts([first, second])


def test_fast_answer_quotes_the_winner_and_declares_the_conflict():
    synthesizer = GroundedAnswerSynthesizer()
    answer = synthesizer.synthesize(
        question="How often must I rotate my network password?",
        expanded_query="rotate network password",
        matched_concepts=[],
        is_cjk_query=False,
        user_roles=[],
        authorized_chunks=[
            _chunk("draft.md", _policy(2026, 30), "draft_notes", "2026-09-01"),
            _chunk("official.md", _policy(2026, 180), "official_policy", "2026-01-01"),
        ],
        masked_restricted_docs=[],
    )
    assert "180 days" in answer
    assert "Conflicting sources" in answer
    assert "higher trust tier" in answer


def test_llm_prompt_carries_tier_dates_and_precedence_rules():
    captured: List[Any] = []

    def fake_model(prompt_value: Any) -> str:
        captured.append(prompt_value.to_string())
        return "answer"

    synthesizer = GroundedAnswerSynthesizer(llm_factory=lambda **_: fake_model)
    synthesizer.synthesize(
        question="How often must I rotate my network password?",
        expanded_query="rotate network password",
        matched_concepts=[],
        is_cjk_query=False,
        user_roles=[],
        authorized_chunks=[_chunk("official.md", _policy(2026, 180), "official_policy", "2026-01-01")],
        masked_restricted_docs=[],
        synthesis_mode="llm",
    )
    assert "Trust tier: official_policy" in captured[0]
    assert "Effective: 2026-01-01" in captured[0]
    assert SOURCE_PRECEDENCE_RULES.splitlines()[0] in captured[0]
    assert "Never blend" in captured[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trust_tier": "gospel"},
        {"effective_date": "next week"},
        {"effective_date": "2026-02-01", "expires_at": "2026-01-01"},
    ],
)
def test_invalid_governance_is_rejected_at_ingestion(kwargs: Dict[str, str]):
    with pytest.raises(ValueError):
        DocumentGovernance(**kwargs)


# --- GDPR and data deletion ----------------------------------------------------


class ConceptStub:
    def __init__(self, stored: Dict[str, List[str]] | None = None) -> None:
        self.stored = stored or {}

    def forget(self, concept_ids: List[str], *, subject_id: str | None = None) -> int:
        removed = [key for key, ids in self.stored.items() if set(ids) & set(concept_ids)]
        for key in removed:
            del self.stored[key]
        return len(removed)

    def count_mentions(self, subject_id: str) -> int:
        return 0


def _erasure(stack: Stack, tmp_path: Path) -> ErasureService:
    return ErasureService(stack.repo, concept_registry=ConceptStub(), audit_log_path=tmp_path / "audit.jsonl")


def test_emails_in_text_are_detected_as_subjects():
    assert detect_subject_ids("Owner: Maya.Chen@Example.com, cc x@y.org. Not an email: a@b") == [
        "maya.chen@example.com",
        "x@y.org",
    ]


def test_every_chunk_is_tagged_with_lineage(stack: Stack):
    stack.ingest("contract.md", f"Contract owner {PERSON}.", governance=DocumentGovernance(subject_ids=("EMP-1",)))
    chunk = stack.es.memory_chunks[0]
    assert chunk["source_key"] == "default:contract.md"
    assert set(chunk["subject_ids"]) == {"emp-1", PERSON}


def test_erasure_reaches_every_retained_version(stack: Stack, tmp_path: Path):
    # The trap: delete by doc_id removes v2 and silently leaves superseded v1.
    stack.ingest("hr.md", f"Review of {PERSON}: rating 3.", source_key="hr-file", governance=DocumentGovernance(subject_ids=(PERSON,)))
    stack.ingest("hr.md", f"Review of {PERSON}: rating 4.", source_key="hr-file", governance=DocumentGovernance(subject_ids=(PERSON,)))
    assert len(stack.repo.versions_for_sources(["hr-file"])) == 2

    service = _erasure(stack, tmp_path)
    receipt = service.execute(service.plan_subject(PERSON))

    assert receipt.verified, receipt.residuals
    assert receipt.removed["versions"] == 2
    assert stack.es.memory_chunks == []
    assert list((tmp_path / "uploads").glob("*")) == []
    assert stack.repo.versions_for_sources(["hr-file"]) == []


def test_plain_delete_by_doc_id_would_have_left_old_version_behind(stack: Stack):
    """Documents why erasure uses lineage: this is what the older delete path does."""
    stack.ingest("hr.md", f"Review of {PERSON}: rating 3.", source_key="hr-file")
    latest = stack.ingest("hr.md", f"Review of {PERSON}: rating 4.", source_key="hr-file")
    stack.repo.delete_document(latest["doc_id"])
    assert any(PERSON in chunk["content"] for chunk in stack.es.memory_chunks)


def test_erasure_removes_large_table_rows_too(stack: Stack, tmp_path: Path):
    report = stack.ingest("regional_budget.xlsx", XLSX.read_bytes(), tags=["finance"], source_key="budget")
    assert stack.repo.tabular_store.describe(report["table_ids"][0]) is not None
    service = _erasure(stack, tmp_path)
    receipt = service.execute(service.plan_source("budget"))
    assert receipt.removed["tables"] == 1
    assert stack.repo.tabular_store.describe(report["table_ids"][0]) is None
    assert receipt.verified


def test_erasure_leaves_other_people_untouched(stack: Stack, tmp_path: Path):
    stack.ingest("a.md", f"File for {PERSON}.", governance=DocumentGovernance(subject_ids=(PERSON,)))
    stack.ingest("b.md", "File for other@example.com.", governance=DocumentGovernance(subject_ids=("other@example.com",)))
    service = _erasure(stack, tmp_path)
    service.execute(service.plan_subject(PERSON))
    assert [chunk["filename"] for chunk in stack.es.memory_chunks] == ["b.md"]


def test_plan_is_a_dry_run_and_separates_declared_from_detected(stack: Stack, tmp_path: Path):
    stack.ingest("owned.md", "Employee file.", governance=DocumentGovernance(subject_ids=(PERSON,)))
    stack.ingest("minutes.md", f"Action item assigned to {PERSON} at the review.")
    plan = _erasure(stack, tmp_path).plan_subject(PERSON)
    assert plan.match_type == {"default:owned.md": "declared", "default:minutes.md": "detected"}
    assert len(stack.es.memory_chunks) == 2, "planning must not delete anything"


def test_keeping_a_detected_document_is_reported_as_incomplete(stack: Stack, tmp_path: Path):
    stack.ingest("owned.md", "Employee file.", governance=DocumentGovernance(subject_ids=(PERSON,)))
    stack.ingest("minutes.md", f"Action item assigned to {PERSON} at the review.")
    service = _erasure(stack, tmp_path)
    receipt = service.execute(service.plan_subject(PERSON, include_detected=False))
    assert not receipt.verified, "the raw file for minutes.md still names the person"
    assert receipt.residuals["sources_still_linked"] == 1


def test_mention_that_lineage_missed_fails_verification(stack: Stack, tmp_path: Path):
    stack.ingest("owned.md", "Employee file.", governance=DocumentGovernance(subject_ids=(PERSON,)))
    # Written directly: an old chunk from before lineage tracking existed.
    stack.es.memory_chunks.append(
        {
            "chunk_id": "legacy",
            "doc_id": "legacy",
            "source_key": "legacy",
            "content": f"Old note about {PERSON}",
            "subject_ids": [],
        }
    )
    service = _erasure(stack, tmp_path)
    plan = service.plan_subject(PERSON)
    assert plan.residual_mentions == 1, "the plan warns about it"
    receipt = service.execute(plan)
    assert not receipt.verified
    assert receipt.residuals["chunk_text_mentions"] == 1


def test_tombstone_blocks_resurrection_by_nightly_resync(stack: Stack, tmp_path: Path):
    body = f"File for {PERSON}."
    stack.ingest("hr.md", body, source_key="hr-file", governance=DocumentGovernance(subject_ids=(PERSON,)))
    service = _erasure(stack, tmp_path)
    service.execute(service.plan_subject(PERSON))

    with pytest.raises(ErasedSourceError, match="source"):
        stack.ingest("hr.md", body, source_key="hr-file")  # Same path.
    with pytest.raises(ErasedSourceError, match="content"):
        stack.ingest("copy.md", body, source_key="renamed")  # Same bytes, new path.
    with pytest.raises(ErasedSourceError, match="data subject"):
        stack.ingest("new.md", "Other text.", source_key="new", governance=DocumentGovernance(subject_ids=(PERSON,)))
    with pytest.raises(ErasedSourceError, match="data subject"):
        stack.ingest("mention.md", f"A fresh mention of {PERSON} in new text.", source_key="fresh")
    assert stack.es.memory_chunks == []


def test_tombstones_survive_restart_and_can_be_lifted_deliberately(stack: Stack, tmp_path: Path):
    stack.ingest("hr.md", "File.", source_key="hr-file")
    service = _erasure(stack, tmp_path)
    service.execute(service.plan_source("hr-file"))

    reopened = DocumentRepository(
        es_manager=stack.es,
        uploads_dir=tmp_path / "uploads",
        registry_path=tmp_path / "registry.json",
    )
    with pytest.raises(ErasedSourceError):
        reopened.assert_not_erased(source_key="hr-file")
    assert reopened.lift_tombstone("hr-file")
    reopened.assert_not_erased(source_key="hr-file")


def test_audit_log_and_registry_never_store_the_identifier(stack: Stack, tmp_path: Path):
    stack.ingest("maya_chen_review.md", f"Review of {PERSON}.", governance=DocumentGovernance(subject_ids=(PERSON,)))
    service = _erasure(stack, tmp_path)
    receipt = service.execute(service.plan_subject(PERSON))
    log = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    registry = (tmp_path / "registry.json").read_text(encoding="utf-8")
    for stored in (log, registry):
        assert PERSON not in stored
        assert "maya_chen_review" not in stored
    assert json.loads(log)["subject_hash"] == receipt.subject_hash
    assert "backups" in receipt.notice


def test_physical_purge_timeout_is_recorded_not_claimed_as_success(stack: Stack, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(stack.repo, "expunge_deleted_documents", lambda: 4)
    service = _erasure(stack, tmp_path)
    assert service.confirm_physical_purge("req-1", max_wait_seconds=0) == 4
    record = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["soft_deleted_remaining"] == 4


def test_identifier_hash_is_keyed(stack: Stack, tmp_path: Path):
    import hashlib

    assert stack.repo.keyed_hash(PERSON) != hashlib.sha256(PERSON.encode()).hexdigest()
    assert (tmp_path / ".erasure_key").stat().st_mode & 0o077 == 0


def test_concept_registry_forgets_derived_data(tmp_path: Path):
    from src.rag.cross_lingual import DynamicConceptRegistry

    registry = DynamicConceptRegistry(tmp_path / "concepts.json")
    registry.register_document_concepts("hash1", "hr.md", [{"concept_id": "concept:1", "en_triggers": ["bonus"]}], "m")
    registry.register_document_concepts("hash2", "other.md", [{"concept_id": "concept:2", "en_triggers": ["travel"]}], "m")
    registry.cache_query_translation(f"what is {PERSON} rating", "rating")
    assert registry.count_mentions(PERSON) == 1

    assert registry.forget(["concept:1"], subject_id=PERSON) == 2
    assert registry.get_document_concepts("hash1") is None
    assert registry.get_document_concepts("hash2") is not None
    assert registry.count_mentions(PERSON) == 0
    assert PERSON not in (tmp_path / "concepts.json").read_text(encoding="utf-8")


def test_source_delete_event_erases_and_retries_are_idempotent(stack: Stack, tmp_path: Path):
    stack.ingest("sp.md", "SharePoint file.", source_key="sharepoint:42")
    handler = SourceSyncHandler(stack.repo, _erasure(stack, tmp_path))
    first = handler.handle(event_id="evt-1", event_type="deleted", source_key="sharepoint:42")
    assert first["status"] == "erased" and first["receipt"]["verified"]
    assert handler.handle(event_id="evt-1", event_type="deleted", source_key="sharepoint:42")["status"] == "duplicate_ignored"
    assert len((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_source_archive_event_keeps_data_but_removes_it_from_retrieval(stack: Stack, tmp_path: Path):
    stack.ingest("sf.md", "Salesforce account record about renewals.", source_key="salesforce:7")
    handler = SourceSyncHandler(stack.repo, _erasure(stack, tmp_path))
    assert handler.handle(event_id="evt-2", event_type="archived", source_key="salesforce:7")["versions"] == 1
    assert stack.search("renewals") == []
    assert len(stack.es.memory_chunks) == 1
    assert stack.repo.versions_for_sources(["salesforce:7"])[0].status == IngestionState.ARCHIVED


def test_unknown_sync_event_type_is_rejected(stack: Stack, tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported"):
        SourceSyncHandler(stack.repo, _erasure(stack, tmp_path)).handle(event_id="e", event_type="purge", source_key="k")


@pytest.fixture
def api() -> Iterator[TestClient]:
    from src.web.app import app

    yield TestClient(app)


@pytest.mark.parametrize("path", ["/api/erasure", "/api/sync/events"])
def test_destructive_endpoints_fail_closed_without_configured_token(api: TestClient, path: str, monkeypatch):
    monkeypatch.delenv("RAG_GOVERNANCE_TOKEN", raising=False)
    payload = {"subject_id": PERSON, "event_id": "e", "event_type": "deleted", "source_key": "k"}
    assert api.post(path, json=payload).status_code == 503


@pytest.mark.parametrize("path", ["/api/erasure", "/api/sync/events"])
def test_destructive_endpoints_reject_wrong_token(api: TestClient, path: str, monkeypatch):
    monkeypatch.setenv("RAG_GOVERNANCE_TOKEN", "right")
    payload = {"subject_id": PERSON, "event_id": "e", "event_type": "deleted", "source_key": "k"}
    assert api.post(path, json=payload, headers={"X-Governance-Token": "wrong"}).status_code == 401


def test_erasure_api_is_dry_run_by_default(api: TestClient, stack: Stack, tmp_path: Path, monkeypatch):
    stack.ingest("hr.md", f"File for {PERSON}.", governance=DocumentGovernance(subject_ids=(PERSON,)))

    class Service:
        def erasure_service(self) -> ErasureService:
            return _erasure(stack, tmp_path)

    monkeypatch.setenv("RAG_GOVERNANCE_TOKEN", "right")
    # "src.web.app" resolves to the FastAPI object re-exported by the package.
    monkeypatch.setattr(sys.modules["src.web.app"], "get_kb_service", lambda: Service())
    headers = {"X-Governance-Token": "right"}

    response = api.post("/api/erasure", json={"subject_id": PERSON}, headers=headers)
    assert response.json()["status"] == "dry_run"
    assert len(stack.es.memory_chunks) == 1

    response = api.post("/api/erasure", json={"subject_id": PERSON, "dry_run": False}, headers=headers)
    assert response.json()["status"] == "erased"
    assert stack.es.memory_chunks == []

    assert api.post("/api/erasure", json={}, headers=headers).status_code == 422


# --- Continuous evaluation -----------------------------------------------------


def test_context_precision_rewards_relevant_chunks_at_high_ranks():
    assert context_precision([True, False, False]) == 1.0
    assert context_precision([False, True, False]) == 0.5
    assert context_precision([False, False, False]) == 0.0
    assert context_precision([True, False, True]) == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-4)


def test_value_matching_ignores_formatting():
    assert contains_value("It is HKD 520,000,000 daily", "520000000")
    assert contains_value("الحد ٥٢٠", "520")
    assert not contains_value("It is HKD 450,000,000", "520,000,000")


def _case(case_id: str = "c", **kwargs: Any) -> CaseResult:
    values = {"context_precision": 1.0, "faithfulness": 1.0, "answer_correct": True}
    values.update(kwargs)
    return CaseResult(case_id=case_id, question="q", **values)


THRESHOLDS = {"context_precision": 0.7, "faithfulness": 0.8, "answer_correct": 0.75, "retired_fact_leaks": 0}


def _report(cases: List[CaseResult]):
    return build_report(cases, THRESHOLDS, backend="memory", manifest="m", drift="none")


def test_gate_passes_when_all_metrics_clear_their_floors():
    assert _report([_case()]).status == PASSED


@pytest.mark.parametrize(
    "bad",
    [
        {"context_precision": 0.5},
        {"faithfulness": 0.5},
        {"answer_correct": False},
        {"retired_fact_leaks": ["450,000,000"]},
    ],
)
def test_gate_fails_on_each_metric_independently(bad: Dict[str, Any]):
    report = _report([_case(**bad)])
    assert report.status == FAILED
    assert report.failures


def test_gate_is_inconclusive_when_any_case_could_not_be_judged():
    report = _report([_case(), _case("dead", error="judge unavailable")])
    assert report.status == INCONCLUSIVE, "one healthy case must not hide a dead judge"


def test_gate_is_inconclusive_with_no_cases():
    assert _report([]).status == INCONCLUSIVE


class AnswerKeyJudge:
    """Offline judge: relevance by the expected value, claims by literal support."""

    def __init__(self, dataset: Dict[str, Any]) -> None:
        self._expected = {case["expected_answer"]: case["must_contain_any"] for case in dataset["cases"]}

    def is_relevant(self, question: str, expected_answer: str, passage: str) -> bool:
        return any(contains_value(passage, value) for value in self._expected[expected_answer])

    def judge_claims(self, answer: str, sources: Sequence[str]) -> List[ClaimVerdict]:
        supported = any(contains_value(" ".join(sources), token) for token in answer.split() if token[:1].isdigit())
        return [ClaimVerdict(claim=answer, supported=supported)]


class EchoFirstSourceLlm:
    """Offline answer model: repeats the source its prompt lists first."""

    def with_structured_output(self, schema: Any) -> Any:
        raise RuntimeError("no table planning offline")

    def __call__(self, prompt_value: Any) -> str:
        return prompt_value.to_string().split("Content:", 1)[-1][:600]


def _offline_dataset(monkeypatch):
    dataset = json.loads((Path(__file__).parents[1] / "eval" / "golden_dataset.json").read_text(encoding="utf-8"))
    dataset["cases"] = [case for case in dataset["cases"] if case["id"] == "versioned-table-value"]
    monkeypatch.setattr(
        "src.eval.harness.GroundedAnswerSynthesizer",
        lambda: GroundedAnswerSynthesizer(llm_factory=lambda **_: EchoFirstSourceLlm()),
    )
    return dataset


def test_gate_catches_a_stale_index_regression_end_to_end(monkeypatch, tmp_path: Path):
    dataset = _offline_dataset(monkeypatch)
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    clean = run_evaluation(AnswerKeyJudge(dataset), es_url="http://non-existent-es:9200", dataset_path=dataset_path)
    assert clean.status == PASSED, clean.failures
    assert clean.scores["retired_fact_leaks"] == 0

    drifted = run_evaluation(
        AnswerKeyJudge(dataset), es_url="http://non-existent-es:9200", dataset_path=dataset_path, drift="stale_index"
    )
    assert drifted.status == FAILED
    assert any("450,000,000" in failure for failure in drifted.failures)


def test_dead_judge_makes_the_run_inconclusive_not_green(monkeypatch, tmp_path: Path):
    dataset = _offline_dataset(monkeypatch)
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    def broken_llm(**_: Any) -> Any:
        raise ConnectionError("no model reachable")

    report = run_evaluation(LlmJudge(llm_factory=broken_llm), es_url="http://non-existent-es:9200", dataset_path=dataset_path)
    assert report.status == INCONCLUSIVE


def test_silent_fallback_from_llm_synthesis_is_inconclusive(monkeypatch, tmp_path: Path):
    dataset = _offline_dataset(monkeypatch)
    monkeypatch.setattr(
        "src.eval.harness.GroundedAnswerSynthesizer",
        lambda: GroundedAnswerSynthesizer(llm_factory=lambda **_: (_ for _ in ()).throw(ConnectionError("down"))),
    )
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    report = run_evaluation(AnswerKeyJudge(dataset), es_url="http://non-existent-es:9200", dataset_path=dataset_path)
    assert report.status == INCONCLUSIVE
    assert "fell back" in report.failures[0]


def test_workflow_runs_gate_after_merge_with_required_secret():
    text = (Path(__file__).parents[1] / ".github" / "workflows" / "rag-eval.yml").read_text(encoding="utf-8")
    assert "branches: [main]" in text
    assert "secrets.GOOGLE_API_KEY" in text
    assert "--require-gemini" in text and "--require-elasticsearch" in text


# --- Against a real Elasticsearch ----------------------------------------------


def _es_url() -> str:
    return os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")


@pytest.fixture
def live_stack(tmp_path: Path) -> Iterator[Stack]:
    index = f"beyondrag-test-{uuid.uuid4().hex[:10]}"
    live = Stack(tmp_path, es_url=_es_url(), index_name=index)
    if not live.es.is_online():
        pytest.skip("Elasticsearch is not reachable")
    try:
        yield live
    finally:
        live.es.client.indices.delete(index=index, ignore_unavailable=True)


def test_live_es_filters_expired_and_archived_in_bm25_and_knn(live_stack: Stack, tmp_path: Path):
    live_stack.ingest("old.md", _policy(2023, 90), governance=DocumentGovernance(expires_at="2024-12-31"))
    live_stack.ingest("new.md", _policy(2026, 180))
    live_stack.ingest("archived.md", _policy(2025, 120), source_key="sp:1")
    SourceSyncHandler(live_stack.repo, _erasure(live_stack, tmp_path)).handle(
        event_id="e1", event_type="archived", source_key="sp:1"
    )
    assert {c["filename"] for c in live_stack.search("rotate network password policy")} == {"new.md"}


def test_live_es_analyzer_matches_arabic_with_or_without_diacritics(live_stack: Stack):
    live_stack.ingest("ar.pdf", ARABIC_PDF.read_bytes())
    live_stack.ingest("noise.md", "Unrelated English content about office plants.")
    for query in ("الحد الأدنى للسيولة", "الْحَدُّ الأَدْنَى لِلسُّيُولَة"):
        assert live_stack.search(query, top_k=1)[0]["filename"] == "ar.pdf"
    analyzed = live_stack.es.client.indices.analyze(
        index=live_stack.es.index_name, analyzer="multilingual", text="الْحَدُّ ٤٥٠"
    )
    assert [token["token"] for token in analyzed["tokens"]] == ["الحد", "450"]


def test_live_es_analyzer_uses_han_bigrams(live_stack: Stack):
    analyzed = live_stack.es.client.indices.analyze(
        index=live_stack.es.index_name, analyzer="multilingual", text="现金储备"
    )
    assert [token["token"] for token in analyzed["tokens"]] == ["现金", "金储", "储备"]


def test_live_es_erasure_removes_all_versions_and_passes_verification(live_stack: Stack, tmp_path: Path):
    governance = DocumentGovernance(subject_ids=(PERSON,))
    live_stack.ingest("hr.md", f"Review of {PERSON}: rating 3.", source_key="hr", governance=governance)
    live_stack.ingest("hr.md", f"Review of {PERSON}: rating 4.", source_key="hr", governance=governance)
    live_stack.ingest("other.md", "A separate document about office plants.", source_key="other")

    service = _erasure(live_stack, tmp_path)
    plan = service.plan_subject(PERSON)
    assert plan.chunk_count == 2
    receipt = service.execute(plan)

    assert receipt.verified, receipt.residuals
    assert receipt.removed["chunks"] == 2
    client, index = live_stack.es.client, live_stack.es.index_name
    assert client.count(index=index, query={"match_phrase": {"content": PERSON}})["count"] == 0
    assert client.count(index=index)["count"] == 1

    # Hidden from search is not gone from disk: ES holds deleted text under a retention lease.
    assert receipt.soft_deleted_pending > 0
    assert service.confirm_physical_purge(receipt.request_id, max_wait_seconds=120, poll_seconds=5) == 0
    stats = client.indices.stats(index=index, metric="docs")["indices"][index]["primaries"]["docs"]
    assert stats["deleted"] == 0
    assert stats["count"] == 1
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "physical_purge_check"
    assert records[-1]["soft_deleted_remaining"] == 0


def test_live_es_trust_rank_and_conflict_flow(live_stack: Stack):
    live_stack.ingest(
        "draft.md", _policy(2026, 30), governance=DocumentGovernance(trust_tier="draft_notes", effective_date="2026-09-01")
    )
    live_stack.ingest(
        "official.md", _policy(2026, 180), governance=DocumentGovernance(trust_tier="official_policy", effective_date="2026-01-01")
    )
    chunks = live_stack.search("rotate network password", top_k=2)
    assert chunks[0]["filename"] == "official.md"
    assert chunks[0]["effective_date"] == "2026-01-01"
    assert detect_conflicts(chunks)[0]["winner"]["filename"] == "official.md"
