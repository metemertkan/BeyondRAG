#!/usr/bin/env python3
"""Measure the real behaviour behind each Episode 2 visual and write it as JSON.

Every number shown in the episode comes from here, so a visual can be regenerated
rather than trusted. "Before" columns use the naive technique each challenge warns
about. Needs Elasticsearch for the deletion timeline and Ollama bge-m3 for the
embedding sections; a missing service marks that section as skipped.

    python scripts/ep2_showcase.py --out showcase.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

import pdfplumber  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

from src.eval.harness import NoopCrossLingual  # noqa: E402
from src.rag.chunker import AdaptiveChunker, ChunkingConfig, SemanticChunker, SlidingWindowChunker, create_chunker  # noqa: E402
from src.rag.embeddings import CrossLingualEmbedder, FeatureHashEmbeddingProvider, OllamaEmbeddingProvider, normalize_l2  # noqa: E402
from src.rag.erasure import ErasureService  # noqa: E402
from src.rag.es_client import ElasticsearchIndexManager  # noqa: E402
from src.rag.extractor import DocumentParserRegistry, PdfPlumberDocumentParser, XlsxDocumentParser  # noqa: E402
from src.rag.governance import DocumentGovernance, detect_conflicts  # noqa: E402
from src.rag.ingestor import PdfIngestor  # noqa: E402
from src.rag.multilingual import MultilingualTokenizer  # noqa: E402
from src.rag.repository import DocumentRepository  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.rag.synthesizer import GroundedAnswerSynthesizer  # noqa: E402
from src.rag.tabular_store import TableQuery  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PERSON = "maya.chen@example.com"

THREE_TOPIC_DOCUMENT = [
    "The treasury team holds an intraday cash buffer in Hong Kong dollars.",
    "This buffer protects settlement obligations when large payments arrive late in the day.",
    "If the balance falls under the threshold, the treasurer must be told within one hour.",
    "Liquidity forecasts are refreshed each morning using the previous day's bank statements.",
    "Any shortfall is covered by drawing on the committed credit facility.",
    "Foreign exchange exposure from these payments is reviewed at the end of each week.",
    "Staff must complete phishing awareness training once a year.",
    "Passwords for network accounts are rotated on a fixed schedule set by the security office.",
    "Multi-factor authentication is required for every remote login.",
    "Lost laptops have to be reported to the service desk on the same day.",
    "The security office reviews administrator access rights every quarter.",
    "Software from unknown publishers cannot be installed on company machines.",
    "Employees book flights through the approved travel portal.",
    "Economy class is the default for any trip shorter than eight hours.",
    "Hotel costs are reimbursed up to a nightly limit that depends on the city.",
    "Receipts must be submitted within fourteen days of returning from a trip.",
    "Meals during business travel are covered by a daily allowance.",
    "Personal days added to a work trip are paid for by the employee.",
]
TOPIC_ENDS = [5, 11]

SIMILARITY_TEXTS = {
    "en": "What is the minimum intraday cash buffer?",
    "zh": "最低日内现金缓冲是多少？",
    "ar": "ما هو الحد الأدنى للسيولة النقدية خلال اليوم؟",
    "en_passage": "The treasury team must hold an intraday cash buffer of HKD 520 million.",
    "zh_passage": "财务团队必须持有五亿二千万港元的日内现金缓冲。",
    "ar_passage": "يجب على قسم الخزينة الاحتفاظ بسيولة نقدية خلال اليوم قدرها 520 مليون دولار هونغ كونغ.",
    "unrelated_passage": "Employees book flights through the approved travel portal.",
}


def flat_xlsx_text(raw: bytes) -> Dict[str, List[str]]:
    """The naive path: dump cell values row by row, ignoring formulas and merges."""
    book = load_workbook(io.BytesIO(raw), data_only=False)
    output: Dict[str, List[str]] = {}
    for sheet in book.worksheets:
        output[sheet.title] = [
            " ".join("" if cell is None else str(cell) for cell in row).strip()
            for row in sheet.iter_rows(values_only=True, max_row=6)
        ]
    return output


def section_excel() -> Dict[str, Any]:
    raw = (FIXTURES / "xlsx" / "regional_budget.xlsx").read_bytes()
    parsed = XlsxDocumentParser().extract_document(raw)
    budget = next(block.text for block in parsed.blocks if block.metadata.get("sheet") == "Budget")
    flat = flat_xlsx_text(raw)
    return {
        "flat_budget_rows": flat["Budget"],
        "flat_exposes_hidden_sheet": "Salaries_Confidential" in flat,
        "structured_budget": budget,
        "large_sheet": {
            "rows": len(parsed.tables[0].rows),
            "chunks_if_embedded_per_row": len(parsed.tables[0].rows),
            "chunks_with_routing": 1,
            "summary": next(block.text for block in parsed.blocks if block.block_type == "table_summary"),
        },
    }


def section_multilingual() -> Dict[str, Any]:
    raw = (FIXTURES / "pdfs" / "arabic_mixed_policy.pdf").read_bytes()
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        extracted = pdf.pages[0].extract_text()
    repaired = PdfPlumberDocumentParser().extract_document(raw).blocks[0].text

    chinese = "本政策规定了公司的现金管理要求。" * 60
    tokenizer = MultilingualTokenizer()
    naive_words = chinese.split()
    chunks = SlidingWindowChunker(ChunkingConfig(max_tokens=100, overlap_tokens=15)).split(chinese)
    return {
        "rtl": {"naive_extraction": extracted.split("\n"), "repaired": repaired.split("\n")},
        "chinese_chunking": {
            "limit_tokens": 100,
            "whitespace_split_units": len(naive_words),
            "tokens_in_whitespace_chunk": tokenizer.count_tokens(chinese),
            "fixed_chunks": len(chunks),
            "fixed_chunk_token_counts": [tokenizer.count_tokens(chunk) for chunk in chunks],
        },
        "tokens": {
            "input": "最低现金储备 الْحَدُّ الأدنى؟ HKD 450,000,000",
            "output": tokenizer.split_into_tokens("最低现金储备 الْحَدُّ الأدنى؟ HKD 450,000,000"),
        },
    }


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(normalize_l2(a), normalize_l2(b)))


def section_embeddings(ollama_url: str) -> Dict[str, Any]:
    provider = OllamaEmbeddingProvider(model_name="bge-m3", dimensions=1024, base_url=ollama_url, timeout=600.0)
    embedder = CrossLingualEmbedder(provider)
    try:
        embedder.embed("probe")
    except Exception as exc:
        return {"skipped": f"bge-m3 unavailable: {exc}"}

    vectors = {name: embedder.embed(text) for name, text in SIMILARITY_TEXTS.items()}
    queries = ["en", "zh", "ar"]
    passages = ["en_passage", "zh_passage", "ar_passage", "unrelated_passage"]
    matrix = [[round(_cosine(vectors[q], vectors[p]), 3) for p in passages] for q in queries]

    text = " ".join(THREE_TOPIC_DOCUMENT)
    config = ChunkingConfig(strategy="semantic", max_tokens=120, overlap_tokens=18)
    semantic = SemanticChunker(config, embedder, overlap_ratio=0.15)
    started = time.perf_counter()
    semantic_chunks = semantic.split(text)
    semantic_ms = (time.perf_counter() - started) * 1000.0

    fixed = SlidingWindowChunker(ChunkingConfig(max_tokens=65, overlap_tokens=10))
    started = time.perf_counter()
    fixed_chunks = fixed.split(text)
    fixed_ms = (time.perf_counter() - started) * 1000.0

    uniform = "\n".join(f"item-{n}: value {n} recorded by the nightly job and stored safely." for n in range(60))
    router = AdaptiveChunker(ChunkingConfig(strategy="adaptive", max_tokens=120, overlap_tokens=18), embedder)
    routes = [
        {"document": "Long free-form policy prose", **router.route(text * 2)},
        {"document": "Line-oriented export or log", **router.route(uniform)},
        {"document": "Short notice", **router.route("Offices close at 18:00 on Friday.")},
    ]

    def coverage(chunk: str) -> Dict[str, Any]:
        """Which sentences a chunk holds whole, and which it cuts through."""
        whole, cut = [], []
        for index, sentence in enumerate(THREE_TOPIC_DOCUMENT):
            if sentence in chunk:
                whole.append(index)
            else:
                words = sentence.split()
                head, tail = " ".join(words[:3]), " ".join(words[-3:])
                if chunk.startswith(tail) or head in chunk or tail in chunk:
                    cut.append(index)
        return {"whole_sentences": whole, "cut_sentences": cut}

    return {
        "model": provider.manifest.config_identifier,
        "similarity": {"queries": queries, "passages": passages, "texts": SIMILARITY_TEXTS, "matrix": matrix},
        "semantic": {
            "sentences": THREE_TOPIC_DOCUMENT,
            "true_topic_ends": TOPIC_ENDS,
            "similarities": semantic.last_stats.similarities,
            "depths": semantic.last_stats.depths,
            "threshold": semantic.last_stats.threshold,
            "boundaries": semantic.last_stats.boundaries,
            "embed_calls": semantic.last_stats.embed_calls,
            "embed_ms": semantic.last_stats.embed_ms,
            "semantic_total_ms": round(semantic_ms, 2),
            "fixed_total_ms": round(fixed_ms, 3),
            "semantic_chunks": [{"text": c, **coverage(c)} for c in semantic_chunks],
            "fixed_chunks": [{"text": c, **coverage(c)} for c in fixed_chunks],
        },
        "routes": routes,
    }


class _Stack:
    def __init__(self, workdir: Path, es_url: str, chunker: Any = None) -> None:
        self.embedder = CrossLingualEmbedder(FeatureHashEmbeddingProvider(dimensions=256))
        self.es = ElasticsearchIndexManager(
            es_url=es_url, index_name=f"beyondrag-showcase-{uuid.uuid4().hex[:8]}", dimensions=256
        )
        self.repo = DocumentRepository(
            es_manager=self.es, uploads_dir=workdir / "uploads", registry_path=workdir / "registry.json"
        )
        self.ingestor = PdfIngestor(
            extractor=DocumentParserRegistry(),
            chunker=chunker or create_chunker("sliding_window", max_tokens=200, overlap_tokens=30),
            cross_lingual=NoopCrossLingual(),
            embedder=self.embedder,
            repository=self.repo,
        )
        self.retriever = HybridRetriever(self.es)

    def search(self, query: str, roles: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
        return self.retriever.search(query, self.embedder.embed(query), roles, top_k=top_k, tenant="default").authorized_chunks

    def close(self) -> None:
        if self.es.is_online():
            self.es.client.indices.delete(index=self.es.index_name, ignore_unavailable=True)


def _hit_view(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "filename": chunk["filename"],
        "trust_tier": chunk.get("trust_tier"),
        "effective_date": chunk.get("effective_date"),
        "score": chunk["score"],
    }


def section_versioning(es_url: str) -> Dict[str, Any]:
    v1 = (FIXTURES / "pdfs" / "policy_v1_liquidity_table.pdf").read_bytes()
    v2 = (FIXTURES / "pdfs" / "policy_v2_liquidity_table.pdf").read_bytes()
    query = "intraday HKD cash buffer threshold"
    views: Dict[str, Any] = {}
    for label, keys in (("appended_no_lifecycle", ("old", "new")), ("versioned_same_source_key", ("t", "t"))):
        with tempfile.TemporaryDirectory() as tmp:
            stack = _Stack(Path(tmp), es_url)
            try:
                stack.ingestor.ingest("policy_v1.pdf", v1, tags=["finance"], source_key=keys[0])
                stack.ingestor.ingest("policy_v2.pdf", v2, tags=["finance"], source_key=keys[1])
                blob = " ".join(c["content"] for c in stack.search(query, ["finance"]))
                views[label] = {"contains_v1_450m": "450,000,000" in blob, "contains_v2_520m": "520,000,000" in blob}
            finally:
                stack.close()
    return views


def section_conflicts(es_url: str) -> Dict[str, Any]:
    corpus = [
        ("it_policy_2023.md", "official_policy", "2023-03-01", "2025-12-31", 90),
        ("it_policy_2026.md", "official_policy", "2026-01-01", None, 180),
        ("security_draft_notes.md", "draft_notes", "2026-09-10", None, 30),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        stack = _Stack(Path(tmp), es_url)
        try:
            for name, tier, effective, expires, days in corpus:
                text = f"Section 4: Password rotation. Employees must rotate their network password every {days} days."
                stack.ingestor.ingest(
                    name,
                    text.encode("utf-8"),
                    source_key=name,
                    governance=DocumentGovernance(trust_tier=tier, effective_date=effective, expires_at=expires),
                )
            question = "How often must employees rotate their network password?"
            hits = stack.search(question, ["__public__"])
            answer = GroundedAnswerSynthesizer().synthesize(
                question=question,
                expanded_query=question,
                matched_concepts=[],
                is_cjk_query=False,
                user_roles=[],
                authorized_chunks=hits,
                masked_restricted_docs=[],
            )
            return {
                "corpus": [
                    {"filename": name, "trust_tier": tier, "effective_date": effective, "expires_at": expires, "rotation_days": days}
                    for name, tier, effective, expires, days in corpus
                ],
                "hits": [_hit_view(hit) for hit in hits],
                "conflicts": detect_conflicts(hits),
                "fast_answer": answer,
            }
        finally:
            stack.close()


def section_erasure(es_url: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        stack = _Stack(workdir, es_url)
        try:
            governance = DocumentGovernance(subject_ids=(PERSON,))
            stack.ingestor.ingest("hr_review.md", f"Review of {PERSON}: rating 3.".encode(), source_key="hr", governance=governance)
            stack.ingestor.ingest("hr_review.md", f"Review of {PERSON}: rating 4.".encode(), source_key="hr", governance=governance)
            stack.ingestor.ingest(
                "regional_budget.xlsx",
                (FIXTURES / "xlsx" / "regional_budget.xlsx").read_bytes(),
                tags=["finance"],
                source_key="budget",
            )
            stack.ingestor.ingest("plants.md", b"Office plants are watered every Tuesday.", source_key="plants")

            service = ErasureService(stack.repo, audit_log_path=workdir / "audit.jsonl")
            plan = service.plan_subject(PERSON)

            def es_docs() -> Dict[str, int]:
                client, index = stack.es.client, stack.es.index_name
                client.indices.refresh(index=index)
                stats = client.indices.stats(index=index, metric="docs")["indices"][index]["primaries"]["docs"]
                mentions = client.count(index=index, query={"match_phrase": {"content": PERSON}})["count"]
                return {"searchable_mentions": mentions, "live_docs": stats["count"], "soft_deleted_on_disk": stats["deleted"]}

            timeline = [{"stage": "before erasure", "seconds": 0.0, **es_docs()}]
            started = time.monotonic()
            receipt = service.execute(plan)
            timeline.append({"stage": "after delete + force merge", "seconds": round(time.monotonic() - started, 1), **es_docs()})
            remaining = service.confirm_physical_purge(receipt.request_id, max_wait_seconds=120, poll_seconds=3)
            timeline.append({"stage": "after retention lease sync + merge", "seconds": round(time.monotonic() - started, 1), **es_docs()})

            blocked = ""
            try:
                stack.ingestor.ingest("hr_review.md", f"Review of {PERSON}: rating 4.".encode(), source_key="hr", governance=governance)
            except ValueError as exc:
                blocked = str(exc)

            audit = (workdir / "audit.jsonl").read_text(encoding="utf-8")
            return {
                "plan": {
                    "versions": plan.versions,
                    "match_type": plan.match_type,
                    "chunk_count": plan.chunk_count,
                    "table_ids": plan.table_ids,
                },
                "receipt": receipt.to_dict(),
                "physical_purge_remaining": remaining,
                "timeline": timeline,
                "resurrection_blocked": blocked,
                "audit_log_contains_identifier": PERSON in audit,
            }
        finally:
            stack.close()


def section_tables(es_url: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        stack = _Stack(Path(tmp), es_url)
        try:
            report = stack.ingestor.ingest(
                "regional_budget.xlsx", (FIXTURES / "xlsx" / "regional_budget.xlsx").read_bytes(), tags=["finance"]
            )
            table_id = report["table_ids"][0]
            query = TableQuery(table_id=table_id, aggregate="sum", aggregate_column="Amount USD", group_by="Region")
            result = stack.repo.tabular_store.query(query, ["finance"])
            denied = ""
            try:
                stack.repo.tabular_store.query(query, ["__public__"])
            except PermissionError as exc:
                denied = str(exc)
            return {"tool_call": query.model_dump(exclude_defaults=True), "result": result, "denied_for_public": denied}
        finally:
            stack.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--es-url", default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    args = parser.parse_args()

    data = {
        "excel": section_excel(),
        "tables": section_tables(args.es_url),
        "multilingual": section_multilingual(),
        "embeddings": section_embeddings(args.ollama_url),
        "versioning": section_versioning(args.es_url),
        "conflicts": section_conflicts(args.es_url),
        "erasure": section_erasure(args.es_url),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
