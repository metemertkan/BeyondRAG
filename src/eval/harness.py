"""Runs the golden dataset through the real pipeline and applies the quality gate.

Two kinds of check, on purpose:

  - Deterministic, needing no LLM: did retired facts reach the context, and does the
    answer contain the expected value?
  - Judged: Context Precision (did we fetch the right chunks, rank-weighted) and
    Faithfulness (did the answer stick to the chunks).

A run where the judge could not answer, or where synthesis silently fell back from the
LLM, is INCONCLUSIVE and never counts as a pass. A gate that goes green because
nothing was evaluated is worse than having no gate.
"""

from __future__ import annotations

import json
import random
import tempfile
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.eval.judge import Judge, JudgeUnavailable
from src.rag.chunker import create_chunker
from src.rag.embeddings import CrossLingualEmbedder, create_embedding_provider
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import DocumentParserRegistry
from src.rag.governance import DocumentGovernance
from src.rag.ingestor import PdfIngestor
from src.rag.multilingual import normalize_for_matching
from src.rag.repository import DocumentRepository
from src.rag.retriever import HybridRetriever
from src.rag.service import RBACKnowledgeBaseService
from src.rag.synthesizer import GroundedAnswerSynthesizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "eval" / "golden_dataset.json"
DRIFT_MODES = ("none", "stale_index", "noise")
PASSED, FAILED, INCONCLUSIVE = "PASSED", "FAILED", "INCONCLUSIVE"


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str = ""
    retrieved: List[str] = field(default_factory=list)
    relevance: List[bool] = field(default_factory=list)
    context_precision: Optional[float] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)
    faithfulness: Optional[float] = None
    answer_correct: bool = False
    retired_fact_leaks: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvalReport:
    status: str
    scores: Dict[str, float]
    thresholds: Dict[str, float]
    failures: List[str]
    cases: List[CaseResult]
    retrieval_backend: str
    embedding_manifest: str
    drift: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NoopCrossLingual:
    """Skips LLM concept extraction and query expansion: the gate measures ingestion and retrieval."""

    def is_cross_lingual(self, text: str) -> bool:
        return False

    def extract_concepts_at_ingestion(self, filename: str, full_text: str) -> Tuple[List[Dict[str, Any]], str]:
        return [], "none"

    def prepare_query(self, question: str, allow_llm_fallback: bool = False) -> Tuple[str, List[str]]:
        return question, []


def context_precision(relevance: Sequence[bool]) -> float:
    """Rank-weighted precision (the Ragas formula): a relevant chunk at rank 1 is worth more."""
    total_relevant = sum(relevance)
    if not total_relevant:
        return 0.0
    hits = 0
    weighted = 0.0
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            weighted += hits / rank
    return round(weighted / total_relevant, 4)


def _canonical(text: str) -> str:
    """Compare values regardless of thousands separators, width, case or diacritics."""
    return unicodedata.normalize("NFKC", normalize_for_matching(text)).replace(",", "").lower()


def contains_value(haystack: str, needle: str) -> bool:
    return _canonical(needle) in _canonical(haystack)


class EvalPipeline:
    """The real service wired against an isolated index and temporary stores."""

    def __init__(self, *, es_url: str, keep_index: bool = False) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="beyondrag-eval-")
        workdir = Path(self._tmp.name)
        self._keep_index = keep_index
        provider = create_embedding_provider()
        self.embedder = CrossLingualEmbedder(provider)
        # A throwaway index name, so a local run can never touch the app's own index.
        self.es = ElasticsearchIndexManager(
            es_url=es_url,
            index_name=f"beyondrag-eval-{uuid.uuid4().hex[:10]}",
            dimensions=provider.dimensions,
        )
        self.repository = DocumentRepository(
            es_manager=self.es,
            uploads_dir=workdir / "uploads",
            registry_path=workdir / "registry.json",
        )
        self.cross_lingual = NoopCrossLingual()
        self.synthesizer = GroundedAnswerSynthesizer()
        self.ingestor = PdfIngestor(
            extractor=DocumentParserRegistry(),
            chunker=create_chunker("adaptive", max_tokens=220, overlap_tokens=35, embedder=self.embedder),
            cross_lingual=self.cross_lingual,
            embedder=self.embedder,
            repository=self.repository,
        )
        self.service = RBACKnowledgeBaseService(
            index_manager=self.es,
            repository=self.repository,
            ingestor=self.ingestor,
            cross_lingual=self.cross_lingual,
            embedder=self.embedder,
            retriever=HybridRetriever(self.es),
            synthesizer=self.synthesizer,
            seeder=object(),  # type: ignore[arg-type]
        )

    @property
    def backend(self) -> str:
        return "elasticsearch" if self.es.is_online() else "memory"

    def ingest_corpus(self, documents: Sequence[Dict[str, Any]], *, drift: str = "none") -> None:
        for number, document in enumerate(documents):
            raw = (REPO_ROOT / document["path"]).read_bytes()
            governance = DocumentGovernance(**document.get("governance", {}))
            source_key = document.get("source_key")
            if drift == "stale_index":
                # The classic silent regression: a sync that appends instead of replacing
                # and drops lifecycle metadata, so old and new versions compete.
                governance = DocumentGovernance()
                source_key = f"stale-{number}"
            self.service.ingest_pdf_bytes(
                filename=Path(document["path"]).name,
                pdf_bytes=raw,
                tags=document.get("tags"),
                source_key=source_key,
                governance=governance,
            )
        if drift == "noise":
            self._ingest_noise(documents)

    def _ingest_noise(self, documents: Sequence[Dict[str, Any]], count: int = 80) -> None:
        """Near-duplicates that reuse the corpus vocabulary but contain none of the answers."""
        rng = random.Random(7)
        vocabulary = sorted(
            {
                word
                for document in documents
                if document["path"].endswith(".md")
                for word in (REPO_ROOT / document["path"]).read_text(encoding="utf-8").split()
                if word.isalpha() and len(word) > 3
            }
            | {"intraday", "cash", "buffer", "threshold", "liquidity", "password", "rotation", "budget"}
        )
        for number in range(count):
            body = " ".join(rng.choices(vocabulary, k=120))
            self.service.ingest_pdf_bytes(
                filename=f"noise_{number:03d}.md",
                pdf_bytes=f"Meeting notes {number}. {body}.".encode("utf-8"),
                source_key=f"noise-{number}",
            )

    def close(self) -> None:
        if not self._keep_index and self.es.is_online() and self.es.client.indices.exists(index=self.es.index_name):
            self.es.client.indices.delete(index=self.es.index_name)
        self._tmp.cleanup()


def evaluate_case(case: Dict[str, Any], pipeline: EvalPipeline, judge: Judge, *, top_k: int) -> CaseResult:
    result = CaseResult(case_id=case["id"], question=case["question"])
    response = pipeline.service.query_and_chat(
        question=case["question"],
        user_roles=case.get("roles", []),
        top_k=top_k,
        synthesis_mode="llm",
    )
    contexts = [chunk["content"] for chunk in response["authorized_chunks"]]
    result.answer = response["answer"]
    result.retrieved = [chunk.get("filename", "") for chunk in response["authorized_chunks"]]

    if pipeline.synthesizer.last_mode_used != "llm":
        result.error = "Synthesis fell back to the citation answer: the LLM was not exercised."
        return result

    blob = "\n".join(contexts)
    result.retired_fact_leaks = [value for value in case.get("retired_facts", []) if contains_value(blob, value)]
    result.answer_correct = any(contains_value(result.answer, value) for value in case["must_contain_any"])

    result.relevance = [judge.is_relevant(case["question"], case["expected_answer"], context) for context in contexts]
    result.context_precision = context_precision(result.relevance)

    # Faithfulness is measured against what the model actually saw, including structured
    # query results and source headers; retrieved text alone would mark those as invented.
    claims = judge.judge_claims(result.answer, pipeline.synthesizer.last_context_parts)
    result.claims = [{"claim": claim.claim, "supported": claim.supported} for claim in claims]
    if claims:
        result.faithfulness = round(sum(1 for claim in claims if claim.supported) / len(claims), 4)
    else:
        result.faithfulness = 1.0 if not contexts else 0.0
    return result


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_report(
    cases: List[CaseResult],
    thresholds: Dict[str, float],
    *,
    backend: str,
    manifest: str,
    drift: str,
) -> EvalReport:
    errors = [f"{case.case_id}: {case.error}" for case in cases if case.error]
    scored = [case for case in cases if not case.error]
    scores = {
        "context_precision": _mean([c.context_precision or 0.0 for c in scored]),
        "faithfulness": _mean([c.faithfulness or 0.0 for c in scored]),
        "answer_correct": _mean([1.0 if c.answer_correct else 0.0 for c in scored]),
        "retired_fact_leaks": float(sum(len(c.retired_fact_leaks) for c in scored)),
    }
    if errors or not scored:
        return EvalReport(
            INCONCLUSIVE, scores, thresholds, errors or ["No case was evaluated."], cases, backend, manifest, drift
        )
    failures = [
        f"{name} {scores[name]:.3f} is below the floor {thresholds[name]:.3f}"
        for name in ("context_precision", "faithfulness", "answer_correct")
        if scores[name] < thresholds[name]
    ]
    if scores["retired_fact_leaks"] > thresholds["retired_fact_leaks"]:
        leaked = sorted({value for case in scored for value in case.retired_fact_leaks})
        failures.append(f"retired facts reached the context: {', '.join(leaked)}")
    return EvalReport(FAILED if failures else PASSED, scores, thresholds, failures, cases, backend, manifest, drift)


def run_evaluation(
    judge: Judge,
    *,
    es_url: str,
    dataset_path: Path = DEFAULT_DATASET,
    drift: str = "none",
) -> EvalReport:
    if drift not in DRIFT_MODES:
        raise ValueError(f"Unknown drift mode {drift!r}. Valid: {DRIFT_MODES}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    pipeline = EvalPipeline(es_url=es_url)
    try:
        pipeline.ingest_corpus(dataset["corpus"], drift=drift)
        results: List[CaseResult] = []
        for case in dataset["cases"]:
            try:
                results.append(evaluate_case(case, pipeline, judge, top_k=dataset.get("top_k", 3)))
            except JudgeUnavailable as exc:
                results.append(CaseResult(case_id=case["id"], question=case["question"], error=str(exc)))
        return build_report(
            results,
            dataset["thresholds"],
            backend=pipeline.backend,
            manifest=pipeline.embedder.manifest.config_identifier,
            drift=drift,
        )
    finally:
        pipeline.close()


def render_markdown(report: EvalReport) -> str:
    icon = {PASSED: "✅", FAILED: "❌", INCONCLUSIVE: "⚠️"}[report.status]
    lines = [
        f"## {icon} RAG evaluation: {report.status}",
        "",
        f"Backend `{report.retrieval_backend}`, embeddings `{report.embedding_manifest}`"
        + (f", simulated drift `{report.drift}`" if report.drift != "none" else ""),
        "",
        "| Metric | Score | Gate |",
        "| --- | --- | --- |",
    ]
    for name in ("context_precision", "faithfulness", "answer_correct"):
        lines.append(f"| {name} | {report.scores[name]:.3f} | at least {report.thresholds[name]:.2f} |")
    lines.append(
        f"| retired_fact_leaks | {report.scores['retired_fact_leaks']:.0f} | "
        f"at most {report.thresholds['retired_fact_leaks']:.0f} |"
    )
    lines += ["", "| Case | Precision | Faithfulness | Correct | Top source |", "| --- | --- | --- | --- | --- |"]
    for case in report.cases:
        if case.error:
            lines.append(f"| {case.case_id} | error | error | error | {case.error[:80]} |")
            continue
        lines.append(
            f"| {case.case_id} | {case.context_precision:.2f} | {case.faithfulness:.2f} | "
            f"{'yes' if case.answer_correct else 'no'} | {case.retrieved[0] if case.retrieved else 'none'} |"
        )
    if report.failures:
        lines += ["", "### Why it did not pass", ""] + [f"- {failure}" for failure in report.failures]
    return "\n".join(lines) + "\n"
