#!/usr/bin/env python3
"""Measure the Episode 4 trust features. Writes one JSON file per section.

  python scripts/ep4_benchmark.py --out out --sections detection,scifact,end_to_end

  detection   Planted errors in single sentences: which checks catch which kind of error.
  scifact     Human-labeled claims (SciFact): how often the LLM verifier agrees with people.
  end_to_end  Questions over the lab corpus, half unanswerable: citations, hallucinations, abstention.

Needs Ollama (bge-m3 and the --llm model) and, for end_to_end, Elasticsearch and `.[rerank]`.
SciFact is downloaded to ~/.cache/beyondrag-bench and is not part of this repository.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import statistics
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep3_benchmark as base  # noqa: E402  (sets the local-only LLM environment)
import ep4_fixtures as fixtures  # noqa: E402
from src.config import get_llm  # noqa: E402
from src.rag.pipeline import PipelineOptions, RetrievalPipeline  # noqa: E402
from src.rag.reranker import FastEmbedCrossEncoder  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.rag.synthesizer import GroundedAnswerSynthesizer  # noqa: E402
from src.rag.trust import (  # noqa: E402
    ABSTAIN_MESSAGE,
    AnswerVerifier,
    CitedSentence,
    LlmClaimVerifier,
    apply_trust_policy,
    checkable_values,
    split_cited_sentences,
    value_in,
)

ALL_TEXT = [""]
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
ROOT = Path(__file__).resolve().parents[1]


def run_detection(_: argparse.Namespace) -> Dict[str, Any]:
    cases = fixtures.claim_cases()
    numeric_only = AnswerVerifier(numeric=True)
    llm_only = AnswerVerifier(LlmClaimVerifier(get_llm), numeric=False)
    rows: List[Dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        print("detection", number, case["kind"], flush=True)
        sources = {sid: {"filename": f"{sid}.md", "page_number": 1, "text": text} for sid, text in case["sources"].items()}
        answer = f"{case['sentence']} {''.join(f'[{sid}]' for sid in case['cites'])}"
        started = time.perf_counter()
        llm_report = llm_only.report(answer, sources)
        llm_ms = (time.perf_counter() - started) * 1000
        row = {"kind": case["kind"], "should_flag": case["should_flag"], "llm_ms": llm_ms}
        for name, report in (("values", numeric_only.report(answer, sources)), ("llm", llm_report)):
            status = report.sentences[0].status if report.sentences else "empty"
            # "unchecked" means no check had anything to say, which is not a pass or a flag.
            row[name] = {"flagged": status not in {"supported", "unchecked"}, "status": status, "llm_failed": "llm_failed" in report.checks}
        # Both checks together: either one flagging the sentence is enough.
        row["values + llm"] = {"flagged": row["values"]["flagged"] or row["llm"]["flagged"], "status": "", "llm_failed": row["llm"]["llm_failed"]}
        rows.append(row)
    detectors = ("values", "llm", "values + llm")
    by_kind: Dict[str, Dict[str, float]] = {}
    for kind in fixtures.PERTURBATIONS:
        subset = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {d: round(sum(r[d]["flagged"] for r in subset) / len(subset), 3) for d in detectors}
    errors = [r for r in rows if r["should_flag"]]
    clean = [r for r in rows if not r["should_flag"]]
    summary = {
        d: {
            "caught": round(sum(r[d]["flagged"] for r in errors) / len(errors), 3),
            "false_alarms": round(sum(r[d]["flagged"] for r in clean) / len(clean), 3),
        }
        for d in detectors
    }
    return {
        "llm": base.BENCH_LLM,
        "cases": len(rows),
        "flag_rate_by_kind": by_kind,
        "summary": summary,
        "llm_latency_p50_ms": round(statistics.median(r["llm_ms"] for r in rows)),
        "llm_failures": sum(r["llm"]["llm_failed"] for r in rows),
        "rows": rows,
    }


def _load_scifact_claims(per_label: int) -> List[Dict[str, Any]]:
    root = base.CACHE / "scifact_orig" / "data"
    if not root.exists():
        data = httpx.get(SCIFACT_URL, timeout=300, follow_redirects=True).content
        tarfile.open(fileobj=io.BytesIO(data)).extractall(base.CACHE / "scifact_orig")
    corpus = {}
    for line in (root / "corpus.jsonl").open(encoding="utf-8"):
        doc = json.loads(line)
        corpus[doc["doc_id"]] = doc["title"] + ". " + " ".join(doc["abstract"])
    grouped: Dict[str, List[Dict[str, Any]]] = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    for line in (root / "claims_dev.jsonl").open(encoding="utf-8"):
        claim = json.loads(line)
        if claim["evidence"]:
            doc_id, evidence = next(iter(claim["evidence"].items()))
            grouped[evidence[0]["label"]].append({"claim": claim["claim"], "label": evidence[0]["label"], "doc": corpus[int(doc_id)]})
        elif claim["cited_doc_ids"]:
            grouped["NEI"].append({"claim": claim["claim"], "label": "NEI", "doc": corpus[claim["cited_doc_ids"][0]]})
    rng = random.Random(4)
    picked = [item for label in grouped for item in rng.sample(grouped[label], min(per_label, len(grouped[label])))]
    rng.shuffle(picked)
    return picked


def _kappa(pairs: List[tuple[str, str]], labels: List[str]) -> float:
    total = len(pairs)
    observed = sum(a == b for a, b in pairs) / total
    expected = sum((sum(a == l for a, _ in pairs) / total) * (sum(b == l for _, b in pairs) / total) for l in labels)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def run_scifact(args: argparse.Namespace) -> Dict[str, Any]:
    claims = _load_scifact_claims(args.per_label)
    verifier = LlmClaimVerifier(get_llm)
    mapping = {"supported": "SUPPORT", "contradicted": "CONTRADICT", "not_stated": "NEI"}
    pairs: List[tuple[str, str]] = []
    numeric_pairs: List[tuple[str, bool]] = []
    failures = 0
    latencies = []
    for number, item in enumerate(claims, start=1):
        print("scifact", number, item["label"], flush=True)
        started = time.perf_counter()
        try:
            verdicts = verifier.verify([CitedSentence(item["claim"], ["S1"])], {"S1": item["doc"]})
            predicted = mapping[verdicts[1].label] if 1 in verdicts else "NEI"
        except Exception:
            failures += 1
            predicted = "FAILED"
        latencies.append((time.perf_counter() - started) * 1000)
        pairs.append((item["label"], predicted))
        values = checkable_values(item["claim"])
        if values:
            numeric_pairs.append((item["label"], all(value_in(v, item["doc"]) for v in values)))
    labels = ["SUPPORT", "CONTRADICT", "NEI"]
    confusion = {gold: {pred: sum(1 for g, p in pairs if g == gold and p == pred) for pred in labels + ["FAILED"]} for gold in labels}
    return {
        "dataset": "SciFact dev claims, human labels",
        "llm": base.BENCH_LLM,
        "claims": len(pairs),
        "accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 3),
        "cohens_kappa": _kappa(pairs, labels),
        "confusion_gold_by_predicted": confusion,
        "failures": failures,
        "latency_p50_ms": round(statistics.median(latencies)),
        "value_check": {
            "claims_with_values": len(numeric_pairs),
            "all_values_found_when_supported": round(
                sum(found for g, found in numeric_pairs if g == "SUPPORT") / max(1, sum(g == "SUPPORT" for g, _ in numeric_pairs)), 3
            ),
            "all_values_found_when_not_supported": round(
                sum(found for g, found in numeric_pairs if g != "SUPPORT") / max(1, sum(g != "SUPPORT" for g, _ in numeric_pairs)), 3
            ),
        },
    }


def _invented(answer: str, question: str, all_text: str) -> List[str]:
    claims = re.sub(r"\[[^\]]*\]", " ", answer)  # citations name files, not facts
    return [v for v in checkable_values(claims) if not value_in(v, all_text) and not value_in(v, question)]


def _contains_any(text: str, values: List[str]) -> bool:
    return any(value.lower() in text.lower().replace(" ", " ") for value in values)


def run_end_to_end(args: argparse.Namespace) -> Dict[str, Any]:
    documents = json.loads((ROOT / "config" / "ep3_demo_docs.json").read_text(encoding="utf-8"))
    docs = [{"id": d["filename"], "filename": d["filename"], "text": d["text"]} for d in documents]
    embedder = base.CachedEmbedder()
    vectors = embedder.embed_many([d["text"] for d in docs])
    all_text = " ".join(d["text"] for d in docs)
    ALL_TEXT[0] = all_text
    verifier = AnswerVerifier(LlmClaimVerifier(get_llm))
    rows: List[Dict[str, Any]] = []
    with base.TempIndex() as index:
        index.load(docs, vectors)
        pipeline = RetrievalPipeline(HybridRetriever(index.manager), embedder, reranker=FastEmbedCrossEncoder())
        for prompt_name, cite in (("old prompt", False), ("cited prompt", True)):
            synthesizer = GroundedAnswerSynthesizer(get_llm, cite_sources=cite)
            for number, item in enumerate(fixtures.QUESTIONS, start=1):
                print("end_to_end", prompt_name, number, flush=True)
                retrieved = pipeline.run(item["q"], item["q"], base.ACL, top_k=4, options=PipelineOptions(rerank=True, candidates=20))
                best = max((c["score"] for c in retrieved.chunks), default=0.0)
                started = time.perf_counter()
                answer = synthesizer.synthesize(item["q"], item["q"], [], False, ["finance"], retrieved.chunks, [], synthesis_mode="llm", ranked_by_model=True)
                synth_ms = (time.perf_counter() - started) * 1000
                row: Dict[str, Any] = {
                    "prompt": prompt_name, "q": item["q"], "answerable": item["expect"] is not None,
                    "llm_used": synthesizer.last_mode_used == "llm", "best_relevance": best, "synth_ms": round(synth_ms),
                    "raw": synthesizer.last_raw_answer, "answer": answer,
                }
                # A value that no document contains, and the user didn't type, is invented.
                row["invented_values"] = _invented(answer, item["q"], all_text)
                row["self_abstained"] = answer == ABSTAIN_MESSAGE
                if cite and row["llm_used"]:
                    started = time.perf_counter()
                    report = verifier.report(synthesizer.last_raw_answer, synthesizer.last_sources)
                    row["verify_ms"] = round((time.perf_counter() - started) * 1000)
                    row["report"] = report.to_dict()
                    for mode in ("strip", "abstain"):
                        fresh = verifier.report(synthesizer.last_raw_answer, synthesizer.last_sources, use_llm=False)
                        fresh.sentences = [type(s)(**vars(s)) for s in report.sentences]
                        row[f"{mode}_answer"] = apply_trust_policy(fresh, answer, synthesizer.last_sources, mode)
                    row["citation_ok"] = _citation_ok(item, synthesizer, report)
                rows.append(row)
    return {"llm": base.BENCH_LLM, "questions": len(fixtures.QUESTIONS), "summary": _summarize(rows), "rows": rows}


def _citation_ok(item: Dict[str, Any], synthesizer: GroundedAnswerSynthesizer, report: Any) -> Any:
    """For a correct answer: does the sentence stating the value cite a source that contains it?"""
    if not item["expect"]:
        return None
    for sentence in report.sentences:
        if _contains_any(sentence.text, item["expect"]):
            cited = " ".join(str(synthesizer.last_sources[s]["text"]) for s in sentence.citations if s in synthesizer.last_sources)
            return _contains_any(cited, item["expect"])
    return None


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    def outcome(rows_: List[Dict[str, Any]], answer_key: str) -> Dict[str, Any]:
        answerable = [r for r in rows_ if r["answerable"]]
        unanswerable = [r for r in rows_ if not r["answerable"]]
        expect = {f["q"]: f["expect"] for f in fixtures.QUESTIONS}
        correct = sum(_contains_any(r[answer_key], expect[r["q"]]) for r in answerable)
        wrongly_refused = sum(r[answer_key] == ABSTAIN_MESSAGE or "couldn't find" in r[answer_key] for r in answerable)
        refused = sum(r[answer_key] == ABSTAIN_MESSAGE or "couldn't find" in r[answer_key] for r in unanswerable)
        invented = sum(bool(_invented(r[answer_key], r["q"], ALL_TEXT[0])) for r in rows_)
        return {
            "answerable_correct": f"{correct}/{len(answerable)}",
            "answerable_refused": f"{wrongly_refused}/{len(answerable)}",
            "unanswerable_refused": f"{refused}/{len(unanswerable)}",
            "answers_with_invented_values": f"{invented}/{len(rows_)}",
        }

    old = [r for r in rows if r["prompt"] == "old prompt"]
    new = [r for r in rows if r["prompt"] == "cited prompt"]
    summary["old prompt"] = outcome(old, "answer")
    summary["cited prompt"] = outcome(new, "answer")
    checked = [r for r in new if "report" in r]
    summary["cited prompt + strip"] = outcome(checked, "strip_answer")
    summary["cited prompt + abstain"] = outcome(checked, "abstain_answer")
    oks = [r["citation_ok"] for r in checked if r["citation_ok"] is not None]
    summary["citation_precision"] = f"{sum(oks)}/{len(oks)}"
    summary["verify_p50_ms"] = round(statistics.median(r["verify_ms"] for r in checked)) if checked else None
    summary["synth_p50_ms"] = round(statistics.median(r["synth_ms"] for r in new)) if new else None
    # How well the re-ranker's best score separates answerable from unanswerable questions.
    scored = [(r["best_relevance"], r["answerable"]) for r in new]
    positives = [s for s, a in scored if a]
    negatives = [s for s, a in scored if not a]
    pairs = [(p, n) for p in positives for n in negatives]
    summary["relevance_auc"] = round(sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs), 3) if pairs else None
    summary["relevance_scores"] = {"answerable": sorted(round(s, 3) for s in positives), "unanswerable": sorted(round(s, 3) for s in negatives)}
    return summary


SECTIONS: Dict[str, Callable[[argparse.Namespace], Dict[str, Any]]] = {
    "detection": run_detection,
    "scifact": run_scifact,
    "end_to_end": run_end_to_end,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--llm", default="qwen3:8b")
    parser.add_argument("--per-label", type=int, default=40, help="SciFact claims per label.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for section in args.sections.split(","):
        started = time.perf_counter()
        result = SECTIONS[section](args)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        (args.out / f"{section}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("DONE", section, result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
