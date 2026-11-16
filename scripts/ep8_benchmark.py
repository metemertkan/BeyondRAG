#!/usr/bin/env python3
"""Episode 8: Einstein's 1905 claims through the Episode 4 checker, with libraries of different dates.

  python scripts/ep8_benchmark.py --out runs/ep8 [--dates 1905,1914] [--claims E04,K01] [--phrasings 1]

For each library date: a temporary Elasticsearch index of every work published by then, the app's
retrieval (hybrid search, cross-encoder rerank, top 4 of 20), then the Episode 4 trust report
(`AnswerVerifier(LlmClaimVerifier(get_llm))`, value check included) on the claim citing all four
sources. Local only, `qwen3:8b`, as in Episode 4. Results are appended as they come, so a stopped
run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep3_benchmark as base  # noqa: E402  (sets the local-only LLM environment)
import ep8_claims as claims  # noqa: E402
import ep8_library as library  # noqa: E402
from src.config import get_llm  # noqa: E402
from src.rag.pipeline import PipelineOptions, RetrievalPipeline  # noqa: E402
from src.rag.reranker import FastEmbedCrossEncoder  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.rag.trust import AnswerVerifier, LlmClaimVerifier  # noqa: E402

PASSAGE_WORDS = 150
TOP_K, CANDIDATES = 4, 20
TENANT = "default"
STATUS = {"supported": "supported", "contradicted": "contradicted", "unsupported": "not stated"}


def sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    return [p.strip() for p in re.split(r"(?<=[.?!])\s+(?=[A-Z\"(])", text) if p.strip()]


def passages(text: str) -> List[str]:
    """About 150 words each, cut at sentence ends, one sentence of overlap."""
    out: List[str] = []
    cur: List[str] = []
    n = 0
    for s in sentences(re.sub(r"(?m)^## .*$", " ", text)):
        w = len(s.split())
        if cur and n + w > PASSAGE_WORDS:
            out.append(" ".join(cur))
            cur, n = cur[-1:], len(cur[-1].split())
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return [p for p in out if len(p.split()) >= 25]


def library_docs(date: tuple) -> List[Dict[str, Any]]:
    records = [json.loads(l) for l in library.MANIFEST.read_text().splitlines() if l.strip()]
    docs = []
    for r in records:
        if not library.in_library(r, date):
            continue
        text = library.read_text(r)
        for n, p in enumerate(passages(text)):
            docs.append({"id": f"{r['id']}#{n}", "filename": r["title"], "title": r["title"], "text": p,
                         "work": r["id"], "year": r["year"], "author": r["author"]})
    return docs


def source_text(chunk: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """The header the app puts on every source, with the work's year as its effective date."""
    return (f"Source: {chunk['filename']} (Page 1, Tags: ['PUBLIC'], Trust tier: published, Effective: {meta['year']})\n"
            f"Content: {chunk['content']}")


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "verdicts.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["library"], row["claim"], row["phrasing"]))
    rows = claims.phrasings()
    if args.claims:
        rows = [r for r in rows if r["claim"] in args.claims.split(",")]
    if args.phrasings:
        rows = [r for r in rows if r["phrasing"] in args.phrasings.split(",")]
    dates = [d for d in library.LIBRARY_DATES if not args.dates or str(d[0]) in args.dates.split(",")]
    embedder = base.CachedEmbedder()
    reranker = FastEmbedCrossEncoder()
    verifier = AnswerVerifier(LlmClaimVerifier(get_llm))
    for date in dates:
        label = f"{date[0]}-{date[1]:02}-{date[2]:02}"
        todo = [r for r in rows if (label, r["claim"], r["phrasing"]) not in done]
        if not todo:
            continue
        docs = library_docs(date)
        meta = {d["id"]: d for d in docs}
        started = time.time()
        vectors = embedder.embed_many([d["text"] for d in docs])
        print(f"library {label}: {len({d['work'] for d in docs})} works, {len(docs)} passages, embedded in {time.time() - started:.0f}s", flush=True)
        with base.TempIndex() as index:
            index.load(docs, vectors)
            # Episode 3's loader predates tenants (Episode 5); every query filters on one.
            index.manager.client.update_by_query(
                index=index.manager.index_name, refresh=True, conflicts="proceed",
                script={"source": "ctx._source.tenant = params.t", "params": {"t": TENANT}},
            )
            pipeline = RetrievalPipeline(HybridRetriever(index.manager), embedder, reranker=reranker)
            for n, row in enumerate(todo, 1):
                t0 = time.perf_counter()
                retrieved = pipeline.run(row["text"], row["text"], base.ACL, top_k=TOP_K,
                                         options=PipelineOptions(rerank=True, candidates=CANDIDATES), tenant=TENANT)
                chunks = retrieved.chunks[:TOP_K]
                sources = {
                    f"S{i}": {"filename": c["filename"], "page_number": 1, "chunk_id": c.get("chunk_id"), "text": source_text(c, meta[c["chunk_id"]])}
                    for i, c in enumerate(chunks, start=1)
                }
                raw = row["text"] + " " + "".join(f"[{sid}]" for sid in sources)
                report = verifier.report(raw, sources)
                sentence = report.sentences[0] if report.sentences else None
                status = sentence.status if sentence else "empty"
                record = {
                    "library": label, **row,
                    "status": status, "verdict": STATUS.get(status, status),
                    "reasons": sentence.reasons if sentence else [], "checks": report.checks,
                    "sources": [{"sid": sid, "work": meta[c["chunk_id"]]["work"], "title": c["filename"],
                                 "author": meta[c["chunk_id"]]["author"], "year": meta[c["chunk_id"]]["year"],
                                 "score": round(float(c.get("score", 0.0)), 4), "text": c["content"]}
                                for sid, c in zip(sources, chunks)],
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                with path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                works = ", ".join(f"{s['work'][:22]} {s['year']}" for s in record["sources"])
                print(f"{label} {n:3}/{len(todo)} {record['seconds']:5.0f}s {row['claim']}.{row['phrasing']} {record['verdict']:13} | {works}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dates", default="")
    parser.add_argument("--claims", default="")
    parser.add_argument("--phrasings", default="")
    parser.add_argument("--llm", default="qwen3:8b")
    run(parser.parse_args())
