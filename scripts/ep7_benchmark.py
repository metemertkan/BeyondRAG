#!/usr/bin/env python3
"""Episode 7 benchmark: what caching, routing and token optimization save, and what they cost.

Sections (``--sections``, comma-separated), in the order they are meant to run:

  answers        Every question in ep7_fixtures through the real service with the large model:
                 the answer, its tokens and model time. Saved to answers.json; the cache and
                 routing sections replay it instead of asking the model again.
  embeddings     Ingestion of the Episode 3 lab documents: one call per chunk, batched, batched
                 with a cold and a warm cache; re-ingesting a revised document; a demo restart.
  prefix         Ollama's prefix reuse on the large model: cold, warm, two prompts taking turns.
  cache_safety   A naive answer cache (question only) against the scoped one: another role,
                 another tenant, a revised document, an erased document.
  semantic       Similarity of paraphrases, near-misses and unrelated questions with bge-m3;
                 served answers right or wrong at each threshold, with and without the guard;
                 plus Quora question pairs (downloaded, not committed).
  routing_tasks  Quality per task per model tier: answers, verifier, guardrail judge, query
                 rewrite, concept extraction.
  cascade        Answers by the small model alone, the cascade, extractive-first, and the large model.
  tokens         Prompt composition, compact headers, a context budget, an answer cap, compact
                 verification: tokens, time and correctness.
  llm_cache      Repeated questions with query rewriting and verification, with the call cache.
  workload       Simulated traffic replayed over answers.json for each cache policy and repeat rate.
  live           The same traffic, shorter, through the real service and Redis, to check the replay.
  overhead       Cache lookup latency in Redis.

Needs Elasticsearch, Ollama (bge-m3 and the qwen3 tiers) and Redis on ``--redis``.
Everything goes into indexes named ``beyondrag-ep7-bench*``, keys under ``beyondrag-ep7-bench``
in Redis, and a workspace folder of its own; ``--cleanup`` deletes them.

    uv run scripts/ep7_benchmark.py --out /tmp/ep7 --sections answers,semantic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep7_fixtures as fx  # noqa: E402

PREFIX = "beyondrag-ep7-bench"


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.update(
        {
            "RAG_INDEX_PREFIX": PREFIX,
            "RAG_WORKSPACE_DIR": str(args.workspace),
            "RAG_EMBEDDING_PROVIDER": "ollama",
            "RAG_EMBEDDING_MODEL": "bge-m3",
            "RAG_EMBEDDING_DIMS": "1024",
            "OLLAMA_MODEL": args.large,
            "RAG_MODEL_SMALL": args.small,
            "RAG_MODEL_MEDIUM": args.medium,
            "RAG_MODEL_LARGE": args.large,
            "USE_LOCAL_ONLY": "true",
            "RAG_AUTH_MODE": "demo",
            "RAG_AUTH_SECRET": "ep7-benchmark-signing-secret-0123456789abcdef",
            "RAG_GUARDRAILS": "off",
            "RAG_QUERY_MODE": "off",
            "RAG_RERANK": "false",
            "RAG_MULTI_HOP": "false",
            "RAG_VERIFY": "false",
            "RAG_ANSWER_CACHE": "off",
            "RAG_LLM_CACHE": "off",
            "RAG_ROUTING": "off",
            "REDIS_URL": args.redis,
            "RAG_CACHE_PREFIX": PREFIX,
            "OTEL_SERVICE_NAME": "beyondrag-bench7",
            "OTEL_METRIC_EXPORT_INTERVAL": "5000",
        }
    )
    if args.otlp:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = args.otlp
    else:
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    args.workspace.mkdir(parents=True, exist_ok=True)


# --- Shared helpers ---------------------------------------------------------------------


def pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 3)


def contains(answer: str, expected: Optional[str]) -> bool:
    if expected is None:
        return False
    canon = lambda s: str(s).casefold().replace(" ", "").replace(",", "")
    return canon(expected) in canon(answer)


def abstained(answer: str) -> bool:
    text = answer.casefold()
    text = text.replace("’", "'")
    return any(marker in text for marker in (
        "not_found", "couldn't find", "could not find", "can't find", "cannot find", "no matching documents",
        "don't contain", "do not contain", "doesn't contain", "does not contain", "not stated", "doesn't say", "does not say",
        "not mentioned", "isn't mentioned", "don't mention", "do not mention", "doesn't mention", "does not mention",
        "not specified", "don't specify", "do not specify", "doesn't specify", "does not specify", "no information",
        "not provided", "don't provide", "do not provide", "doesn't provide", "does not provide", "not in the",
        "access blocked",
    ))


def correct(answer: str, expected: Optional[str]) -> bool:
    """Right if it contains the fact; for an unanswerable question, right if it doesn't pretend."""
    return contains(answer, expected) if expected is not None else abstained(answer)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def ollama(model: str, messages: List[Dict[str, str]], **options: Any) -> Dict[str, Any]:
    import httpx

    from src.config import OLLAMA_BASE_URL

    body = {"model": model, "messages": messages, "stream": False, "think": False, "options": {"temperature": 0, **options}}
    started = time.perf_counter()
    reply = httpx.post(f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=body, timeout=900).json()
    reply["wall_seconds"] = round(time.perf_counter() - started, 3)
    return reply


def summarize(body: Dict[str, Any], expected: Optional[str]) -> Dict[str, Any]:
    info = body.get("telemetry") or {}
    calls = [c for c in info.get("llm_calls", []) if not c.get("cached")]
    return {
        "outcome": info.get("outcome"),
        "answer": str(body.get("answer", ""))[:600],
        "correct": correct(str(body.get("answer", "")), expected),
        "input_tokens": (info.get("tokens") or {}).get("input", 0),
        "output_tokens": (info.get("tokens") or {}).get("output", 0),
        "by_stage": (info.get("tokens") or {}).get("by_stage", {}),
        "model_seconds": round(sum(c["duration_ms"] for c in calls if c["operation"] == "chat") / 1000, 3),
        "embedding_seconds": round(sum(c["duration_ms"] for c in calls if c["operation"] == "embeddings") / 1000, 3),
        "duration_s": round((info.get("duration_ms") or 0) / 1000, 3),
        "models": sorted({c["model"] for c in calls if c["operation"] == "chat"}),
        "routes": info.get("routes", []),
        "events": [e["event"] for e in info.get("events", [])],
        "saved": info.get("saved"),
        "cache": body.get("cache"),
    }


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from src.rag import telemetry

        self.args = args
        self.spans = InMemorySpanExporter()
        telemetry.setup_telemetry(span_processors=[SimpleSpanProcessor(self.spans)])
        from src.rag.cache_store import get_cache_store
        from src.rag.service import RBACKnowledgeBaseService

        self.store = get_cache_store()
        assert self.store.kind == "redis", "The benchmark needs Redis (--redis)."
        self.service = RBACKnowledgeBaseService()
        self.seed()

    def seed(self) -> None:
        from src.rag.seeding import DEMO_TENANT_SAMPLES, create_pdf_bytes

        service = self.service
        service._index_manager.ensure_index()
        have = {d["filename"] for d in service.list_documents(tenant="acme")}
        if have and not self.args.reseed and set(fx.EXTRA_ACME_DOCS) <= have:
            return
        service._index_manager.reset_index()
        service._repository.forget_all_documents()
        for tenant, samples in DEMO_TENANT_SAMPLES.items():
            for sample in samples:
                service.ingest_pdf_bytes(filename=sample["filename"], pdf_bytes=create_pdf_bytes(sample["filename"], sample["pages"]),
                                         tags=sample["tags"], tenant=tenant, extract_concepts=False)
        for name in fx.EXTRA_ACME_DOCS:
            service.ingest_pdf_bytes(filename=name, pdf_bytes=(ROOT / "eval" / "corpus" / name).read_bytes(), tags=[], tenant="acme", extract_concepts=False)

    def ask(self, question: str, tenant: str, *, mode: str = "llm", options: Any = None, roles: Sequence[str] = fx.ROLES) -> Dict[str, Any]:
        from src.rag.pipeline import PipelineOptions

        return self.service.query_and_chat(question, user_roles=list(roles), synthesis_mode=mode,
                                           pipeline_options=options or PipelineOptions(), tenant=tenant)

    def answers(self) -> Dict[str, Dict[str, Any]]:
        path = self.args.out / "answers.json"
        if not path.exists():
            raise SystemExit("Run the 'answers' section first.")
        rows = json.loads(path.read_text())["rows"]
        for row in rows:  # Scored again on load, so a fix to the scoring needs no new model calls.
            row["correct"] = correct(row["answer"], row["expected"])
        return {row["q"]: row for row in rows}


def with_env(**values: str) -> Callable[[Callable[[], Any]], Any]:
    """Run a function with some environment variables set, then restore them."""
    def runner(fn: Callable[[], Any]) -> Any:
        old = {k: os.environ.get(k) for k in values}
        os.environ.update(values)
        try:
            return fn()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return runner


# --- answers ----------------------------------------------------------------------------


def run_answers(bench: Bench) -> Dict[str, Any]:
    rows = []
    questions = fx.all_questions()
    for i, item in enumerate(questions, 1):
        body = bench.ask(str(item["q"]), str(item["tenant"]))
        row = {**item, **summarize(body, item["expected"])}  # type: ignore[arg-type]
        rows.append(row)
        print(f"  {i:>3}/{len(questions)} {item['kind']:<10} correct={row['correct']!s:<5} in={row['input_tokens']:>4} out={row['output_tokens']:>3} {row['model_seconds']:>6.1f}s  {str(item['q'])[:60]}", flush=True)
    by_kind = {}
    for kind in ("base", "paraphrase", "near_miss"):
        mine = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"questions": len(mine), "correct": sum(r["correct"] for r in mine),
                         "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in mine), 1),
                         "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in mine), 1),
                         "mean_model_seconds": round(statistics.mean(r["model_seconds"] for r in mine), 2)}
    return {"model": bench.args.large, "by_kind": by_kind, "rows": rows}


# --- embeddings --------------------------------------------------------------------------


def run_embeddings(bench: Bench) -> Dict[str, Any]:
    from src.rag.embedding_cache import CachingEmbedder
    from src.rag.seeding import DEMO_TENANT_SAMPLES, create_pdf_bytes

    service = bench.service
    lab = json.loads((ROOT / "config" / "ep3_demo_docs.json").read_text())[: bench.args.lab_docs]
    # The lab documents are one chunk each, which leaves batching nothing to batch. Grouped
    # eight to a file, each file is several chunks, like a real policy document.
    docs = [
        {"filename": f"lab_group_{i // 8:02d}.md", "text": "\n\n".join(d["text"] for d in lab[i : i + 8]), "tags": []}
        for i in range(0, len(lab), 8)
    ]
    embedder = service._embedder
    assert isinstance(embedder, CachingEmbedder)

    def ingest_all(tenant: str) -> Dict[str, Any]:
        bench.spans.clear()
        started = time.perf_counter()
        for doc in docs:
            service.ingest_pdf_bytes(filename=doc["filename"], pdf_bytes=doc["text"].encode("utf-8"), tags=doc.get("tags") or [],
                                     tenant=tenant, extract_concepts=False)
        calls = [s for s in bench.spans.get_finished_spans() if s.attributes.get("gen_ai.operation.name") == "embeddings"]
        return {"seconds": round(time.perf_counter() - started, 2), "embedding_calls": len(calls),
                "embedding_seconds": round(sum((s.end_time - s.start_time) / 1e9 for s in calls), 2),
                "embedding_tokens": sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in calls)}

    arms: Dict[str, Any] = {}
    bench.store.delete(bench.store.keys(f"{PREFIX}:emb:*"))
    arms["one_call_per_chunk"] = with_env(RAG_EMBEDDING_BATCH="off")(lambda: _uncached_ingest(service, docs, "emb-single", bench))
    arms["batched_no_cache"] = with_env(RAG_EMBEDDING_BATCH="on")(lambda: _uncached_ingest(service, docs, "emb-batched", bench))
    arms["batched_cold_cache"] = ingest_all("emb-cached")
    # A rebuild: the index reset and every document loaded again (what a demo restart does).
    service._index_manager.reset_index()
    service._repository.forget_all_documents()
    arms["rebuild_warm_cache"] = ingest_all("emb-cached")
    for name, arm in arms.items():
        print(f"  {name:<22} {arm}", flush=True)
    bench.args.reseed = True
    bench.seed()

    # A revised document: one section changed. How much of it is embedded again?
    sample = DEMO_TENANT_SAMPLES["acme"][1]
    pages = [list(page) for page in sample["pages"]]
    pages[0][-1] = pages[0][-1].replace("5th business day", "7th business day")
    bench.spans.clear()
    service.ingest_pdf_bytes(filename="revision_v1.pdf", pdf_bytes=create_pdf_bytes("revision", sample["pages"]), tags=["finance"],
                             tenant="emb-revision", source_key="treasury-policy", extract_concepts=False)
    first = [s for s in bench.spans.get_finished_spans() if s.attributes.get("gen_ai.operation.name") == "embeddings"]
    bench.spans.clear()
    service.ingest_pdf_bytes(filename="revision_v2.pdf", pdf_bytes=create_pdf_bytes("revision", pages), tags=["finance"],
                             tenant="emb-revision", source_key="treasury-policy", extract_concepts=False)
    second = [s for s in bench.spans.get_finished_spans() if s.attributes.get("gen_ai.operation.name") == "embeddings"]
    chunks_v2 = [c for c in service._repository.es_manager.client.search(
        index=service._index_manager.ensure_index("emb-revision"), size=50,
        query={"bool": {"filter": [{"term": {"tenant": "emb-revision"}}, {"term": {"is_current": True}}]}})["hits"]["hits"]]
    revision = {"chunks_in_v2": len(chunks_v2), "embedding_calls_v1": len(first), "embedding_calls_v2": len(second),
                "texts_embedded_v2": sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) > 0 for s in second)}

    return {"documents": len(docs), "source_documents": len(lab), "arms": arms, "revision": revision,
            "note": "Concept extraction is off here, so only embedding costs are compared."}


def _uncached_ingest(service: Any, docs: List[Dict[str, Any]], tenant: str, bench: Bench) -> Dict[str, Any]:
    """Ingest with the cache wrapper bypassed: the embedder as it was before this episode."""
    ingestor = service._ingestor
    wrapper = ingestor._embedder
    ingestor._embedder = wrapper._inner
    try:
        bench.spans.clear()
        started = time.perf_counter()
        for doc in docs:
            service.ingest_pdf_bytes(filename=doc["filename"], pdf_bytes=doc["text"].encode("utf-8"), tags=doc.get("tags") or [],
                                     tenant=tenant, extract_concepts=False)
        calls = [s for s in bench.spans.get_finished_spans() if s.attributes.get("gen_ai.operation.name") == "embeddings"]
        return {"seconds": round(time.perf_counter() - started, 2), "embedding_calls": len(calls),
                "embedding_seconds": round(sum((s.end_time - s.start_time) / 1e9 for s in calls), 2),
                "embedding_tokens": sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in calls)}
    finally:
        ingestor._embedder = wrapper


# --- prefix ------------------------------------------------------------------------------


def _generation_messages(bench: Bench, question: str, tenant: str = "acme") -> List[Dict[str, str]]:
    """The exact generation prompt the service would send, captured instead of sent."""
    from langchain_core.runnables import RunnableLambda

    synth = bench.service._synthesizer
    captured: Dict[str, Any] = {}

    def factory(**_: Any) -> Any:
        def capture(prompt: Any) -> Any:
            captured["messages"] = prompt.to_messages()
            from types import SimpleNamespace
            return SimpleNamespace(content="NOT_FOUND")
        return RunnableLambda(capture)

    original = synth._llm_factory
    synth._llm_factory = factory
    try:
        bench.ask(question, tenant)
    finally:
        synth._llm_factory = original
    role = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role[m.type], "content": m.content} for m in captured["messages"]]


def run_prefix(bench: Bench) -> Dict[str, Any]:
    model = bench.args.large
    qa = _generation_messages(bench, fx.CASES[3]["q"])
    qb = _generation_messages(bench, fx.CASES[8]["q"])
    verify_prompt = [{"role": "user", "content": "Check each numbered sentence against the sources it cites. " + qa[-1]["content"][:1800]}]

    def run(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        r = ollama(model, messages, num_predict=1)
        return {"prompt_tokens": r.get("prompt_eval_count"), "prefill_s": round(r.get("prompt_eval_duration", 0) / 1e9, 2),
                "load_s": round(r.get("load_duration", 0) / 1e9, 2), "wall_s": r["wall_seconds"]}

    steps = []
    ollama(bench.args.small, [{"role": "user", "content": "hi"}], num_predict=1)  # evict the large model's cache
    for label, messages in (
        ("question A, cold (model just loaded)", qa),
        ("question A again: identical prompt", qa),
        ("question B: same system prompt, other context", qb),
        ("verification prompt on the same model", verify_prompt),
        ("question B again, after the verifier ran", qb),
    ):
        steps.append({"step": label, **run(messages)})
        print(f"  {label:<46} {steps[-1]}", flush=True)
    system_tokens = ollama(model, [qa[0]], num_predict=1).get("prompt_eval_count")
    return {"model": model, "system_prompt_tokens": system_tokens, "steps": steps,
            "note": "num_predict=1, so the time is almost all prefill. prompt_eval_count does not shrink when the prefix is reused."}


# --- cache_safety ---------------------------------------------------------------------------


class NaiveAnswerCache:
    """What a quick answer cache looks like: question in, answer out, nothing else in the key."""

    def __init__(self) -> None:
        self.entries: Dict[str, Dict[str, Any]] = {}

    def ask(self, bench: Bench, question: str, tenant: str, roles: Sequence[str]) -> Dict[str, Any]:
        key = " ".join(question.casefold().split())
        if key in self.entries:
            return {**self.entries[key], "served_from_cache": True}
        body = bench.ask(question, tenant, mode="fast", roles=roles)
        self.entries[key] = {"answer": body["answer"]}
        return {"answer": body["answer"], "served_from_cache": False}


def run_cache_safety(bench: Bench) -> Dict[str, Any]:
    from src.rag.answer_cache import AnswerCache, CachePolicy
    from src.rag.seeding import create_pdf_bytes

    service = bench.service
    scoped = AnswerCache(CachePolicy(exact=True), store=bench.store, embedder=service._embedder)
    service._answer_cache = scoped
    tenant = f"safety-{uuid.uuid4().hex[:6]}"
    policy_v1 = [["RESTRICTED — FINANCE. Minimum Cash Buffer: Treasury must keep an intraday minimum cash reserve of HKD 450,000,000. "
                  "Contact: treasury.lead@example.com for exceptions."]]
    policy_v2 = [["RESTRICTED — FINANCE. Minimum Cash Buffer: Treasury must keep an intraday minimum cash reserve of HKD 520,000,000. "
                  "Contact: treasury.lead@example.com for exceptions."]]
    other = f"{tenant}-other"
    service.ingest_pdf_bytes(filename="reserve.pdf", pdf_bytes=create_pdf_bytes("reserve", policy_v1), tags=["finance"], tenant=tenant,
                             source_key="reserve-policy", extract_concepts=False)
    service.ingest_pdf_bytes(filename="other_reserve.pdf", pdf_bytes=create_pdf_bytes("other", [["Minimum Cash Buffer: Treasury must keep an intraday minimum cash reserve of EUR 38,000,000."]]),
                             tags=["finance"], tenant=other, extract_concepts=False)
    q = "What minimum cash reserve must Treasury keep?"
    finance, public = ["finance"], []
    scenarios = []

    def check(name: str, expect_absent: str, run_naive: Callable[[], Dict[str, Any]], run_scoped: Callable[[], Dict[str, Any]], leak_means: str) -> None:
        naive_answer, scoped_answer = run_naive(), run_scoped()
        scenarios.append({
            "scenario": name, "leak_means": leak_means,
            "naive": {"answer": naive_answer["answer"][:200], "from_cache": naive_answer.get("served_from_cache"), "leaked": expect_absent in naive_answer["answer"]},
            "scoped": {"answer": scoped_answer["answer"][:200], "from_cache": bool(scoped_answer.get("cache")), "leaked": expect_absent in scoped_answer["answer"]},
        })
        print(f"  {name:<32} naive leaked={scenarios[-1]['naive']['leaked']!s:<5} scoped leaked={scenarios[-1]['scoped']['leaked']}", flush=True)

    naive = NaiveAnswerCache()
    ask_scoped = lambda question, t, roles: service.query_and_chat(question, user_roles=roles, synthesis_mode="fast", tenant=t)
    # Fill both caches as a finance user.
    naive.ask(bench, q, tenant, finance)
    ask_scoped(q, tenant, finance)
    check("public user, same question", "450,000,000", lambda: naive.ask(bench, q, tenant, public), lambda: ask_scoped(q, tenant, public),
          "a user without the finance role reads the finance figure")
    check("other tenant, same question", "450,000,000", lambda: naive.ask(bench, q, other, finance), lambda: ask_scoped(q, other, finance),
          "another company reads this company's figure")
    service.ingest_pdf_bytes(filename="reserve_v2.pdf", pdf_bytes=create_pdf_bytes("reserve", policy_v2), tags=["finance"], tenant=tenant,
                             source_key="reserve-policy", extract_concepts=False)
    check("after the policy was revised", "450,000,000", lambda: naive.ask(bench, q, tenant, finance), lambda: ask_scoped(q, tenant, finance),
          "the superseded figure is served after the new version was published")
    ask_scoped(q, tenant, finance)  # cache the 520 answer
    for doc in service.list_documents(tenant=tenant):
        service.delete_document(doc["doc_id"], tenant=tenant)
    entries_after_delete = len(bench.store.keys(f"{PREFIX}:ans:{tenant}:*"))
    check("after the documents were deleted", "treasury.lead@example.com", lambda: naive.ask(bench, q, tenant, finance),
          lambda: ask_scoped(q, tenant, finance), "text from erased documents is still served")
    service._answer_cache = AnswerCache(store=bench.store, embedder=service._embedder)
    return {"scenarios": scenarios, "scoped_cache_entries_left_after_delete": entries_after_delete,
            "note": "Fast (extractive) answers, so no model time; the cache logic is the same for LLM answers."}


# --- semantic ---------------------------------------------------------------------------


def fetch_qqp(cache: Path, rows: int) -> List[Dict[str, Any]]:
    import httpx

    if cache.exists():
        return json.loads(cache.read_text())
    out: List[Dict[str, Any]] = []
    offset = 0
    while len(out) < rows:
        for attempt in range(5):
            r = httpx.get("https://datasets-server.huggingface.co/rows",
                          params={"dataset": "nyu-mll/glue", "config": "qqp", "split": "validation", "offset": offset, "length": 100}, timeout=60)
            if r.status_code < 500:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        out += [item["row"] for item in r.json()["rows"]]
        offset += 100
    cache.write_text(json.dumps(out[:rows]))
    return out[:rows]


def run_semantic(bench: Bench) -> Dict[str, Any]:
    from src.rag.answer_cache import guard_allows

    embed = bench.service._embedder.embed
    answers = bench.answers()
    questions = fx.all_questions()
    vectors = {str(q["q"]): embed(str(q["q"])) for q in questions}
    bases = {c["id"]: c for c in fx.CASES}
    pairs = []
    # Each incoming question against every base question of its tenant: the cache holds the bases.
    for item in questions:
        if item["kind"] == "base":
            continue
        for case in fx.CASES:
            if case["tenant"] != item["tenant"]:
                continue
            cached_answer = answers.get(case["q"], {}).get("answer", "")
            same_case = case["id"] == item["case"]
            kind = item["kind"] if same_case else "unrelated"
            pairs.append({
                "incoming": item["q"], "cached": case["q"], "kind": kind,
                "similarity": round(cosine(vectors[str(item["q"])], vectors[case["q"]]), 4),
                "guard_allows": guard_allows(str(item["q"]), case["q"])[0],
                # Serving the cached answer is right only if it answers the incoming question.
                "serving_is_right": correct(cached_answer, item["expected"]) if cached_answer else (same_case and item["kind"] == "paraphrase"),
            })
    thresholds = [round(0.70 + 0.02 * i, 2) for i in range(15)]
    sweep = []
    for t in thresholds:
        for guard in (False, True):
            # Per incoming question, the best-scoring cached base above the threshold is served.
            served = {}
            for pair in pairs:
                if pair["similarity"] >= t and (pair["guard_allows"] or not guard):
                    best = served.get(pair["incoming"])
                    if best is None or pair["similarity"] > best["similarity"]:
                        served[pair["incoming"]] = pair
            paraphrases = [q for q in questions if q["kind"] == "paraphrase"]
            near = [q for q in questions if q["kind"] == "near_miss"]
            hits = [served[q["q"]] for q in paraphrases if q["q"] in served]
            sweep.append({
                "threshold": t, "guard": guard,
                "paraphrase_hits": len(hits), "paraphrases": len(paraphrases),
                "paraphrase_hits_right": sum(p["serving_is_right"] for p in hits),
                "near_miss_served": sum(1 for q in near if q["q"] in served),
                "near_miss_served_wrong": sum(1 for q in near if q["q"] in served and not served[q["q"]]["serving_is_right"]),
                "near_misses": len(near),
                "wrong_answers_served": sum(1 for p in served.values() if not p["serving_is_right"]),
                "served": len(served),
            })
    for row in sweep:
        print(f"  t={row['threshold']:.2f} guard={row['guard']!s:<5} paraphrase hits {row['paraphrase_hits']}/{row['paraphrases']}  near-miss served {row['near_miss_served']}/{row['near_misses']}  wrong {row['wrong_answers_served']}/{row['served']}", flush=True)
    # Quora question pairs: duplicates should be served, non-duplicates should not.
    qqp = fetch_qqp(bench.args.workspace / "qqp_validation.json", bench.args.qqp)
    qqp_rows = []
    for row in qqp:
        a, b = row["question1"], row["question2"]
        qqp_rows.append({"duplicate": bool(row["label"]), "similarity": cosine(embed(a), embed(b)), "guard_allows": guard_allows(a, b)[0]})
    qqp_sweep = []
    for t in thresholds:
        for guard in (False, True):
            served = [r for r in qqp_rows if r["similarity"] >= t and (r["guard_allows"] or not guard)]
            dup = sum(r["duplicate"] for r in qqp_rows)
            qqp_sweep.append({"threshold": t, "guard": guard, "served": len(served), "served_duplicates": sum(r["duplicate"] for r in served),
                              "duplicates": dup, "pairs": len(qqp_rows),
                              "precision": round(sum(r["duplicate"] for r in served) / len(served), 3) if served else None,
                              "recall": round(sum(r["duplicate"] for r in served) / dup, 3) if dup else None})
    dist = {kind: [p["similarity"] for p in pairs if p["kind"] == kind] for kind in ("paraphrase", "near_miss", "unrelated")}
    return {
        "embedding_model": "bge-m3",
        "similarity": {k: {"p10": pct(v, .1), "p50": pct(v, .5), "p90": pct(v, .9), "max": max(v) if v else None} for k, v in dist.items()},
        "sweep": sweep, "qqp": {"pairs": len(qqp_rows), "sweep": qqp_sweep},
        "near_miss_pairs": [p for p in pairs if p["kind"] == "near_miss"],
    }


# --- routing_tasks -------------------------------------------------------------------------


def run_routing_tasks(bench: Bench) -> Dict[str, Any]:
    from langchain_ollama import ChatOllama

    from src.config import OLLAMA_BASE_URL
    from src.rag.guardrails import GuardrailEngine, GuardrailPolicy
    from src.rag.telemetry import genai_handler
    from src.rag.trust import CitedSentence, LlmClaimVerifier

    tiers = {"small": bench.args.small, "medium": bench.args.medium, "large": bench.args.large}
    out: Dict[str, Any] = {"tiers": tiers}

    def factory_for(model: str) -> Callable[..., Any]:
        def factory(**kwargs: Any) -> Any:
            return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=kwargs.get("temperature", 0.0), reasoning=False,
                              callbacks=[genai_handler()], client_kwargs={"timeout": 600}, num_predict=kwargs.get("max_tokens"))
        return factory

    def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        try:
            return fn(), round(time.perf_counter() - started, 2)
        except Exception as exc:  # A failure is a result here.
            return exc, round(time.perf_counter() - started, 2)

    # 1. Answers: every base question, each tier writing the answer.
    synth = bench.service._synthesizer
    original = synth._llm_factory
    gen: Dict[str, Any] = {}
    for tier, model in tiers.items():
        if tier == "large":
            answers = bench.answers()
            rows = [{"case": c["id"], "correct": answers[c["q"]]["correct"], "seconds": answers[c["q"]]["model_seconds"],
                     "output_tokens": answers[c["q"]]["output_tokens"]} for c in fx.CASES]
        else:
            synth._llm_factory = factory_for(model)
            rows = []
            try:
                for case in fx.CASES:
                    body = bench.ask(case["q"], case["tenant"])
                    s = summarize(body, case["expected"])
                    rows.append({"case": case["id"], "correct": s["correct"], "seconds": s["model_seconds"], "output_tokens": s["output_tokens"],
                                 "outcome": s["outcome"]})
            finally:
                synth._llm_factory = original
        gen[tier] = {"model": model, "correct": sum(r["correct"] for r in rows), "questions": len(rows),
                     "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 2), "rows": rows}
        print(f"  generate {tier:<6} {model:<12} {gen[tier]['correct']}/{len(rows)} right, {gen[tier]['mean_seconds']} s", flush=True)
    out["generate"] = gen

    # 2. Verifier: a true and a false sentence per base question, against the chunk text.
    verify: Dict[str, Any] = {}
    cases = []
    answers = bench.answers()
    for case in fx.CASES:
        body = answers[case["q"]]
        source = next((c for c in bench.ask(case["q"], case["tenant"], mode="fast")["authorized_chunks"]), None)
        if not source:
            continue
        cases.append((case, source["content"]))
    for tier, model in tiers.items():
        verifier = LlmClaimVerifier(factory_for(model))
        right, total, seconds = 0, 0, []
        for case, text in cases:
            true_s = CitedSentence(f"The answer to '{case['q']}' is {case['expected']}.", ["S1"])
            false_s = CitedSentence(f"The answer to '{case['q']}' is not {case['expected']}; the documents give a different figure.", ["S1"])
            result, took = timed(lambda: verifier.verify([true_s, false_s], {"S1": text}))
            seconds.append(took)
            if isinstance(result, Exception):
                total += 2
                continue
            right += (getattr(result.get(1), "label", None) == "supported") + (getattr(result.get(2), "label", None) in {"contradicted", "not_stated"})
            total += 2
        verify[tier] = {"model": model, "right": right, "sentences": total, "mean_seconds": round(statistics.mean(seconds), 2)}
        print(f"  verify   {tier:<6} {model:<12} {right}/{total}, {verify[tier]['mean_seconds']} s", flush=True)
    out["verify"] = verify

    # 3. Guardrail judge on the Episode 6 labelled set.
    sys.path.insert(0, str(ROOT / "scripts"))
    import ep6_fixtures

    judge: Dict[str, Any] = {}
    for tier, model in tiers.items():
        engine = GuardrailEngine(GuardrailPolicy(enabled=True, judge="llm", pii_question="off"), llm_factory=factory_for(model))
        tp = fp = fn = tn = 0
        seconds = []
        for text, label in ep6_fixtures.GUARDRAIL_SET:
            verdict, took = timed(lambda: engine.judge(text))
            seconds.append(took)
            flagged = verdict in {"prompt_injection", "investment_advice"}
            positive = label != "benign"
            tp += flagged and positive
            fp += flagged and not positive
            fn += (not flagged) and positive
            tn += (not flagged) and not positive
        judge[tier] = {"model": model, "caught": tp, "attacks": tp + fn, "ordinary_flagged": fp, "ordinary": fp + tn,
                       "mean_seconds": round(statistics.mean(seconds), 2)}
        print(f"  judge    {tier:<6} {model:<12} caught {tp}/{tp + fn}, flagged {fp}/{fp + tn} ordinary", flush=True)
    out["judge"] = judge

    # 4. Query rewrite: does searching with the rewrites find the expected fact in the top 4?
    from src.rag.pipeline import PipelineOptions
    from src.rag.query_transform import LlmQueryTransformer

    rewrite: Dict[str, Any] = {}
    pipeline = bench.service._get_pipeline()
    baseline = sum(
        any(contains(c["content"], case["expected"]) for c in bench.ask(case["q"], case["tenant"], mode="fast")["authorized_chunks"])
        for case in fx.CASES
    )
    rewrite["none"] = {"found_in_top4": baseline, "questions": len(fx.CASES)}
    for tier, model in tiers.items():
        pipeline._transformer = LlmQueryTransformer(factory_for(model))
        found, seconds = 0, []
        for case in fx.CASES:
            started = time.perf_counter()
            body = bench.ask(case["q"], case["tenant"], mode="fast", options=PipelineOptions(query_mode="multi"))
            seconds.append(time.perf_counter() - started)
            found += any(contains(c["content"], case["expected"]) for c in body["authorized_chunks"])
        rewrite[tier] = {"model": model, "found_in_top4": found, "questions": len(fx.CASES), "mean_seconds": round(statistics.mean(seconds), 2)}
        print(f"  rewrite  {tier:<6} {model:<12} {found}/{len(fx.CASES)} found", flush=True)
    pipeline._transformer = LlmQueryTransformer()
    out["query_rewrite"] = rewrite

    # 5. Concept extraction at ingestion: valid output, per tier (the large model was measured in Episode 6).
    from src.rag.cross_lingual import extract_concepts_at_ingestion
    from src.rag.seeding import DEMO_TENANT_SAMPLES

    concepts: Dict[str, Any] = {}
    texts = [(s["filename"], " ".join(" ".join(p) for p in s["pages"])) for samples in DEMO_TENANT_SAMPLES.values() for s in samples]
    for tier in ("small", "medium"):
        model = tiers[tier]
        ok, seconds, counts = 0, [], []
        for filename, text in texts:
            bench.spans.clear()
            result, took = with_env(RAG_ROUTING="static", **{"RAG_ROUTE_CONCEPTS": tier})(
                lambda: timed(lambda: extract_concepts_at_ingestion(filename, text + f" [{uuid.uuid4().hex[:6]}]", force_refresh=True, tenant=f"concepts-{tier}")))
            seconds.append(took)
            if not isinstance(result, Exception) and result[0]:
                ok += 1
                counts.append(len(result[0]))
        concepts[tier] = {"model": model, "documents_with_concepts": ok, "documents": len(texts), "mean_seconds": round(statistics.mean(seconds), 1),
                          "mean_concepts": round(statistics.mean(counts), 1) if counts else 0}
        print(f"  concepts {tier:<6} {model:<12} {ok}/{len(texts)} documents, {concepts[tier]['mean_seconds']} s", flush=True)
    episode6 = ROOT.parent.parent / "ContentEngine" / "episodes" / "ep06-beyondrag" / "bench" / "concepts.json"
    if episode6.exists():
        e6 = json.loads(episode6.read_text())
        concepts["large"] = {"model": e6.get("llm"), "documents_with_concepts": e6["documents"] - e6["failed"], "documents": e6["documents"],
                             "mean_seconds": round(e6["total_seconds"] / e6["documents"], 1), "source": "Episode 6 concepts.json"}
    out["concepts"] = concepts
    return out


# --- cascade --------------------------------------------------------------------------------


def run_cascade(bench: Bench) -> Dict[str, Any]:
    from src.rag.pipeline import PipelineOptions

    answers = bench.answers()
    arms: Dict[str, Any] = {}
    configs = {
        "cascade_small_first": dict(RAG_ROUTING="cascade", RAG_CASCADE_FIRST="small"),
        "cascade_medium_first": dict(RAG_ROUTING="cascade", RAG_CASCADE_FIRST="medium"),
        "extractive_first": dict(RAG_ROUTE_EXTRACTIVE="on"),
    }
    for name, env in configs.items():
        rows = []
        for case in fx.CASES:
            options = PipelineOptions(rerank=True) if name == "extractive_first" else PipelineOptions()
            body = with_env(**env)(lambda: bench.ask(case["q"], case["tenant"], options=options))
            s = summarize(body, case["expected"])
            rows.append({"case": case["id"], **{k: s[k] for k in ("correct", "model_seconds", "input_tokens", "output_tokens", "models", "events", "outcome")}})
        escalated = sum("route_escalated" in r["events"] for r in rows)
        no_model = sum("route_extractive" in r["events"] for r in rows)
        arms[name] = {"correct": sum(r["correct"] for r in rows), "questions": len(rows), "escalated": escalated, "no_model": no_model,
                      "model_seconds": round(sum(r["model_seconds"] for r in rows), 1),
                      "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows), "rows": rows}
        print(f"  {name:<22} {arms[name]['correct']}/{len(rows)} right, escalated {escalated}, no model {no_model}, {arms[name]['model_seconds']} s", flush=True)
    large = [answers[c["q"]] for c in fx.CASES]
    arms["large_only"] = {"correct": sum(r["correct"] for r in large), "questions": len(large), "escalated": 0, "no_model": 0,
                          "model_seconds": round(sum(r["model_seconds"] for r in large), 1),
                          "input_tokens": sum(r["input_tokens"] for r in large), "output_tokens": sum(r["output_tokens"] for r in large),
                          "source": "answers.json"}
    return {"tiers": {"small": bench.args.small, "medium": bench.args.medium, "large": bench.args.large}, "arms": arms}


# --- tokens ---------------------------------------------------------------------------------


def run_tokens(bench: Bench) -> Dict[str, Any]:
    from src.rag.pipeline import PipelineOptions
    from src.rag.synthesizer import PromptPolicy

    synth = bench.service._synthesizer
    cases = fx.CASES[: bench.args.token_cases]
    # Prompt composition, measured by the model's own tokenizer.
    messages = _generation_messages(bench, cases[3]["q"])
    system = ollama(bench.args.large, [messages[0]], num_predict=1).get("prompt_eval_count")
    full = ollama(bench.args.large, messages, num_predict=1).get("prompt_eval_count")
    composition = {"system_prompt": system, "whole_prompt": full, "user_part": (full or 0) - (system or 0)}
    configs = {
        "baseline": (PromptPolicy(), {}),
        "compact_sources": (PromptPolicy(compact_sources=True), {}),
        "compact_plus_budget_300": (PromptPolicy(compact_sources=True, context_budget=300), {}),
        "compact_plus_answer_cap_96": (PromptPolicy(compact_sources=True, max_answer_tokens=96), {}),
        "verify_full": (PromptPolicy(), {"verify": True}),
        "verify_compact": (PromptPolicy(), {"verify": True, "compact": True}),
    }
    original = synth._prompt_policy
    arms = {}
    for name, (policy, extra) in configs.items():
        synth._prompt_policy = policy
        rows = []
        for case in cases:
            options = PipelineOptions(verify=extra.get("verify", False))
            env = {"RAG_VERIFY_COMPACT": "on" if extra.get("compact") else "off"}
            body = with_env(**env)(lambda: bench.ask(case["q"], case["tenant"], options=options))
            s = summarize(body, case["expected"])
            rows.append({"case": case["id"], **{k: s[k] for k in ("correct", "input_tokens", "output_tokens", "model_seconds", "by_stage")}})
        arms[name] = {"correct": sum(r["correct"] for r in rows), "questions": len(rows),
                      "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows), 1),
                      "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 1),
                      "mean_model_seconds": round(statistics.mean(r["model_seconds"] for r in rows), 2), "rows": rows}
        print(f"  {name:<28} {arms[name]['correct']}/{len(rows)} right  in={arms[name]['mean_input_tokens']} out={arms[name]['mean_output_tokens']} {arms[name]['mean_model_seconds']} s", flush=True)
    synth._prompt_policy = original
    return {"model": bench.args.large, "composition": composition, "arms": arms}


# --- llm_cache ------------------------------------------------------------------------------


def run_llm_cache(bench: Bench) -> Dict[str, Any]:
    from src.rag.pipeline import PipelineOptions

    bench.store.delete(bench.store.keys(f"{PREFIX}:llm:*"))
    cases = fx.CASES[: bench.args.llm_cache_cases]
    passes = []
    for label in ("first pass", "second pass, same questions"):
        rows = []
        for case in cases:
            body = with_env(RAG_LLM_CACHE="on")(lambda: bench.ask(case["q"], case["tenant"], options=PipelineOptions(query_mode="multi", verify=True)))
            s = summarize(body, case["expected"])
            rows.append({k: s[k] for k in ("correct", "input_tokens", "output_tokens", "model_seconds", "by_stage", "saved", "duration_s")})
        passes.append({"pass": label, "correct": sum(r["correct"] for r in rows),
                       "input_tokens": sum(r["input_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
                       "saved_input_tokens": sum((r["saved"] or {}).get("input_tokens", 0) for r in rows),
                       "model_seconds": round(sum(r["model_seconds"] for r in rows), 1),
                       "seconds": round(sum(r["duration_s"] for r in rows), 1), "rows": rows})
        print(f"  {label:<30} {passes[-1]['correct']}/{len(rows)} right, spent in={passes[-1]['input_tokens']}, saved in={passes[-1]['saved_input_tokens']}, {passes[-1]['seconds']} s", flush=True)
    return {"questions": len(cases), "passes": passes,
            "note": "Query rewriting and verification are temperature 0 and cached; generation (temperature 0.1) is not."}


# --- workload -------------------------------------------------------------------------------


def traffic(n: int, zipf_s: float, seed: int, variant_mix: Dict[str, float]) -> List[Dict[str, Any]]:
    """Questions drawn with a Zipf popularity over cases, and a mix of how each is worded."""
    rng = random.Random(seed)
    cases = list(fx.CASES)
    rng.shuffle(cases)
    weights = [1 / (rank ** zipf_s) for rank in range(1, len(cases) + 1)]
    out = []
    for _ in range(n):
        case = rng.choices(cases, weights)[0]
        kind = rng.choices(list(variant_mix), list(variant_mix.values()))[0]
        if kind == "base":
            item = {"q": case["q"], "expected": case["expected"]}
        elif kind == "paraphrase":
            item = {"q": rng.choice(case["paraphrases"]), "expected": case["expected"]}
        else:
            near = rng.choice(case["near"])
            item = {"q": near["q"], "expected": near["expected"]}
        out.append({**item, "tenant": case["tenant"], "kind": kind, "case": case["id"]})
    return out


def run_workload(bench: Bench) -> Dict[str, Any]:
    from src.rag.answer_cache import guard_allows, normalize_question

    import numpy as np

    answers = bench.answers()
    embed = bench.service._embedder.embed
    names = list(answers)
    matrix = np.asarray([embed(q) for q in names], dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    index = {q: i for i, q in enumerate(names)}
    sims = matrix @ matrix.T
    mix = {"base": 0.5, "paraphrase": 0.4, "near_miss": 0.1}
    policies = {"none": None, "exact": ("exact", 1.0, False)}
    for t in bench.args.thresholds:
        policies[f"semantic_{t:.2f}"] = ("semantic", t, False)
        policies[f"semantic_{t:.2f}_guard"] = ("semantic", t, True)
    results = []
    for zipf_s in bench.args.zipf:
        stream = traffic(bench.args.requests, zipf_s, seed=7, variant_mix=mix)
        uncached = {"input": 0, "output": 0, "seconds": 0.0}
        for req in stream:
            a = answers[req["q"]]
            uncached["input"] += a["input_tokens"]
            uncached["output"] += a["output_tokens"]
            uncached["seconds"] += a["model_seconds"]
        for name, policy in policies.items():
            cache: Dict[str, Dict[str, Any]] = {}  # (tenant, normalized) -> entry; fills on miss
            spent = {"input": 0, "output": 0, "seconds": 0.0}
            hits = wrong = right = 0
            for req in stream:
                key = (req["tenant"], normalize_question(req["q"]))
                served = None
                if policy is not None:
                    served = cache.get(key)
                    if served is None and policy[0] == "semantic":
                        best, best_sim = None, policy[1]
                        for (tenant, _), entry in cache.items():
                            if tenant != req["tenant"]:
                                continue
                            sim = float(sims[index[req["q"]], index[entry["q"]]])
                            if sim >= best_sim and (not policy[2] or guard_allows(req["q"], entry["q"])[0]):
                                best, best_sim = entry, sim
                        served = best
                if served is not None:
                    hits += 1
                    ok = correct(served["answer"], req["expected"])
                else:
                    a = answers[req["q"]]
                    spent["input"] += a["input_tokens"]
                    spent["output"] += a["output_tokens"]
                    spent["seconds"] += a["model_seconds"]
                    ok = a["correct"]
                    if policy is not None and a["outcome"] in {"answered", "answered_extractive"}:
                        cache[key] = {"q": req["q"], "answer": a["answer"]}
                right += ok
                wrong += (served is not None) and not ok and answers[req["q"]]["correct"]
            results.append({
                "zipf_s": zipf_s, "policy": name, "requests": len(stream), "hits": hits, "hit_rate": round(hits / len(stream), 3),
                "right": right, "wrong_from_cache": wrong,
                "input_tokens": spent["input"], "output_tokens": spent["output"], "model_seconds": round(spent["seconds"], 1),
                "saved_model_seconds": round(uncached["seconds"] - spent["seconds"], 1),
                "saved_share": round(1 - spent["seconds"] / uncached["seconds"], 3) if uncached["seconds"] else None,
                "repeat_share": round(1 - len({(r['tenant'], normalize_question(r['q'])) for r in stream}) / len(stream), 3),
            })
            r = results[-1]
            print(f"  zipf={zipf_s} {name:<22} hits {r['hit_rate']:.2f} saved {r['saved_share']} right {r['right']}/{r['requests']} wrong-from-cache {r['wrong_from_cache']}", flush=True)
    return {"mix": mix, "requests": bench.args.requests, "results": results,
            "note": "Replayed: misses cost what answers.json recorded for that question; a cached answer is right if it contains the incoming question's fact. "
                    "wrong_from_cache counts hits that were wrong where the model's own answer was right."}


# --- live -----------------------------------------------------------------------------------


def run_live(bench: Bench) -> Dict[str, Any]:
    from src.rag.answer_cache import AnswerCache, CachePolicy

    service = bench.service
    for key in bench.store.keys(f"{PREFIX}:ans*"):
        bench.store.delete([key])
    t = bench.args.live_threshold
    service._answer_cache = AnswerCache(CachePolicy(exact=True, semantic=True, threshold=t, guard=True), store=bench.store, embedder=service._embedder)
    stream = traffic(bench.args.live_requests, bench.args.zipf[-1], seed=11, variant_mix={"base": 0.5, "paraphrase": 0.4, "near_miss": 0.1})
    rows = []
    for i, req in enumerate(stream, 1):
        body = bench.ask(req["q"], req["tenant"])
        s = summarize(body, req["expected"])
        rows.append({**req, **{k: s[k] for k in ("outcome", "correct", "model_seconds", "input_tokens", "saved", "cache", "duration_s")}})
        print(f"  {i:>3} {s['outcome']:<22} correct={s['correct']!s:<5} {s['duration_s']:>6.2f}s {req['q'][:60]}", flush=True)
    service._answer_cache = AnswerCache(store=bench.store, embedder=service._embedder)
    hits = [r for r in rows if r["outcome"] == "answered_cached"]
    return {"threshold": t, "requests": len(rows), "hits": len(hits), "right": sum(r["correct"] for r in rows),
            "hit_latency_p50_s": pct([r["duration_s"] for r in hits], .5),
            "miss_latency_p50_s": pct([r["duration_s"] for r in rows if r["outcome"] != "answered_cached"], .5),
            "model_seconds": round(sum(r["model_seconds"] for r in rows), 1),
            "saved_model_seconds": round(sum((r["saved"] or {}).get("model_seconds", 0) for r in rows), 1), "rows": rows}


# --- overhead -------------------------------------------------------------------------------


def run_overhead(bench: Bench) -> Dict[str, Any]:
    from src.rag.answer_cache import AnswerCache, CachePolicy

    cache = AnswerCache(CachePolicy(exact=True, semantic=True, threshold=0.99), store=bench.store, embedder=bench.service._embedder)
    tenant = f"overhead-{uuid.uuid4().hex[:6]}"
    sizes = {}
    for n in (100, 1000):
        for i in range(len(sizes) and 100 or 0, n):
            ctx = cache.context(f"synthetic question number {i}", tenant, ["x"], mode="llm", options={}, top_k=4, profile=None, model="m")
            ctx.vector = [math.sin(i * (j + 1)) for j in range(1024)]
            cache.store_answer(ctx, {"answer": "a", "authorized_chunks": []}, {})
        exact, semantic = [], []
        for i in range(200):
            ctx = cache.context(f"synthetic question number {i % n}", tenant, ["x"], mode="llm", options={}, top_k=4, profile=None, model="m")
            started = time.perf_counter()
            bench.store.get(ctx.exact_key)
            exact.append((time.perf_counter() - started) * 1000)
            vec = [math.cos(i * (j + 1)) for j in range(1024)]
            started = time.perf_counter()
            bench.store.vector_search(vec, ctx.scope, k=3)
            semantic.append((time.perf_counter() - started) * 1000)
        sizes[n] = {"exact_p50_ms": pct(exact, .5), "exact_p95_ms": pct(exact, .95), "knn_p50_ms": pct(semantic, .5), "knn_p95_ms": pct(semantic, .95)}
        print(f"  {n} entries: {sizes[n]}", flush=True)
    embed_times = []
    for i in range(20):
        started = time.perf_counter()
        bench.service._embedder._inner.embed(f"How many days per week may staff work remotely? {i}")
        embed_times.append((time.perf_counter() - started) * 1000)
    cache.purge_tenant(tenant)
    return {"by_entries": sizes, "query_embedding_p50_ms": pct(embed_times, .5),
            "note": "Lookups from this process to Redis on the same machine. A semantic lookup also needs the question's embedding."}


SECTIONS: Dict[str, Callable[[Bench], Dict[str, Any]]] = {
    "answers": run_answers, "embeddings": run_embeddings, "prefix": run_prefix, "cache_safety": run_cache_safety,
    "semantic": run_semantic, "routing_tasks": run_routing_tasks, "cascade": run_cascade, "tokens": run_tokens,
    "llm_cache": run_llm_cache, "workload": run_workload, "live": run_live, "overhead": run_overhead,
}


def cleanup(args: argparse.Namespace) -> None:
    from elasticsearch import Elasticsearch

    import redis

    from src.config import ELASTICSEARCH_URL

    client = Elasticsearch(ELASTICSEARCH_URL)
    names = [n for n in client.indices.get(index=f"{PREFIX}*").keys() if n.startswith(PREFIX)]
    for name in names:
        client.indices.delete(index=name)
    r = redis.Redis.from_url(args.redis)
    keys = list(r.scan_iter(match=f"{PREFIX}:*", count=1000))
    for start in range(0, len(keys), 500):
        r.delete(*keys[start:start + 500])
    try:
        r.ft(f"{PREFIX}:semantic-index").dropindex(delete_documents=True)
    except Exception:
        pass
    shutil.rmtree(args.workspace, ignore_errors=True)
    print(f"Deleted {len(names)} indexes, {len(keys)} Redis keys and {args.workspace}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sections", default="answers,embeddings,prefix,cache_safety,semantic,routing_tasks,cascade,tokens,llm_cache,workload,live,overhead")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/beyondrag-ep7-bench-workspace"))
    parser.add_argument("--redis", default="redis://localhost:6379/0")
    parser.add_argument("--otlp", default="")
    parser.add_argument("--small", default="qwen3:1.7b")
    parser.add_argument("--medium", default="qwen3:4b")
    parser.add_argument("--large", default="qwen3:8b")
    parser.add_argument("--lab-docs", type=int, default=67)
    parser.add_argument("--qqp", type=int, default=1000)
    parser.add_argument("--token-cases", type=int, default=12)
    parser.add_argument("--llm-cache-cases", type=int, default=6)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--zipf", type=float, nargs="+", default=[0.6, 1.0, 1.4])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--live-requests", type=int, default=60)
    parser.add_argument("--live-threshold", type=float, default=0.90)
    parser.add_argument("--reseed", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    configure_environment(args)
    if args.cleanup:
        cleanup(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    bench = Bench(args)
    for name in [s.strip() for s in args.sections.split(",") if s.strip()]:
        print(f"== {name}", flush=True)
        started = time.perf_counter()
        result = SECTIONS[name](bench)
        result["section_seconds"] = round(time.perf_counter() - started, 1)
        result["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
        print(f"   wrote {args.out / (name + '.json')} in {result['section_seconds']}s", flush=True)
    from opentelemetry import metrics, trace

    for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
        if hasattr(provider, "force_flush"):
            provider.force_flush()


if __name__ == "__main__":
    main()
