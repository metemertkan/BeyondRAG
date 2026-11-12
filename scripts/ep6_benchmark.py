#!/usr/bin/env python3
"""Episode 6 benchmark: observability and guardrails, measured.

Sections (``--sections``, comma-separated):

  failures           The same questions with Ollama healthy, down, hung, and with a model name
                     that is not installed. HTTP status next to the outcome label, per answer.
                     Plus one ingestion with the model down.
  tokens             Tokens per question and per pipeline stage for five configurations.
  guardrails         Injection and advice rules, and a model judge, on a labelled set and on
                     the public deepset/prompt-injections test split (downloaded, not committed).
  pii                Detection per kind of personal data, and false positives on business text.
  context_injection  Documents with instructions for the assistant planted in the index:
                     how often the answer follows them, with the context filter off and on.
  budget             A daily token budget per tenant: when it stops the model, and by how much
                     the last request overshoots.
  overhead           What telemetry and the guardrail checks cost per question.

Needs Elasticsearch and Ollama (bge-m3 plus ``--llm``). Everything goes into indexes named
``beyondrag-ep6-bench*`` and a workspace folder of its own; ``--cleanup`` deletes both.
With ``--otlp`` set, telemetry also goes to that collector, so the Grafana dashboard fills.

    uv run scripts/ep6_benchmark.py --out /tmp/ep6 --otlp http://localhost:4318 --sections pii,guardrails
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep6_fixtures as fq  # noqa: E402

INDEX_PREFIX = "beyondrag-ep6-bench"
DEAD_PORT = 11435
HUNG_PORT = 11436


def configure_environment(args: argparse.Namespace) -> None:
    """Before any src import: the config module reads these once."""
    os.environ.update(
        {
            "RAG_INDEX_PREFIX": INDEX_PREFIX,
            "RAG_WORKSPACE_DIR": str(args.workspace),
            "RAG_EMBEDDING_PROVIDER": "ollama",
            "RAG_EMBEDDING_MODEL": "bge-m3",
            "RAG_EMBEDDING_DIMS": "1024",
            "OLLAMA_MODEL": args.llm,
            "USE_LOCAL_ONLY": "true",
            "RAG_AUTH_MODE": "demo",
            "RAG_AUTH_SECRET": "ep6-benchmark-signing-secret-0123456789abcdef",
            "RAG_GUARDRAILS": "off",
            "RAG_QUERY_MODE": "off",
            "RAG_RERANK": "false",
            "RAG_MULTI_HOP": "false",
            "RAG_VERIFY": "false",
            "OTEL_SERVICE_NAME": os.environ.get("OTEL_SERVICE_NAME", "beyondrag-bench"),
            "OTEL_METRIC_EXPORT_INTERVAL": "5000",
        }
    )
    if args.otlp:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = args.otlp
    else:
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    args.workspace.mkdir(parents=True, exist_ok=True)


# --- Shared helpers -------------------------------------------------------------------


def pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 2)


def ratio(part: float, whole: float) -> Optional[float]:
    return round(part / whole, 4) if whole else None


class Bench:
    """The service, the HTTP app and an in-memory copy of every span, for one run."""

    def __init__(self, args: argparse.Namespace) -> None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from src.rag import telemetry

        self.args = args
        self.spans = InMemorySpanExporter()
        telemetry.setup_telemetry(span_processors=[SimpleSpanProcessor(self.spans)])

        import src.rag.service as service_module
        from src.rag.service import RBACKnowledgeBaseService

        self.service = RBACKnowledgeBaseService()
        # The app's singleton, so the HTTP routes use this service and do not reseed.
        service_module._KB_SERVICE_INSTANCE = self.service
        self.seeding: Dict[str, Any] = {}
        _seed(self)

        from fastapi.testclient import TestClient

        from src.web.app import app

        self.client = TestClient(app)
        self._tokens: Dict[str, str] = {}

    def token(self, tenant: str, roles: Sequence[str]) -> str:
        key = f"{tenant}:{','.join(roles)}"
        if key not in self._tokens:
            response = self.client.post("/api/auth/demo-token", json={"tenant": tenant, "roles": list(roles)})
            response.raise_for_status()
            self._tokens[key] = response.json()["token"]
        return self._tokens[key]

    def model_spans(self, stage: Optional[str] = None) -> List[Any]:
        return [
            span
            for span in self.spans.get_finished_spans()
            if span.attributes.get("gen_ai.operation.name") and (stage is None or span.attributes.get("rag.stage") == stage)
        ]


def _seed(bench: "Bench") -> None:
    """Load the demo documents once. Concept extraction at ingestion calls the model."""
    from src.rag.seeding import DEMO_TENANT_SAMPLES, create_pdf_bytes

    service = bench.service
    service._index_manager.ensure_index()
    if service.list_documents(tenant="acme") and not bench.args.reseed:
        bench.seeding = {"reused": True}
        return
    bench.spans.clear()
    started = time.perf_counter()
    # By hand rather than through the seeder, so the time and the model calls are this step's alone.
    service._index_manager.reset_index()
    service._repository.forget_all_documents()
    for tenant, samples in DEMO_TENANT_SAMPLES.items():
        for sample in samples:
            service.ingest_pdf_bytes(
                filename=sample["filename"], pdf_bytes=create_pdf_bytes(sample["filename"], sample["pages"]),
                tags=sample["tags"], tenant=tenant,
            )
    calls = bench.model_spans()
    bench.seeding = {
        "reused": False,
        "documents": sum(len(samples) for samples in DEMO_TENANT_SAMPLES.values()),
        "seconds": round(time.perf_counter() - started, 1),
        "model_calls": {
            op: sum(1 for s in calls if s.attributes.get("gen_ai.operation.name") == op) for op in ("chat", "embeddings")
        },
        "input_tokens": {
            op: sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in calls if s.attributes.get("gen_ai.operation.name") == op)
            for op in ("chat", "embeddings")
        },
        "output_tokens_chat": sum(int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in calls if s.attributes.get("gen_ai.operation.name") == "chat"),
        "concept_extraction_failed": sum(
            1 for s in bench.spans.get_finished_spans() for e in s.events if e.name == "concept_extraction_failed"
        ),
    }


def lab_tenant(bench: Bench, tenant: str, extra: Sequence[Dict[str, Any]] = ()) -> None:
    """The Acme documents in a tenant of their own, without concept extraction."""
    from src.rag.seeding import DEMO_TENANT_SAMPLES, create_pdf_bytes

    service = bench.service
    existing = {d["filename"] for d in service.list_documents(tenant=tenant)}
    for sample in [*DEMO_TENANT_SAMPLES["acme"], *extra]:
        if sample["filename"] in existing:
            continue
        service.ingest_pdf_bytes(
            filename=sample["filename"], pdf_bytes=create_pdf_bytes(sample["filename"], sample["pages"]),
            tags=sample["tags"], tenant=tenant, extract_concepts=False,
        )


def summarize_answer(body: Dict[str, Any], expected: str = "") -> Dict[str, Any]:
    info = body.get("telemetry") or {}
    return {
        "outcome": info.get("outcome"),
        "outcome_group": info.get("outcome_group"),
        "detail": info.get("detail"),
        "duration_ms": info.get("duration_ms"),
        "input_tokens": (info.get("tokens") or {}).get("input"),
        "output_tokens": (info.get("tokens") or {}).get("output"),
        "by_stage": (info.get("tokens") or {}).get("by_stage"),
        "models": sorted({f"{c['operation']} {c['model']}" for c in info.get("llm_calls", [])}),
        "llm_errors": [c["error"] for c in info.get("llm_calls", []) if c.get("error")],
        "events": [e["event"] for e in info.get("events", [])],
        "trace_id": info.get("trace_id"),
        "contains_expected": (expected.lower() in str(body.get("answer", "")).lower()) if expected else None,
        "answer": str(body.get("answer", ""))[:400],
    }


# --- failures ---------------------------------------------------------------------------


class HungServer:
    """Accepts connections and reads, and never answers. A model server stuck on a load."""

    def __init__(self, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(16)
        self.connections: List[socket.socket] = []
        self.running = True
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while self.running:
            try:
                connection, _ = self.sock.accept()
            except OSError:
                return
            self.connections.append(connection)

    def stop(self) -> None:
        self.running = False
        for connection in self.connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
                connection.close()
            except OSError:
                pass
        self.sock.close()


def run_failures(bench: Bench) -> Dict[str, Any]:
    import src.config as config
    import src.rag.cross_lingual as cross_lingual

    real_url = config.OLLAMA_BASE_URL
    token = bench.token("acme", fq.ROLES)
    questions = fq.QUESTIONS[: bench.args.questions]

    def ask_all(arm: str) -> List[Dict[str, Any]]:
        rows = []
        for question, expected in questions:
            started = time.perf_counter()
            response = bench.client.post(
                "/api/chat", headers={"Authorization": f"Bearer {token}"},
                json={"question": question, "synthesis_mode": "llm"},
            )
            row = {"arm": arm, "question": question, "http_status": response.status_code,
                   "wall_seconds": round(time.perf_counter() - started, 2)}
            if response.status_code == 200:
                row.update(summarize_answer(response.json(), expected))
            rows.append(row)
            print(f"  {arm:<22} {row['http_status']} {row.get('outcome')!s:<24} {row['wall_seconds']:>6}s {row.get('models')}", flush=True)
        return rows

    results: List[Dict[str, Any]] = []
    results += ask_all("healthy")

    config.OLLAMA_BASE_URL = f"http://127.0.0.1:{DEAD_PORT}"
    try:
        results += ask_all("ollama_down")
    finally:
        config.OLLAMA_BASE_URL = real_url

    # A model name that is not installed. get_llm picks an installed model instead.
    os.environ["OLLAMA_MODEL"] = bench.args.missing_model
    try:
        results += ask_all("model_not_installed")
    finally:
        os.environ["OLLAMA_MODEL"] = bench.args.llm

    hung = HungServer(HUNG_PORT)
    config.OLLAMA_BASE_URL = f"http://127.0.0.1:{HUNG_PORT}"
    os.environ["RAG_LLM_TIMEOUT_SECONDS"] = str(bench.args.hung_timeout)
    try:
        results += ask_all(f"hung_timeout_{bench.args.hung_timeout:g}s")[:2]
        # No timeout: does the request ever come back? Waited for up to --hung-cap seconds.
        os.environ["RAG_LLM_TIMEOUT_SECONDS"] = "0"
        box: Dict[str, Any] = {}
        question, expected = questions[0]

        def waiting() -> None:
            started = time.perf_counter()
            try:
                body = bench.service.query_and_chat(question, user_roles=fq.ROLES, synthesis_mode="llm", tenant="acme")
                box.update(summarize_answer(body, expected))
            except Exception as exc:  # noqa: BLE001
                box["exception"] = f"{type(exc).__name__}: {exc}"
            box["returned_after_seconds"] = round(time.perf_counter() - started, 1)

        worker = threading.Thread(target=waiting, daemon=True)
        worker.start()
        worker.join(bench.args.hung_cap)
        still_waiting = worker.is_alive()
        hung.stop()  # Lets the stuck call fail, so the thread ends.
        worker.join(30)
        results.append({"arm": "hung_no_timeout", "question": question, "cap_seconds": bench.args.hung_cap,
                        "still_waiting_at_cap": still_waiting, **box})
        print(f"  hung_no_timeout        still waiting after {bench.args.hung_cap}s: {still_waiting}", flush=True)
    finally:
        hung.stop()
        config.OLLAMA_BASE_URL = real_url
        os.environ["RAG_LLM_TIMEOUT_SECONDS"] = "180"

    # Ingestion with the model down. Before this episode the error was swallowed in silence.
    bench.spans.clear()
    cross_lingual.OLLAMA_BASE_URL = f"http://127.0.0.1:{DEAD_PORT}"
    try:
        from src.rag.seeding import create_pdf_bytes

        receipt = bench.service.ingest_pdf_bytes(
            filename=f"fault_{uuid.uuid4().hex[:6]}.pdf",
            pdf_bytes=create_pdf_bytes("fault", [[f"Fault drill {uuid.uuid4()}: dock 7 closes at 18:00 on public holidays."]]),
            tags=[], tenant="faultlab",
        )
    finally:
        cross_lingual.OLLAMA_BASE_URL = real_url
    events = [e.name for s in bench.spans.get_finished_spans() for e in s.events]
    ingestion = {
        "status": receipt.get("status") or receipt.get("ingestion_status"),
        "concepts": len(receipt.get("concepts") or receipt.get("extracted_concepts") or []),
        "concept_extraction_failed_events": events.count("concept_extraction_failed"),
        "receipt_keys": sorted(receipt)[:30],
    }

    arms: Dict[str, Dict[str, Any]] = {}
    for row in results:
        if "http_status" not in row:
            continue
        arm = arms.setdefault(row["arm"], {"requests": 0, "http_200": 0, "outcomes": {}, "wall_seconds": [], "correct": 0, "models": set()})
        arm["requests"] += 1
        arm["http_200"] += row["http_status"] == 200
        arm["outcomes"][row.get("outcome")] = arm["outcomes"].get(row.get("outcome"), 0) + 1
        arm["wall_seconds"].append(row["wall_seconds"])
        arm["correct"] += bool(row.get("contains_expected"))
        arm["models"].update(row.get("models") or [])
    for arm in arms.values():
        arm["p50_seconds"] = pct(arm["wall_seconds"], 0.5)
        arm["max_seconds"] = max(arm["wall_seconds"])
        arm["models"] = sorted(arm["models"])
        del arm["wall_seconds"]
    return {"llm": bench.args.llm, "missing_model": bench.args.missing_model, "arms": arms,
            "ingestion_with_model_down": ingestion, "rows": results}


# --- tokens -----------------------------------------------------------------------------


def run_tokens(bench: Bench) -> Dict[str, Any]:
    from src.rag.pipeline import PipelineOptions

    configs: Dict[str, tuple[str, PipelineOptions]] = {
        "fast": ("fast", PipelineOptions()),
        "llm": ("llm", PipelineOptions()),
        "llm_verify": ("llm", PipelineOptions(verify=True)),
        "llm_query_rewrite": ("llm", PipelineOptions(query_mode="multi")),
        "llm_multi_hop": ("llm", PipelineOptions(multi_hop=True)),
    }
    rows = []
    for name, (mode, options) in configs.items():
        for question, expected in fq.QUESTIONS[: bench.args.questions]:
            body = bench.service.query_and_chat(question, user_roles=fq.ROLES, synthesis_mode=mode, pipeline_options=options, tenant="acme")
            row = {"config": name, "question": question, **summarize_answer(body, expected)}
            rows.append(row)
            print(f"  {name:<18} {row['outcome']!s:<24} in={row['input_tokens']:>5} out={row['output_tokens']:>4} {row['duration_ms'] / 1000:>6.1f}s", flush=True)
    summary = {}
    for name in configs:
        mine = [r for r in rows if r["config"] == name]
        stages: Dict[str, Dict[str, float]] = {}
        for r in mine:
            for stage, values in (r["by_stage"] or {}).items():
                bucket = stages.setdefault(stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                for key in bucket:
                    bucket[key] += values[key]
        n = len(mine)
        summary[name] = {
            "questions": n,
            "mean_input_tokens": round(sum(r["input_tokens"] for r in mine) / n, 1),
            "mean_output_tokens": round(sum(r["output_tokens"] for r in mine) / n, 1),
            "mean_model_calls": round(sum(v["calls"] for v in stages.values()) / n, 2),
            "mean_seconds": round(sum(r["duration_ms"] for r in mine) / n / 1000, 2),
            "by_stage_mean": {s: {k: round(v / n, 1) for k, v in b.items()} for s, b in stages.items()},
            "outcomes": {o: sum(1 for r in mine if r["outcome"] == o) for o in {r["outcome"] for r in mine}},
            "contains_expected": sum(1 for r in mine if r["contains_expected"]),
        }
    return {"llm": bench.args.llm, "embedding_model": "bge-m3", "summary": summary, "rows": rows}


# --- guardrails -------------------------------------------------------------------------


def fetch_deepset(cache: Path) -> List[Dict[str, Any]]:
    """The public test split, downloaded once into the benchmark workspace. Not committed."""
    import httpx

    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        for attempt in range(5):
            response = httpx.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": "deepset/prompt-injections", "config": "default", "split": "test", "offset": offset, "length": 100},
                timeout=60,
            )
            if response.status_code < 500:
                break
            time.sleep(2 ** attempt)
        response.raise_for_status()
        page = [item["row"] for item in response.json()["rows"]]
        rows += page
        if len(page) < 100:
            cache.write_text(json.dumps(rows), encoding="utf-8")
            return rows
        offset += 100


_GERMAN = {"und", "der", "die", "das", "ich", "nicht", "ist", "wie", "sie", "mit", "für", "auf", "ein", "eine", "zu", "von", "den", "dem", "wir", "bitte", "vergiss", "ignoriere"}


def language_guess(text: str) -> str:
    words = [w.strip(".,!?:;\"'()").lower() for w in text.split()]
    return "de" if sum(w in _GERMAN for w in words) >= 2 else "en"


def judge_factory(model: str) -> Callable[..., Any]:
    from langchain_ollama import ChatOllama

    import src.config as config
    from src.rag.telemetry import genai_handler

    def factory(**_: Any) -> Any:
        return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=0.0, reasoning=False,
                          callbacks=[genai_handler()], client_kwargs={"timeout": 120})

    return factory


def classify(engine: Any, text: str) -> Optional[str]:
    verdict = engine.check_input(text)
    return verdict.guardrail if verdict.blocked else None


def score(rows: List[Dict[str, Any]], decide: Callable[[Dict[str, Any]], bool], positive: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
    tp = sum(1 for r in rows if positive(r) and decide(r))
    fn = sum(1 for r in rows if positive(r) and not decide(r))
    fp = sum(1 for r in rows if not positive(r) and decide(r))
    tn = sum(1 for r in rows if not positive(r) and not decide(r))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
            "false_positive_rate": ratio(fp, fp + tn)}


def run_guardrails(bench: Bench) -> Dict[str, Any]:
    from src.rag.guardrails import GuardrailEngine, GuardrailPolicy

    rules = GuardrailEngine(GuardrailPolicy(enabled=True, pii_question="off"))
    judges = [m for m in bench.args.judges.split(",") if m]
    datasets = {
        "labelled": [{"text": t, "label": label} for t, label in fq.GUARDRAIL_SET],
        "deepset_test": [{"text": r["text"], "label": "attack" if r["label"] == 1 else "benign", "lang": language_guess(r["text"])} for r in fetch_deepset(bench.args.workspace / "deepset_test.json")],
    }
    for name, rows in datasets.items():
        for row in rows:
            started = time.perf_counter()
            row["rules"] = classify(rules, row["text"])
            row["rules_us"] = round((time.perf_counter() - started) * 1e6, 1)
        for model in judges:
            engine = GuardrailEngine(GuardrailPolicy(enabled=True, judge="llm", pii_question="off"), llm_factory=judge_factory(model))
            for index, row in enumerate(rows):
                bench.spans.clear()
                started = time.perf_counter()
                row[f"judge:{model}"] = engine.judge(row["text"])
                row[f"judge:{model}:seconds"] = round(time.perf_counter() - started, 2)
                calls = bench.model_spans("guardrail_judge")
                row[f"judge:{model}:tokens"] = sum(
                    int(s.attributes.get("gen_ai.usage.input_tokens", 0)) + int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in calls
                )
                if index % 20 == 0:
                    print(f"  {name} judge {model}: {index}/{len(rows)}", flush=True)

    def is_positive(r: Dict[str, Any]) -> bool:
        return r["label"] != "benign"

    report: Dict[str, Any] = {"judges": judges, "datasets": {}}
    for name, rows in datasets.items():
        arms = {"rules": score(rows, lambda r: r["rules"] is not None, is_positive)}
        for model in judges:
            arms[f"judge {model}"] = score(rows, lambda r, m=model: r[f"judge:{m}"] in {"prompt_injection", "investment_advice"}, is_positive)
            arms[f"rules + judge {model}"] = score(
                rows, lambda r, m=model: r["rules"] is not None or r[f"judge:{m}"] in {"prompt_injection", "investment_advice"}, is_positive
            )
            arms[f"judge {model}"]["errors"] = sum(1 for r in rows if r[f"judge:{model}"] == "error")
            arms[f"judge {model}"]["p50_seconds"] = pct([r[f"judge:{model}:seconds"] for r in rows], 0.5)
            arms[f"judge {model}"]["p95_seconds"] = pct([r[f"judge:{model}:seconds"] for r in rows], 0.95)
            arms[f"judge {model}"]["mean_tokens"] = round(statistics.mean(r[f"judge:{model}:tokens"] for r in rows), 1)
        entry: Dict[str, Any] = {"rows": len(rows), "positives": sum(map(is_positive, rows)), "arms": arms,
                                 "rules_p50_us": pct([r["rules_us"] for r in rows], 0.5)}
        if name == "labelled":
            entry["by_label"] = {}
            for label in ("attack", "advice", "benign"):
                mine = [r for r in rows if r["label"] == label]
                entry["by_label"][label] = {"rows": len(mine), "rules_blocked": sum(1 for r in mine if r["rules"])}
                for model in judges:
                    entry["by_label"][label][f"judge {model} flagged"] = sum(
                        1 for r in mine if r[f"judge:{model}"] in {"prompt_injection", "investment_advice"}
                    )
        else:
            entry["by_language"] = {}
            for lang in ("en", "de"):
                mine = [r for r in rows if r["lang"] == lang]
                entry["by_language"][lang] = {"rows": len(mine), "rules": score(mine, lambda r: r["rules"] is not None, is_positive)}
                for model in judges:
                    entry["by_language"][lang][f"judge {model}"] = score(
                        mine, lambda r, m=model: r[f"judge:{m}"] in {"prompt_injection", "investment_advice"}, is_positive
                    )
        report["datasets"][name] = entry
    # The labelled rows are ours and are kept. The public rows are not redistributed.
    report["labelled_rows"] = datasets["labelled"]
    report["deepset_misses_by_rules"] = sum(1 for r in datasets["deepset_test"] if r["label"] == "attack" and not r["rules"])
    return report


# --- pii --------------------------------------------------------------------------------


def run_pii(bench: Bench) -> Dict[str, Any]:
    from src.rag.pii import PII_KINDS, find_pii

    kinds: Dict[str, Dict[str, int]] = {}
    rows = []
    for text, expected in fq.PII_SET:
        found = sorted({m.kind for m in find_pii(text)})
        rows.append({"text": text, "expected": expected, "found": found})
        for kind in expected:
            bucket = kinds.setdefault(kind, {"cases": 0, "detected": 0})
            bucket["cases"] += 1
            bucket["detected"] += kind in found
    false_positives = []
    for text in fq.PII_BENIGN:
        found = [(m.kind, m.text) for m in find_pii(text)]
        if found:
            false_positives.append({"text": text, "found": found})
    corpus = " ".join(text for text, _ in fq.PII_SET) * 20
    started = time.perf_counter()
    for _ in range(50):
        find_pii(corpus)
    per_kb_us = (time.perf_counter() - started) / 50 / (len(corpus) / 1000) * 1e6
    detectable = [k for k in kinds if k in PII_KINDS]
    return {
        "by_kind": kinds,
        "detectable_kinds_recall": ratio(sum(kinds[k]["detected"] for k in detectable), sum(kinds[k]["cases"] for k in detectable)),
        "undetectable_kinds": {k: v for k, v in kinds.items() if k not in PII_KINDS},
        "benign_texts": len(fq.PII_BENIGN),
        "benign_false_positives": false_positives,
        "microseconds_per_kb": round(per_kb_us, 1),
        "rows": rows,
    }


# --- context injection ------------------------------------------------------------------


def run_context_injection(bench: Bench) -> Dict[str, Any]:
    from src.rag.guardrails import CONTEXT_RULES, GuardrailEngine, GuardrailPolicy, match_rules
    from src.rag.seeding import DEMO_TENANT_SAMPLES

    lab_tenant(bench, "injectlab", fq.PLANTED_DOCUMENTS)
    off = GuardrailEngine(GuardrailPolicy())
    on = GuardrailEngine(GuardrailPolicy(enabled=True))
    original = bench.service._guardrails
    rows = []
    try:
        for planted in fq.PLANTED_DOCUMENTS:
            for arm, engine in (("filter_off", off), ("filter_on", on)):
                bench.service._guardrails = engine
                for rep in range(bench.args.reps):
                    body = bench.service.query_and_chat(planted["question"], user_roles=fq.ROLES, synthesis_mode="llm", tenant="injectlab")
                    answer = str(body["answer"])
                    followed = any(marker.lower() in answer.lower() for marker in planted["planted"])
                    retrieved = any(c.get("filename") == planted["filename"] for c in body["authorized_chunks"])
                    dropped = [d["filename"] for d in body["guardrails"]["dropped_chunks"]]
                    rows.append({
                        "document": planted["filename"], "has_trigger_words": planted["has_trigger_words"], "arm": arm, "rep": rep,
                        "planted_in_context": retrieved or planted["filename"] in dropped,
                        "dropped": planted["filename"] in dropped, "followed_instruction": followed,
                        "outcome": body["telemetry"]["outcome"], "answer": answer[:400],
                    })
                    print(f"  {planted['filename']:<34} {arm:<10} followed={followed} dropped={planted['filename'] in dropped}", flush=True)
    finally:
        bench.service._guardrails = original

    # False drops: how much ordinary text the context rules would remove.
    texts: List[str] = []
    for samples in DEMO_TENANT_SAMPLES.values():
        for sample in samples:
            texts += [" ".join(page) for page in sample["pages"]]
    lab_path = ROOT / "config" / "ep3_demo_docs.json"
    if lab_path.exists():
        for document in json.loads(lab_path.read_text(encoding="utf-8")):
            texts.append(str(document.get("text") or ""))
    flagged = [{"rule": match_rules(t, CONTEXT_RULES), "text": t[:200]} for t in texts if match_rules(t, CONTEXT_RULES)]

    summary = {}
    for planted in fq.PLANTED_DOCUMENTS:
        for arm in ("filter_off", "filter_on"):
            mine = [r for r in rows if r["document"] == planted["filename"] and r["arm"] == arm]
            summary[f"{planted['filename']} {arm}"] = {
                "runs": len(mine), "followed": sum(r["followed_instruction"] for r in mine),
                "dropped": sum(r["dropped"] for r in mine), "in_context": sum(r["planted_in_context"] for r in mine),
            }
    return {"llm": bench.args.llm, "summary": summary, "clean_passages": len(texts), "clean_passages_flagged": flagged, "rows": rows}


# --- concepts ---------------------------------------------------------------------------


def run_concepts(bench: Bench) -> Dict[str, Any]:
    """Concept extraction on the seven demo documents, in a fresh tenant, with the current limit."""
    from src.rag.cross_lingual import concept_max_tokens
    from src.rag.seeding import DEMO_TENANT_SAMPLES, create_pdf_bytes

    tenant = f"conceptlab{uuid.uuid4().hex[:6]}"
    rows = []
    for samples in DEMO_TENANT_SAMPLES.values():
        for sample in samples:
            bench.spans.clear()
            started = time.perf_counter()
            bench.service.ingest_pdf_bytes(
                filename=sample["filename"], pdf_bytes=create_pdf_bytes(sample["filename"], sample["pages"]),
                tags=sample["tags"], tenant=tenant,
            )
            calls = bench.model_spans("concept_extraction")
            events = [e.attributes.get("rag.detail", "") for s in bench.spans.get_finished_spans() for e in s.events if e.name == "concept_extraction_failed"]
            concepts = [d for d in bench.service.list_documents(tenant=tenant) if d.get("filename") == sample["filename"]]
            row = {
                "filename": sample["filename"],
                "seconds": round(time.perf_counter() - started, 1),
                "output_tokens": sum(int(s.attributes.get("gen_ai.usage.output_tokens", 0)) for s in calls),
                "input_tokens": sum(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) for s in calls),
                "failed": bool(events),
                "failure": events[0] if events else "",
                "registered": bool(concepts),
            }
            rows.append(row)
            print(f"  {sample['filename']:<44} failed={row['failed']} out={row['output_tokens']} {row['seconds']}s {row['failure'][:80]}", flush=True)
    return {
        "llm": bench.args.llm, "max_tokens": concept_max_tokens(), "documents": len(rows),
        "failed": sum(r["failed"] for r in rows), "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 1),
        "max_output_tokens": max(r["output_tokens"] for r in rows), "total_seconds": round(sum(r["seconds"] for r in rows), 1),
        "rows": rows,
    }


# --- budget -----------------------------------------------------------------------------


def run_budget(bench: Bench) -> Dict[str, Any]:
    from src.rag.guardrails import GuardrailEngine, GuardrailPolicy, TokenLedger

    lab_tenant(bench, "budgetlab")
    ledger = TokenLedger()
    engine = GuardrailEngine(GuardrailPolicy(enabled=True, daily_token_budget=bench.args.budget), ledger=ledger)
    original = bench.service._guardrails
    bench.service._guardrails = engine
    rows = []
    try:
        for index in range(bench.args.budget_requests):
            question, expected = fq.QUESTIONS[index % len(fq.QUESTIONS)]
            before = ledger.usage("budgetlab")["total"]
            body = bench.service.query_and_chat(question, user_roles=fq.ROLES, synthesis_mode="llm", tenant="budgetlab")
            row = {"request": index + 1, "used_before": before, "used_after": ledger.usage("budgetlab")["total"],
                   **summarize_answer(body, expected)}
            rows.append(row)
            print(f"  request {index + 1}: {row['outcome']:<24} ledger {before} -> {row['used_after']}", flush=True)
    finally:
        bench.service._guardrails = original
    crossed = next((r for r in rows if r["used_after"] >= bench.args.budget), None)
    return {
        "llm": bench.args.llm, "budget": bench.args.budget, "rows": rows,
        "first_degraded_request": next((r["request"] for r in rows if r["outcome"] == "budget_exceeded"), None),
        "overshoot_tokens": (crossed["used_after"] - bench.args.budget) if crossed else None,
        "tokens_after_budget_reached": sum(
            r["used_after"] - r["used_before"] for r in rows if r["used_before"] >= bench.args.budget
        ),
    }


# --- overhead ---------------------------------------------------------------------------


def overhead_arm(arm: str, iterations: int) -> Dict[str, Any]:
    """Runs in a fresh process: the telemetry setup can only happen once per process."""
    if arm == "disabled":
        os.environ["OTEL_SDK_DISABLED"] = "true"
    if arm != "otlp":
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    os.environ["OTEL_SERVICE_NAME"] = "beyondrag-overhead"
    from src.rag import telemetry
    from src.rag.guardrails import GuardrailEngine, GuardrailPolicy

    telemetry.setup_telemetry()

    def one_request() -> None:
        with telemetry.request_scope("overhead", "llm") as record:
            for name in ("guardrail_input", "query_understanding", "search", "generation", "guardrail_output"):
                with telemetry.stage(name):
                    if name == "generation":
                        with telemetry.llm_call("qwen3:8b", "ollama") as call:
                            call.set_usage(900, 120)
                    if name == "search":
                        with telemetry.llm_call("bge-m3", "ollama", operation="embeddings") as call:
                            call.set_usage(12, None)
            telemetry.note("llm_fallback", "synthetic")
            record.outcome = "degraded_llm_fallback"

    for _ in range(50):
        one_request()
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        one_request()
        times.append((time.perf_counter() - started) * 1e6)
    result: Dict[str, Any] = {"arm": arm, "iterations": iterations, "p50_us": pct(times, 0.5), "p95_us": pct(times, 0.95), "mean_us": round(statistics.mean(times), 1)}

    if arm == "sdk":
        engine = GuardrailEngine(GuardrailPolicy(enabled=True))
        question = "What minimum cash reserve must Group Treasury keep under the Q4 liquidity policy?"
        chunk = {"chunk_id": "c", "filename": "f.pdf", "content": "The Group Treasury division must maintain an intraday minimum cash reserve. " * 14}
        answer = "- The Group Treasury must keep a minimum unencumbered cash reserve of HKD 450,000,000 [HK_Treasury_FX_Liquidity_Policy_Q4.pdf, p. 1]. " * 3
        for label, action in (
            ("check_input_us", lambda: engine.check_input(question)),
            ("filter_context_4_chunks_us", lambda: engine.filter_context([chunk] * 4)),
            ("check_output_us", lambda: engine.check_output(answer)),
        ):
            samples = []
            for _ in range(2000):
                started = time.perf_counter()
                action()
                samples.append((time.perf_counter() - started) * 1e6)
            result[label] = {"p50": pct(samples, 0.5), "p95": pct(samples, 0.95)}
    from opentelemetry import metrics, trace

    for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    return result


def run_overhead(bench: Bench) -> Dict[str, Any]:
    arms = []
    for arm in ("disabled", "sdk", "otlp"):
        if arm == "otlp" and not bench.args.otlp:
            continue
        env = {**os.environ}
        output = subprocess.run(
            [sys.executable, __file__, "--overhead-arm", arm, "--iterations", str(bench.args.iterations), "--out", "/dev/null"],
            env=env, capture_output=True, text=True, check=True, cwd=str(ROOT),
        ).stdout
        arms.append(json.loads(output.strip().splitlines()[-1]))
        print(f"  {arm:<9} p50 {arms[-1]['p50_us']} us  p95 {arms[-1]['p95_us']} us", flush=True)
    return {"arms": arms, "note": "A synthetic request: one root span, five stages, two model-call spans, one event, metrics and a log line. No model and no search."}


# --- main -------------------------------------------------------------------------------


SECTIONS: Dict[str, Callable[[Bench], Dict[str, Any]]] = {
    "pii": run_pii,
    "guardrails": run_guardrails,
    "failures": run_failures,
    "tokens": run_tokens,
    "context_injection": run_context_injection,
    "budget": run_budget,
    "concepts": run_concepts,
    "overhead": run_overhead,
}


def cleanup(args: argparse.Namespace) -> None:
    from elasticsearch import Elasticsearch

    from src.config import ELASTICSEARCH_URL

    client = Elasticsearch(ELASTICSEARCH_URL)
    names = [name for name in client.indices.get(index=f"{INDEX_PREFIX}*").keys() if name.startswith(INDEX_PREFIX)]
    for name in names:
        client.indices.delete(index=name)
    shutil.rmtree(args.workspace, ignore_errors=True)
    print(f"Deleted {len(names)} indexes and {args.workspace}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/beyondrag-ep6-bench-workspace"))
    parser.add_argument("--otlp", default="", help="OTLP HTTP endpoint, e.g. http://localhost:4318")
    parser.add_argument("--llm", default="qwen3:8b")
    parser.add_argument("--missing-model", default="qwen3:8b-instruct", help="failures: a model name that is not installed.")
    parser.add_argument("--judges", default="qwen2.5:1.5b,qwen3:8b")
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--reps", type=int, default=2, help="context_injection: answers per document and arm.")
    parser.add_argument("--budget", type=int, default=6000)
    parser.add_argument("--budget-requests", type=int, default=8)
    parser.add_argument("--hung-timeout", type=float, default=20.0)
    parser.add_argument("--hung-cap", type=float, default=120.0)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--reseed", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Delete the benchmark indexes and workspace, then exit.")
    parser.add_argument("--overhead-arm", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_environment(args)

    if args.overhead_arm:
        print(json.dumps(overhead_arm(args.overhead_arm, args.iterations)))
        return
    if args.cleanup:
        cleanup(args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    bench = Bench(args)
    if bench.seeding and not bench.seeding.get("reused"):
        (args.out / "seeding.json").write_text(json.dumps(bench.seeding, indent=2) + "\n")
        print("seeding:", bench.seeding, flush=True)
    for name in [s.strip() for s in args.sections.split(",") if s.strip()]:
        print(f"== {name}", flush=True)
        started = time.perf_counter()
        result = SECTIONS[name](bench)
        result["section_seconds"] = round(time.perf_counter() - started, 1)
        result["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=list) + "\n")
        print(f"   wrote {args.out / (name + '.json')} in {result['section_seconds']}s", flush=True)
    from opentelemetry import metrics, trace

    for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
        if hasattr(provider, "force_flush"):
            provider.force_flush()


if __name__ == "__main__":
    main()
