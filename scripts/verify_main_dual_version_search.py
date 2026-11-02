#!/usr/bin/env python3
"""Run an actual HybridRetriever search against **main** (Ep1) code.

Ingests policy_v1 + policy_v2 PDFs with main's PdfIngestor (no soft supersede),
then runs HybridRetriever.search for a cash-buffer query. Prints JSON with
whether authorized hits contain both 450,000,000 and 520,000,000.

Usage (from Ep2 branch tree, fixtures must exist):
  python3 scripts/verify_main_dual_version_search.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "pdfs"
V1 = FIXTURES / "policy_v1_liquidity_table.pdf"
V2 = FIXTURES / "policy_v2_liquidity_table.pdf"


def main() -> int:
    if not V1.is_file() or not V2.is_file():
        print(json.dumps({"ok": False, "error": "fixtures missing"}))
        return 2

    with tempfile.TemporaryDirectory(prefix="beyondrag-main-wt-") as td:
        wt = Path(td) / "main"
        # Detached worktree of origin/main (or local main).
        ref = "main"
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), ref],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(json.dumps({"ok": False, "error": "worktree failed", "stderr": r.stderr}))
            return 3

        runner = wt / "_ep1_dual_search_runner.py"
        runner.write_text(
            f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(wt)!r})

from src.rag.chunker import create_chunker
from src.rag.cross_lingual import MultilingualQueryAdapter
from src.rag.embeddings import DeterministicFakeEmbeddingProvider
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import DocumentParserRegistry
from src.rag.ingestor import PdfIngestor
from src.rag.repository import DocumentRepository
from src.rag.retriever import HybridRetriever

v1 = Path({str(V1)!r}).read_bytes()
v2 = Path({str(V2)!r}).read_bytes()
es = ElasticsearchIndexManager(es_url="http://non-existent-es:9200")
repo = DocumentRepository(es_manager=es)
embedder = DeterministicFakeEmbeddingProvider(dimensions=32)
ingestor = PdfIngestor(
    extractor=DocumentParserRegistry(),
    chunker=create_chunker("sliding_window", max_tokens=120),
    cross_lingual=MultilingualQueryAdapter(),
    embedder=embedder,
    repository=repo,
)
m1 = ingestor.ingest("policy_v1_liquidity_table.pdf", v1, tags=["finance"])
m2 = ingestor.ingest("policy_v2_liquidity_table.pdf", v2, tags=["finance"])
retriever = HybridRetriever(es_manager=es)
q = "intraday unencumbered HKD cash buffer threshold"
outcome = retriever.search(
    expanded_query=q,
    query_vector=embedder.embed(q),
    effective_acl_tags=["__public__", "finance"],
    top_k=8,
)
hit_blob = "\\n".join(c.get("content", "") for c in outcome.authorized_chunks)
# Also inspect full index (Ep1 had no is_current filter) for honesty about storage.
all_blob = "\\n".join(c.get("content", "") for c in es.memory_chunks)
import json
print(json.dumps({{
    "ok": True,
    "branch_ref": "main",
    "doc_ids": [m1["doc_id"], m2["doc_id"]],
    "statuses": [m1.get("status"), m2.get("status")],
    "hit_count": len(outcome.authorized_chunks),
    "search_has_450": "450,000,000" in hit_blob,
    "search_has_520": "520,000,000" in hit_blob,
    "index_has_450": "450,000,000" in all_blob,
    "index_has_520": "520,000,000" in all_blob,
    "query": q,
}}))
""",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(wt)
        proc = subprocess.run(
            [sys.executable, str(runner)],
            cwd=str(wt),
            capture_output=True,
            text=True,
            env=env,
        )
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=str(ROOT), capture_output=True)
        if proc.returncode != 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "runner failed",
                        "stdout": proc.stdout[-2000:],
                        "stderr": proc.stderr[-2000:],
                        "code": proc.returncode,
                    }
                )
            )
            return 4
        # runner prints one JSON line
        line = proc.stdout.strip().splitlines()[-1]
        print(line)
        payload = json.loads(line)
        if payload.get("search_has_450") and payload.get("search_has_520"):
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
