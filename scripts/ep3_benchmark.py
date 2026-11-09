#!/usr/bin/env python3
"""Measure the Episode 3 retrieval techniques. Writes one JSON file per section.

  python scripts/ep3_benchmark.py --out out --sections quantization,fusion,exact
  python scripts/ep3_benchmark.py --out out --sections mismatch,parent_child,multi_hop,attention

Needs Elasticsearch, Ollama with bge-m3, and `pip install .[rerank]`. LLM sections also need
the model named by --llm. The SciFact dataset (BEIR) is downloaded to ~/.cache/beyondrag-bench
and is not part of this repository. Indices are throwaway and deleted afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BENCH_LLM = next((sys.argv[i + 1] for i, arg in enumerate(sys.argv[:-1]) if arg == "--llm"), "qwen3:8b")
os.environ["USE_LOCAL_ONLY"] = "true"
os.environ["OLLAMA_MODEL"] = BENCH_LLM

from elasticsearch.helpers import bulk  # noqa: E402

import ep3_fixtures as fixtures  # noqa: E402
from src.config import ELASTICSEARCH_URL, OLLAMA_BASE_URL, get_llm  # noqa: E402
from src.rag.es_client import ElasticsearchIndexManager  # noqa: E402
from src.rag.fusion import reorder_for_attention  # noqa: E402
from src.rag.multi_hop import MultiHopPlanner  # noqa: E402
from src.rag.parent_child import ParentChildSplitter, expand_to_parents  # noqa: E402
from src.rag.pipeline import PipelineOptions, RetrievalPipeline  # noqa: E402
from src.rag.chunker import create_chunker  # noqa: E402
from src.rag.query_transform import LlmQueryTransformer  # noqa: E402
from src.rag.reranker import FastEmbedCrossEncoder  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402

CACHE = Path.home() / ".cache" / "beyondrag-bench"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DIMS = 1024
ACL = ["__public__"]


# --- Embedding with a disk cache ------------------------------------------------------


class CachedEmbedder:
    def __init__(self) -> None:
        self._path = CACHE / "embeddings.json"
        CACHE.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = json.loads(self._path.read_text()) if self._path.exists() else {}
        self._client = httpx.Client(timeout=600)

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._cache))

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest()[:24] for text in texts]
        missing = list({key: text for key, text in zip(keys, texts) if key not in self._cache}.items())
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            response = self._client.post(f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed", json={"model": "bge-m3", "input": [t for _, t in batch]})
            response.raise_for_status()
            for (key, _), vector in zip(batch, response.json()["embeddings"]):
                self._cache[key] = vector
        if missing:
            self._save()
        return [self._cache[key] for key in keys]

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]


# --- Corpus helpers -------------------------------------------------------------------


class TempIndex:
    def __init__(self, vector_type: str = "int8_hnsw") -> None:
        self.manager = ElasticsearchIndexManager(
            es_url=ELASTICSEARCH_URL,
            index_name=f"beyondrag-bench-{uuid.uuid4().hex[:10]}",
            dimensions=DIMS,
            vector_index_type=vector_type,
        )
        if not self.manager.is_online():
            raise SystemExit("Elasticsearch is not reachable.")
        self.manager.client = self.manager.client.options(request_timeout=300)
        # Keep segment files separate, or the vector files can't be measured on their own.
        self.manager.client.indices.put_settings(index=self.manager.index_name, settings={"index.compound_format": False})

    def __enter__(self) -> "TempIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.manager.client.indices.delete(index=self.manager.index_name, ignore_unavailable=True)

    def load(self, docs: Sequence[Dict[str, Any]], vectors: Sequence[Sequence[float]]) -> None:
        def actions() -> Any:
            for doc, vector in zip(docs, vectors):
                yield {
                    "_index": self.manager.index_name,
                    "_id": doc["id"],
                    "chunk_id": doc["id"],
                    "doc_id": doc.get("doc_id", doc["id"]),
                    "filename": doc.get("filename", doc["id"]),
                    "title": doc.get("title", ""),
                    "page_number": 1,
                    "content": doc["text"],
                    "is_public": True,
                    "acl_tags": ACL,
                    "is_current": True,
                    "ingestion_status": "INDEXED",
                    "embedding": list(vector),
                    **({"parent_id": doc["parent_id"], "parent_content": doc["parent_content"]} if doc.get("parent_id") else {}),
                }

        bulk(self.manager.client, actions(), chunk_size=200)
        self.manager.client.indices.refresh(index=self.manager.index_name)


def load_scifact() -> Dict[str, Any]:
    root = CACHE / "scifact"
    if not root.exists():
        print("downloading SciFact", flush=True)
        data = httpx.get(SCIFACT_URL, timeout=300, follow_redirects=True).content
        zipfile.ZipFile(io.BytesIO(data)).extractall(CACHE)
    docs = [json.loads(line) for line in (root / "corpus.jsonl").open(encoding="utf-8")]
    qrels: Dict[str, Dict[str, int]] = {}
    for line in list((root / "qrels" / "test.tsv").open())[1:]:
        qid, doc_id, score = line.split("\t")
        qrels.setdefault(qid, {})[doc_id] = int(score)
    queries = [json.loads(line) for line in (root / "queries.jsonl").open(encoding="utf-8")]
    queries = [q for q in queries if q["_id"] in qrels]
    texts = [(d["title"] + " " + d["text"])[:6000] for d in docs]
    return {"docs": docs, "texts": texts, "qrels": qrels, "queries": queries}


def scifact_vectors(data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    root = CACHE / "scifact"
    doc_vecs, query_vecs = root / "d_vec.npy", root / "q_vec.npy"
    if doc_vecs.exists() and query_vecs.exists():
        return np.load(doc_vecs), np.load(query_vecs)
    if (root / "d_vec.json").exists():
        docs = np.array(json.loads((root / "d_vec.json").read_text()), dtype=np.float32)
        order = json.loads((root / "test_qids.json").read_text())
        raw = np.array(json.loads((root / "q_vec.json").read_text()), dtype=np.float32)
        by_id = {qid: vector for qid, vector in zip(order, raw)}
        queries = np.array([by_id[q["_id"]] for q in data["queries"]], dtype=np.float32)
    else:
        embedder = CachedEmbedder()
        docs = np.array(embedder.embed_many(data["texts"]), dtype=np.float32)
        queries = np.array(embedder.embed_many([q["text"] for q in data["queries"]]), dtype=np.float32)
    np.save(doc_vecs, docs)
    np.save(query_vecs, queries)
    return docs, queries


# --- Metrics --------------------------------------------------------------------------


def ndcg_at(ranked: Sequence[str], relevant: Dict[str, int], k: int = 10) -> float:
    dcg = sum(relevant.get(doc, 0) / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]))
    ideal = sum(score / math.log2(rank + 2) for rank, score in enumerate(sorted(relevant.values(), reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0


def recall_at(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Dict[str, int], k: int = 10) -> float:
    return next((1.0 / (rank + 1) for rank, doc in enumerate(ranked[:k]) if doc in relevant), 0.0)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {key: round(statistics.mean(row[key] for row in rows), 4) for key in rows[0]}


def percentile(values: Sequence[float], q: float) -> float:
    return round(float(np.percentile(np.array(values), q)), 2)


# --- 1. Quantization ------------------------------------------------------------------


def run_quantization(_: argparse.Namespace) -> Dict[str, Any]:
    data = load_scifact()
    doc_vectors, query_vectors = scifact_vectors(data)
    docs = [{"id": doc["_id"], "text": text} for doc, text in zip(data["docs"], data["texts"])]
    ids = [doc["_id"] for doc in data["docs"]]
    # Exact top-10 by cosine similarity: the ground truth that ANN recall is measured against.
    normalized = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    queries = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    exact = np.argsort(-queries @ normalized.T, axis=1)[:, :10]
    results: Dict[str, Any] = {}
    for vector_type in ("hnsw", "int8_hnsw", "int4_hnsw"):
        print("quantization", vector_type, flush=True)
        with TempIndex(vector_type) as index:
            client, name = index.manager.client, index.manager.index_name
            started = time.perf_counter()
            index.load(docs, doc_vectors.tolist())
            client.indices.forcemerge(index=name, max_num_segments=1, request_timeout=900)
            client.indices.refresh(index=name)
            build_seconds = time.perf_counter() - started
            stats = client.indices.stats(index=name, metric="segments", include_segment_file_sizes=True)
            files = stats["indices"][name]["primaries"]["segments"]["file_sizes"]
            size = {ext: files.get(ext, {}).get("size_in_bytes", 0) for ext in ("vec", "veq", "vex")}
            rows, latencies, ndcgs = [], [], []
            for position, (query, vector) in enumerate(zip(data["queries"], query_vectors)):
                started = time.perf_counter()
                hits = client.search(
                    index=name, size=10, _source=False,
                    knn={"field": "embedding", "query_vector": vector.tolist(), "k": 10, "num_candidates": 100},
                )["hits"]["hits"]
                latencies.append((time.perf_counter() - started) * 1000)
                got = [hit["_id"] for hit in hits]
                truth = {ids[i] for i in exact[position]}
                rows.append(len(truth & set(got)) / 10)
                ndcgs.append(ndcg_at(got, data["qrels"][query["_id"]]))
            # Search needs the graph plus the vectors it compares: raw floats, or the quantized copy.
            searched_bytes = size["vex"] + (size["veq"] if size["veq"] else size["vec"])
            results[vector_type] = {
                "vectors": len(docs),
                "dims": DIMS,
                "raw_float_bytes": size["vec"],
                "quantized_bytes": size["veq"],
                "graph_bytes": size["vex"],
                "memory_for_search_bytes": searched_bytes,
                "total_disk_bytes": size["vec"] + size["veq"] + size["vex"],
                "memory_gb_per_million": round(searched_bytes / len(docs) * 1e6 / 1e9, 2),
                "recall_at_10_vs_exact": round(statistics.mean(rows), 4),
                "ndcg_at_10": round(statistics.mean(ndcgs), 4),
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "build_seconds": round(build_seconds, 1),
            }
    baseline = results["hnsw"]
    for result in results.values():
        result["memory_saving_vs_float"] = round(1 - result["memory_for_search_bytes"] / baseline["memory_for_search_bytes"], 4)
        result["disk_change_vs_float"] = round(result["total_disk_bytes"] / baseline["total_disk_bytes"] - 1, 4)
    exact_ndcg = statistics.mean(
        ndcg_at([ids[i] for i in row], data["qrels"][query["_id"]]) for row, query in zip(exact, data["queries"])
    )
    return {"dataset": "BEIR SciFact", "num_queries": len(data["queries"]), "exact_ndcg_at_10": round(exact_ndcg, 4), "types": results}


# --- 2. Fusion and re-ranking on SciFact ----------------------------------------------


class VectorLookup:
    def __init__(self, table: Dict[str, List[float]], fallback: CachedEmbedder) -> None:
        self._table, self._fallback = table, fallback

    def embed(self, text: str) -> List[float]:
        return self._table[text] if text in self._table else self._fallback.embed(text)


def run_fusion(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_scifact()
    doc_vectors, query_vectors = scifact_vectors(data)
    docs = [{"id": doc["_id"], "text": text} for doc, text in zip(data["docs"], data["texts"])]
    table = {q["text"]: v.tolist() for q, v in zip(data["queries"], query_vectors)}
    embedder = VectorLookup(table, CachedEmbedder())
    results: Dict[str, Any] = {}
    with TempIndex() as index:
        index.load(docs, doc_vectors.tolist())
        retriever = HybridRetriever(index.manager)

        def evaluate(label: str, search: Callable[[str], List[str]], limit: int | None = None) -> None:
            print("fusion", label, flush=True)
            rows, latencies = [], []
            for query in data["queries"][:limit]:
                started = time.perf_counter()
                ranked = search(query["text"])
                latencies.append((time.perf_counter() - started) * 1000)
                relevant = data["qrels"][query["_id"]]
                rows.append(
                    {
                        "ndcg_at_10": ndcg_at(ranked, relevant),
                        "recall_at_10": recall_at(ranked, relevant, 10),
                        "recall_at_100": recall_at(ranked, relevant, 100),
                        "mrr_at_10": reciprocal_rank(ranked, relevant),
                    }
                )
            results[label] = {**summarize(rows), "latency_p50_ms": percentile(latencies, 50), "latency_p95_ms": percentile(latencies, 95), "queries": len(rows)}

        def first_stage(mode: str, profile: str) -> Callable[[str], List[str]]:
            def search(text: str) -> List[str]:
                vector = embedder.embed(text) if mode != "lexical" else None
                outcome = retriever.search(text, vector, ACL, top_k=100, candidate_k=100, fusion=mode, profile_name=profile)
                return [chunk["chunk_id"] for chunk in outcome.authorized_chunks]

            return search

        for profile in ("default", "high_recall"):
            for mode in ("dense", "lexical", "linear", "rrf"):
                evaluate(f"{mode} ({profile})", first_stage(mode, profile))

        pipeline = RetrievalPipeline(retriever, embedder, reranker=FastEmbedCrossEncoder())

        def reranked(profile: str, candidates: int) -> Callable[[str], List[str]]:
            def search(text: str) -> List[str]:
                result = pipeline.run(text, text, ACL, top_k=candidates, profile_name=profile, options=PipelineOptions(rerank=True, candidates=candidates))
                return [chunk["chunk_id"] for chunk in result.chunks]

            return search

        evaluate("rrf + rerank top 50 (high_recall)", reranked("high_recall", 50))
    return {"dataset": "BEIR SciFact", "num_docs": len(docs), "reranker": "BAAI/bge-reranker-base", "results": results}


def run_rerank_depth(_: argparse.Namespace) -> Dict[str, Any]:
    """How much of the re-ranking gain survives a smaller, faster candidate pool?"""
    data = load_scifact()
    doc_vectors, query_vectors = scifact_vectors(data)
    docs = [{"id": doc["_id"], "text": text} for doc, text in zip(data["docs"], data["texts"])]
    embedder = VectorLookup({q["text"]: v.tolist() for q, v in zip(data["queries"], query_vectors)}, CachedEmbedder())
    results: Dict[str, Any] = {}
    with TempIndex() as index:
        index.load(docs, doc_vectors.tolist())
        pipeline = RetrievalPipeline(HybridRetriever(index.manager), embedder, reranker=FastEmbedCrossEncoder())
        for depth in (10, 20):
            print("rerank depth", depth, flush=True)
            rows, latencies = [], []
            for query in data["queries"]:
                started = time.perf_counter()
                result = pipeline.run(query["text"], query["text"], ACL, top_k=depth, profile_name="high_recall", options=PipelineOptions(rerank=True, candidates=depth))
                latencies.append((time.perf_counter() - started) * 1000)
                ranked = [chunk["chunk_id"] for chunk in result.chunks]
                relevant = data["qrels"][query["_id"]]
                rows.append({"ndcg_at_10": ndcg_at(ranked, relevant), "recall_at_10": recall_at(ranked, relevant, 10), "mrr_at_10": reciprocal_rank(ranked, relevant)})
            results[str(depth)] = {**summarize(rows), "latency_p50_ms": percentile(latencies, 50), "latency_p95_ms": percentile(latencies, 95), "queries": len(rows)}
    return {"dataset": "BEIR SciFact", "results": results}


# --- 3. Exact match -------------------------------------------------------------------


def run_exact(_: argparse.Namespace) -> Dict[str, Any]:
    return {
        "easy_short_distinct_pages": _exact_on(fixtures.exact_match_corpus()),
        "hard_long_similar_pages": _exact_on(fixtures.hard_exact_match_corpus()),
    }


def _exact_on(corpus: Dict[str, Any]) -> Dict[str, Any]:
    embedder = CachedEmbedder()
    vectors = embedder.embed_many([doc["text"] for doc in corpus["docs"]])
    with TempIndex() as index:
        index.load(corpus["docs"], vectors)
        retriever = HybridRetriever(index.manager)
        client, name = index.manager.client, index.manager.index_name

        def split_lexical(text: str) -> List[str]:
            # The analyzer most pipelines start with: "ACME-X200" becomes "acme" + "x200".
            hits = client.search(index=name, size=10, _source=False, query={"match": {"content": text}})["hits"]["hits"]
            return [hit["_id"] for hit in hits]

        def mode_search(mode: str) -> Callable[[str], List[str]]:
            def search(text: str) -> List[str]:
                vector = embedder.embed(text) if mode != "lexical" else None
                outcome = retriever.search(text, vector, ACL, top_k=10, candidate_k=10, fusion=mode)
                return [chunk["chunk_id"] for chunk in outcome.authorized_chunks]

            return search

        strategies = {
            "dense only (bge-m3)": mode_search("dense"),
            "BM25, standard analyzer": split_lexical,
            "BM25, code-preserving analyzer": mode_search("lexical"),
            "hybrid RRF, code-preserving": mode_search("rrf"),
        }
        results: Dict[str, Any] = {}
        samples: Dict[str, Any] = {}
        for label, search in strategies.items():
            print("exact", label, flush=True)
            by_kind: Dict[str, List[Dict[str, float]]] = {}
            for query in corpus["queries"]:
                ranked = search(query["query"])
                row = {
                    "hit_at_1": float(ranked[:1] == [query["target"]]),
                    "hit_at_5": float(query["target"] in ranked[:5]),
                }
                by_kind.setdefault(query["kind"], []).append(row)
                by_kind.setdefault("all", []).append(row)
            results[label] = {kind: {**summarize(rows), "queries": len(rows)} for kind, rows in by_kind.items()}
        for query_text in ("ACME-X200", "What should I do about error ERR-1042?"):
            samples[query_text] = {
                "standard_tokens": [t["token"] for t in client.indices.analyze(index=name, analyzer="multilingual", text=query_text)["tokens"]],
                "code_preserving_tokens": [t["token"] for t in client.indices.analyze(index=name, analyzer="code_preserving", text=query_text)["tokens"]],
            }
    return {"num_docs": len(corpus["docs"]), "num_queries": len(corpus["queries"]), "results": results, "analyzer_samples": samples}


# --- 4. Query mismatch ----------------------------------------------------------------


def run_mismatch(args: argparse.Namespace) -> Dict[str, Any]:
    corpus = fixtures.mismatch_corpus()
    embedder = CachedEmbedder()
    # Noise makes the top 5 meaningful: documents that share everyday words but answer nothing.
    noise = fixtures.exact_match_corpus()["docs"][:150]
    docs = corpus["docs"] + noise
    vectors = embedder.embed_many([doc["text"] for doc in docs])
    transformer = LlmQueryTransformer(get_llm)
    results: Dict[str, Any] = {}
    examples: List[Dict[str, Any]] = []
    with TempIndex() as index:
        index.load(docs, vectors)
        retriever = HybridRetriever(index.manager)
        pipeline = RetrievalPipeline(retriever, embedder, transformer=transformer, reranker=FastEmbedCrossEncoder())
        for label, mode, rerank in (("off", "off", False), ("multi", "multi", False), ("hyde", "hyde", False), ("both", "both", False), ("off + rerank", "off", True), ("hyde + rerank", "hyde", True)):
            print("mismatch", label, flush=True)
            rows, latencies, errors = [], [], 0
            for query in corpus["queries"]:
                started = time.perf_counter()
                result = pipeline.run(query["query"], query["query"], ACL, top_k=10, options=PipelineOptions(query_mode=mode, candidates=30, rerank=rerank))
                latencies.append((time.perf_counter() - started) * 1000)
                ranked = [chunk["chunk_id"] for chunk in result.chunks]
                errors += len(result.trace.get("transform_errors", []))
                rows.append({"hit_at_1": float(ranked[:1] == [query["target"]]), "hit_at_5": float(query["target"] in ranked[:5]), "mrr": reciprocal_rank(ranked, {query["target"]: 1})})
                if label == "both" and len(examples) < 4:
                    examples.append({"query": query["query"], "target": query["target"], "variants": result.trace["variants"], "ranked": ranked[:3]})
            results[label] = {**summarize(rows), "latency_p50_ms": percentile(latencies, 50), "transform_errors": errors, "queries": len(rows)}
    return {"llm": BENCH_LLM, "num_docs": len(docs), "results": results, "examples": examples}


# --- 5. Parent-child ------------------------------------------------------------------


def run_parent_child(_: argparse.Namespace) -> Dict[str, Any]:
    return {
        "single_topic_sections": _parent_child_on(fixtures.parent_child_corpus(), child_tokens=50),
        "multi_topic_pages": _parent_child_on(fixtures.multi_topic_parent_child_corpus(), child_tokens=50),
    }


def _parent_child_on(corpus: Dict[str, Any], child_tokens: int) -> Dict[str, Any]:
    embedder = CachedEmbedder()
    results: Dict[str, Any] = {}
    # Other documents, so that a diluted embedding has something to lose to.
    noise = fixtures.exact_match_corpus()["docs"][:200]

    def records(strategy: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for doc in corpus["docs"]:
            if strategy == "parent_child":
                splitter = ParentChildSplitter(parent_tokens=1000, child_tokens=child_tokens, child_overlap=8)
                for number, child in enumerate(splitter.split(doc["text"])):
                    out.append({"id": f"{doc['id']}-c{number}", "doc_id": doc["id"], "text": child.text, "parent_id": f"{doc['id']}-p{child.parent_index}", "parent_content": child.parent_text})
            else:
                size = child_tokens if strategy == "small" else 1000
                chunker = create_chunker("sentence", max_tokens=size, overlap_tokens=8 if size == child_tokens else 0)
                for number, piece in enumerate(chunker.split(doc["text"])):
                    out.append({"id": f"{doc['id']}-c{number}", "doc_id": doc["id"], "text": piece})
        return out

    for strategy in ("small", "large", "parent_child"):
        print("parent_child", strategy, flush=True)
        docs = records(strategy) + noise
        vectors = embedder.embed_many([doc["text"] for doc in docs])
        with TempIndex() as index:
            index.load(docs, vectors)
            retriever = HybridRetriever(index.manager)
            rows, context_sizes = [], []
            for query in corpus["queries"]:
                outcome = retriever.search(query["query"], embedder.embed(query["query"]), ACL, top_k=30, candidate_k=30, fusion="dense")
                chunks, _ = expand_to_parents(outcome.authorized_chunks)
                top = chunks[:3]
                context = " ".join(chunk["content"] for chunk in top)
                context_sizes.append(len(context.split()))
                rows.append(
                    {
                        "right_section_at_1": float(bool(top) and str(top[0]["doc_id"]).startswith(query["target"])),
                        "answer_in_top_chunk": float(bool(top) and query["answer"] in top[0]["content"]),
                        "answer_in_top_3": float(query["answer"] in context),
                    }
                )
        results[strategy] = {**summarize(rows), "mean_context_words": round(statistics.mean(context_sizes), 1), "chunks_indexed": len(docs) - len(noise), "queries": len(rows)}
    return {"note": "Dense search only, so chunk size is the variable under test.", "child_tokens": child_tokens, "parent_tokens": 1000, "results": results}


# --- 6. Multi-hop ---------------------------------------------------------------------


def run_multi_hop(_: argparse.Namespace) -> Dict[str, Any]:
    corpus = fixtures.multi_hop_corpus()
    embedder = CachedEmbedder()
    vectors = embedder.embed_many([doc["text"] for doc in corpus["docs"]])
    planner = MultiHopPlanner(get_llm)
    single_rows, multi_rows, transcripts = [], [], []
    with TempIndex() as index:
        index.load(corpus["docs"], vectors)
        retriever = HybridRetriever(index.manager)
        pipeline = RetrievalPipeline(retriever, embedder)
        for query in corpus["queries"]:
            print("multi_hop", query["query"], flush=True)
            single = pipeline.run(query["query"], query["query"], ACL, top_k=4)
            single_ids = {chunk["chunk_id"] for chunk in single.chunks}
            single_context = " ".join(chunk["content"] for chunk in single.chunks)
            single_rows.append({"both_documents_found": float(set(query["targets"]) <= single_ids), "answer_in_context": float(query["answer"] in single_context)})
            hop = planner.run(query["query"], lambda text: pipeline.run(text, text, ACL, top_k=4))
            hop_ids = {chunk["chunk_id"] for chunk in hop.chunks}
            hop_context = " ".join(chunk["content"] for chunk in hop.chunks)
            multi_rows.append({"both_documents_found": float(set(query["targets"]) <= hop_ids), "answer_in_context": float(query["answer"] in hop_context)})
            transcripts.append({"query": query["query"], "answer": query["answer"], "hops": [vars(item) for item in hop.hops], "stopped": hop.stopped_reason})
    return {"llm": BENCH_LLM, "num_queries": len(corpus["queries"]), "single_pass": summarize(single_rows), "multi_hop": summarize(multi_rows), "transcripts": transcripts}


# --- 7. Lost in the middle ------------------------------------------------------------


def run_attention(args: argparse.Namespace) -> Dict[str, Any]:
    return _attention_on(fixtures.attention_contexts())


def run_attention_hard(args: argparse.Namespace) -> Dict[str, Any]:
    return _attention_on(fixtures.hard_attention_contexts())


def _attention_on(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Move one needle through a 15-chunk context, and see whether the model still finds it."""
    llm = get_llm(role="architect", temperature=0.0)
    size = 15
    positions: Dict[int, List[float]] = {position: [] for position in range(1, size + 1)}
    for position in range(1, size + 1):
        print("attention position", position, flush=True)
        for case in cases:
            chunks = list(case["distractors"])
            chunks.insert(position - 1, case["fact"])
            context = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunks))
            prompt = f"Answer using only the context. If it isn't there, reply NOT FOUND.\n\nContext:\n{context}\n\nQuestion: {case['question']}\nAnswer briefly."
            answer = str(getattr(llm.invoke(prompt), "content", ""))
            positions[position].append(float(case["answer"].lower() in answer.lower()))
    curve = {position: round(statistics.mean(values), 3) for position, values in positions.items()}
    order = reorder_for_attention(list(range(1, size + 1)))
    placed = {rank: order.index(rank) + 1 for rank in range(1, size + 1)}
    return {"llm": BENCH_LLM, "context_chunks": size, "facts": len(cases), "accuracy_by_position": curve, "reordered_position_of_rank": placed}


SECTIONS: Dict[str, Callable[[argparse.Namespace], Dict[str, Any]]] = {
    "quantization": run_quantization,
    "fusion": run_fusion,
    "rerank_depth": run_rerank_depth,
    "exact": run_exact,
    "mismatch": run_mismatch,
    "parent_child": run_parent_child,
    "multi_hop": run_multi_hop,
    "attention": run_attention,
    "attention_hard": run_attention_hard,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--llm", default="qwen3:8b", help="Ollama model for LLM sections.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for section in args.sections.split(","):
        started = time.perf_counter()
        result = SECTIONS[section](args)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        (args.out / f"{section}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", section, result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
