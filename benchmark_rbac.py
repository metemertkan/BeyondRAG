#!/usr/bin/env python3
"""Standalone Elasticsearch 8.15 Pre-Filtered Hybrid RAG ACL Benchmark.

Demonstrates why naive top-K vector post-filtering causes 0% recall in multi-tenant
Hong Kong banking RAG systems (HKMA / SFC Chinese Wall compliance) and verifies
100% authorized recall using pre-filtered hybrid search (BM25 + int8_hnsw kNN filter).
"""

from __future__ import annotations

import json
import os
import sys
import time
from elasticsearch import Elasticsearch

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "hk-wealth-rag-demo"
QUERY_VECTOR = [0.95, 0.05, 0.01, 0.01]
QUERY_TEXT = "0005.HK 客戶保證金指引 Lombard lending margin guidelines"
TARGET_ROLE = "role:retail_rm"


def build_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(
        index=INDEX_NAME,
        mappings={
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "acl_groups": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 4,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "int8_hnsw",
                        "m": 16,
                        "ef_construction": 100,
                    },
                },
            }
        },
    )


def seed_documents(es: Elasticsearch) -> None:
    documents = [
        {
            "doc_id": f"MA-IBD-00{i}",
            "title": f"Project Victoria Cross-Border M&A Deal Memo #{i}",
            "content": (
                f"CONFIDENTIAL INVESTMENT BANKING DIVISION. Counterparty exposure and "
                f"0005.HK 客戶保證金指引 institutional bridge facility tranche {i}."
            ),
            "acl_groups": ["role:ma_ibd"],
            "embedding": [0.95, 0.05, 0.01, 0.01],
        }
        for i in range(1, 6)
    ] + [
        {
            "doc_id": "RETAIL-WM-001",
            "title": "0005.HK 客戶保證金指引 — Retail Wealth Lombard Margin Policy",
            "content": (
                "0005.HK 客戶保證金指引: Authorized Retail Relationship Managers (role:retail_rm) "
                "may extend up to 65% LTV against HKEX blue-chip equities under SFC Code of Conduct."
            ),
            "acl_groups": ["role:retail_rm"],
            "embedding": [0.89, 0.10, 0.02, 0.01],
        },
        {
            "doc_id": "RETAIL-WM-002",
            "title": "0005.HK 客戶保證金指引 — Intraday Margin Call Escalation",
            "content": (
                "0005.HK 客戶保證金指引: Retail Wealth Management intraday margin monitoring "
                "and HKMA consumer protection notification workflow for role:retail_rm."
            ),
            "acl_groups": ["role:retail_rm"],
            "embedding": [0.88, 0.11, 0.02, 0.01],
        },
    ]

    for doc in documents:
        es.index(index=INDEX_NAME, id=doc["doc_id"], document=doc)
    es.indices.refresh(index=INDEX_NAME)


def run_naive_post_filter(es: Elasticsearch) -> int:
    response = es.search(
        index=INDEX_NAME,
        size=5,
        knn={
            "field": "embedding",
            "query_vector": QUERY_VECTOR,
            "k": 5,
            "num_candidates": 10,
        },
    )
    hits = response["hits"]["hits"]
    authorized_hits = [
        h for h in hits if TARGET_ROLE in h["_source"].get("acl_groups", [])
    ]
    return len(authorized_hits)


def run_prefiltered_hybrid_search(es: Elasticsearch) -> tuple[int, float]:
    latencies_ms: list[float] = []
    authorized_count = 0
    acl_filter = {"term": {"acl_groups": TARGET_ROLE}}

    for _ in range(5):
        start = time.perf_counter()
        response = es.search(
            index=INDEX_NAME,
            size=5,
            query={
                "bool": {
                    "must": [{"match": {"content": QUERY_TEXT}}],
                    "filter": [acl_filter],
                }
            },
            knn={
                "field": "embedding",
                "query_vector": QUERY_VECTOR,
                "k": 5,
                "num_candidates": 10,
                "filter": acl_filter,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        hits = response["hits"]["hits"]
        authorized_count = len(
            [h for h in hits if TARGET_ROLE in h["_source"].get("acl_groups", [])]
        )

    latencies_ms.sort()
    p95_ms = round(latencies_ms[-1], 2)
    return authorized_count, p95_ms


def main() -> int:
    es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=5)
    build_index(es)
    seed_documents(es)

    post_recall = run_naive_post_filter(es)
    pre_recall, p95_latency_ms = run_prefiltered_hybrid_search(es)

    report = {
        "index": INDEX_NAME,
        "target_role": TARGET_ROLE,
        "query": QUERY_TEXT,
        "post_filter_recall": post_recall,
        "pre_filter_recall": pre_recall,
        "p95_latency_ms": p95_latency_ms,
        "passed": (post_recall == 0 and pre_recall == 2),
    }
    print("=== HK WEALTH RAG ACL BENCHMARK REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
