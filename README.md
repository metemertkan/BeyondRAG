# BeyondRAG — Enterprise RBAC & Cross-Lingual RAG Sandbox

> **Series:** *Beyond a PoC* — Episode #01: **Access Management (RBAC) & Cross-Language Retrieval**

Most RAG tutorials stop at chunking a single PDF for a single user. When enterprises deploy a **company-wide shared knowledge base**, two production challenges immediately break naive RAG pipelines:

1. **Access Management (RBAC Pre-Filtering):** Documents uploaded by different departments carry role tags (`finance`, `operation`, or untagged `PUBLIC`). Filtering permissions in Python *after* running an unfiltered `top-K` vector search causes **0% recall starvation** whenever restricted chunks saturate the `top-K` HNSW candidate slots. `BeyondRAG` pushes `terms` pre-filters directly into **Elasticsearch 8.15** across both BM25 and `int8_hnsw` kNN.
2. **Cross-Language (`ZH ↔ EN`) Retrieval:** Enterprise policies are authored in **English**, while teams frequently query the knowledge base in **Chinese** (e.g., asking `「最低港元現金儲備緩衝是多少？」` against an English Treasury PDF). `BeyondRAG` aligns bilingual domain concepts into a shared 256-dim L2-normalized vector space while strictly enforcing RBAC boundaries.

---

## Architecture Comparison: Naive Post-Filtering vs. `BeyondRAG`

```mermaid
flowchart LR
    subgraph Naive["❌ Naive RAG (Python Post-Filter)"]
        N1["Unfiltered Top-K Vector Search"] --> N2["Python Drops Unauthorized Hits"] --> N3["❌ 0% Recall Starvation"]
    end

    subgraph Beyond["✅ BeyondRAG (ES 8.15 Pre-Filter + Cross-Lingual)"]
        B1["ZH/EN Query + Active Role Token"] --> B2["ES 8.15 Storage-Engine Pre-Filter<br/>(BM25 + 256d int8_hnsw)"] --> B3["✅ 100% Authorized Recall (<10ms)"]
    end
```

---

## Quickstart (Docker Compose Sandbox)

```bash
git clone https://github.com/metemertkan/BeyondRAG.git
cd BeyondRAG
docker compose up -d --build
```

Open **`http://localhost:8090`** in your browser:

1. **Left Panel (`1. PDF Upload & Knowledge Base`):**
   - Upload any `.pdf` file and assign role tags (`finance`, `operation`) or leave untagged (`PUBLIC`).
   - View all indexed PDFs in the knowledge base and their live `✓ Unlocked` / `🔒 Locked` state as you switch roles in the top bar.
2. **Right Panel (`2. Multi-Lingual RBAC Chat`):**
   - Click the pre-generated Chinese or English queries (`[Public 公開]`, `[Finance 財務標籤]`, `[Operation 營運標籤]`) while toggling between `Public Only`, `Finance Role`, `Operation Role`, and `All Roles`.

---

## Reproducing the RBAC Pre-Filter vs. Post-Filter Benchmark

Run the standalone Elasticsearch 8.15 benchmark inside the container:

```bash
docker compose exec beyondrag-app python benchmark_rbac.py
```

| Retrieval Strategy | Authorized Recall | P95 Latency | Behavior |
| :--- | :---: | :---: | :--- |
| **Query 1: Unfiltered Top-5 kNN + Python Post-Filter** | `0 / 2` (**0%**) | N/A | Restricted chunks fill all 5 HNSW slots; Python drops all 5 hits. |
| **Query 2: ES 8.15 Pre-Filtered Hybrid (`BM25 + int8_hnsw`)** | `2 / 2` (**100%**) | **`~3–7 ms`** | Pushes `terms: {"acl_tags": ["__public__", ...roles]}` into the storage engine. |

---

## Repository Structure

```text
BeyondRAG/
├── Dockerfile                # Python 3.12 container image for the FastAPI + RAG service
├── docker-compose.yml        # Elasticsearch 8.15 + BeyondRAG Web UI sandbox
├── pyproject.toml            # Python package & dependency definitions
├── web_server.py             # FastAPI Uvicorn entrypoint (port 8090)
├── benchmark_rbac.py         # Standalone ES 8.15 Pre-Filter vs Post-Filter benchmark
├── src/
│   ├── config.py             # Environment & Dual-LLM router (Gemini + Ollama fallback)
│   ├── rag/
│   │   ├── service.py        # Thin Domain Facade (delegates to injected ports; zero infra imports)
│   │   ├── models.py         # Pure Domain: IngestionState, DocumentChunk, Concept, IngestionReport
│   │   ├── ports.py          # Pure Domain: Protocol interfaces for decoupled hexagonal architecture
│   │   ├── acl.py            # 1. ACL Management & Role Hierarchy (AuthorizationPolicy, expansion)
│   │   ├── chunker.py        # 2. Chunking (Pluggable sliding window, paragraph, sentence & recursive chunkers)
│   │   ├── ingestor.py       # 3. Document Ingestion (Idempotent lifecycle, rollback, deterministic IDs)
│   │   ├── es_client.py      # 4. ES Connection & Index Management (int8_hnsw + slowlog lifecycle)
│   │   ├── extractor.py      # 5. Pluggable Document Parsers (PyPDF, PlainText / Markdown registry)
│   │   ├── embeddings.py     # 6. Embedding Providers & Manifests (FeatureHash, Ollama, Google, Fake)
│   │   ├── repository.py     # 7. Document Persistence (Bulk indexing, atomic replacement, uploads)
│   │   ├── profiles.py       # 8. Retrieval Profiles & Tenants (precision, high_recall, executive_summary)
│   │   ├── retriever.py      # 9. Hybrid Retrieval (ES 8.15 Pre-Filtered BM25 + int8_hnsw & mask audit)
│   │   ├── cross_lingual.py  # 10. Multilingual Query Handling (Rule-based detector, query translator)
│   │   ├── synthesizer.py    # 11. Answer Synthesis (grounded citations & RBAC block notices)
│   │   └── seeding.py        # 12. Demo Document Seeding (ReportLab PDF generation & sample seeding)
│   └── web/
│       ├── app.py            # REST API endpoints (/api/upload, /api/documents, /api/chat)
│       └── static/
│           └── index.html    # 2-Section Web UI (PDF Upload & KB List + Multi-Lingual Chat)
├── tests/                    # Comprehensive unit tests (chunker, embeddings, multilingual, profiles, ingestion)
└── workspace/
    ├── dynamic_concepts.json # Persisted concept mappings
    └── uploads/              # Persistent storage for uploaded & seeded PDFs
```

---

## Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RAG_EMBEDDING_PROVIDER` | `feature_hash` | Active embedding provider: `feature_hash`, `ollama`, `google`, or `fake`. |
| `RAG_EMBEDDING_DIMS` | `256` | Vector dimensions for Elasticsearch dense_vector indexing. |
| `RAG_EMBEDDING_MODEL` | `bge-m3` | Embedding model identifier when using `ollama` or `google`. |
| `RAG_CHUNKING_STRATEGY` | `sliding_window` | Active chunking algorithm: `sliding_window`, `paragraph`, `sentence`, `recursive`. |
| `RAG_CHUNK_MAX_TOKENS` | `500` | Maximum token/character limit per chunk. |
| `RAG_CHUNK_OVERLAP_TOKENS`| `80` | Overlap budget between consecutive chunks. |
| `RAG_INDEX_PREFIX` | `hk-rbac-rag-kb` | Elasticsearch index name or prefix. |
| `RAG_TENANT_MODE` | `shared` | Multi-tenancy mode: `shared` or `per_tenant`. |
| `RAG_BASELINE_ROLES` | `finance,operation` | Default enterprise baseline roles. |
| `RAG_CORPUS_LANGUAGE` | `en` | Primary language of the document corpus. |

---

## Running Automated Tests

```bash
# Run the complete test suite (21 unit & integration tests)
pytest tests/ -v
```

---

## *Beyond a PoC* Series Roadmap

- [x] **Episode #01 (`BeyondRAG`):** Access Management (RBAC Pre-Filtering) & Dynamic LLM Cross-Language (`ZH ↔ EN`) Retrieval in Elasticsearch 8.15
- [ ] **Episode #02 (Up Next):** Deep Dive into Production Document Ingestion & Multilingual Dense Embeddings (`bge-m3`, Layout-Aware Parsing & Tables)
