# BeyondRAG Episode 2 — What shipped

**Branch:** `cursor/episode-2-ingestion-b42f`  
**Base:** `main`  
**Run it locally:** `git checkout cursor/episode-2-ingestion-b42f && docker compose up -d --build` then open `http://localhost:8090`.

## Claims → code / tests

| Claim | Where |
| --- | --- |
| Soft `SUPERSEDED` versioning (`source_key` + content hash / `doc_id`) | `src/rag/ingestor.py`, `src/rag/models.py`, `src/rag/repository.py`, `src/rag/retriever.py`; `tests/test_episode2_acceptance.py::test_revised_pdf_same_source_key_soft_supersedes_v1_out_of_search` |
| Retry same bytes → no duplicate chunks | `PdfIngestor.ingest` idempotent replace; `test_retry_same_bytes_still_idempotent` |
| Durable document registry | JSON sidecar `workspace/document_registry.json` via `DocumentRepository` (`persist_registry=True` in `service.py`) |
| Layout-aware PDF parser + offline fallback | Primary: **pdfplumber** (`PdfPlumberDocumentParser`). Offline fallback: **PyPDF** (`PyPdfDocumentParser`). Docling deferred (heavier deps / Docker). PyMuPDF deferred (AGPL). |
| Table + multi-column fixtures | `tests/fixtures/pdfs/*`; `scripts/generate_ep2_fixtures.py` |
| Before/after table extract | `test_before_after_table_retrieval_structure_beats_flat_pypdf` |
| Chunk metadata `block_type`, `page_number`, stable `chunk_id`; ACL preserved | `ingestor._build_chunks_from_blocks`; `test_ingestion_preserves_acl_tags_on_chunks` |
| Embedding contract: Ollama `bge-m3` + remote API + CI `feature_hash` | `create_embedding_provider` in `src/rag/embeddings.py` |
| Manifest mismatch fails fast at ingest | `PdfIngestor(expected_manifest=...)` + index manifest lock; `test_embedding_manifest_mismatch_fails_fast_at_ingest` |
| PENDING recorded before PROCESSING | `PdfIngestor.ingest`; `test_pending_state_actually_used_or_documented` |
| Tenant/scope in identity | `source_key = {tenant}:{filename}`; non-default tenant also prefixes `doc_id`. Default tenant keeps Ep1 `doc-{stem}-{12hex}` shape. |
| Languages config/data-driven | `src/rag/language_packs.py` + `workspace/language_packs.json`; EN/ZH default data; JA probe in `test_language_handling_is_config_or_data_driven_non_en_zh_case` |
| Upload accepts PDF + Markdown + txt | `/api/upload` in `src/web/app.py` (`.pdf`, `.md`, `.txt`). Excel **not** this episode; registry remains pluggable (`ExcelDocumentParser` stub / `register`). |
| Rollback on embed / bulk failure | `test_embed_failure_rolls_back_no_partial_chunks`, `test_bulk_index_failure_rolls_back_no_dual_current` |

## Exact G1 surface answers

1. **Layout parser:** pdfplumber (primary). **Offline fallback:** PyPDF plain `extract_text()`.
2. **Remote embedding API:** OpenAI-compatible HTTP embeddings via `RemoteApiEmbeddingProvider`. **Env vars:** `RAG_EMBEDDING_API_KEY` (or `OPENAI_API_KEY`), `RAG_EMBEDDING_API_BASE` (default `https://api.openai.com/v1`), `RAG_EMBEDDING_PROVIDER=remote_api`, `RAG_EMBEDDING_MODEL` (default `bge-m3`). Ollama path: `RAG_EMBEDDING_PROVIDER=ollama`. CI: `feature_hash`.
3. **md/txt upload this episode:** **yes** (via existing registry + upload API).
4. **Durable registry storage:** `workspace/document_registry.json` (documents, `current_by_source_key`, `index_embedding_manifest`).
5. **Run locally branch:** `cursor/episode-2-ingestion-b42f`.

## Explicitly not shipped

- Full OCR / handwriting
- Excel production parser
- Docling / PyMuPDF as default
- Invented benchmark numbers (before/after asserts marker presence only)
