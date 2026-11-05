# BeyondRAG Episode 2 — What shipped

**Run it locally:** `docker compose up -d --build` then open `http://localhost:8090`.

This file records the first Episode 2 increment (versioning, PDF layout, embedding contract), then the second (Excel, multilingual, semantic chunking, trust tiers, GDPR, CI evaluation) under "Ingestion challenges increment". Where the two disagree, the second is current.

## Ingestion challenges increment

| Claim | Where | Test |
| --- | --- | --- |
| Excel: merged cells, compound headers, cached formula results, cross-sheet lineage, hidden sheets skipped | `XlsxDocumentParser` | `test_excel_*` |
| Large sheets: one summary chunk plus SQLite; ACL-checked structured queries, no raw SQL | `tabular_store.py`, `service.py` | `test_large_table_*`, `test_table_*`, `test_llm_answer_receives_structured_query_result_as_a_source` |
| RTL: visual-order PDF text restored, exact match with source on the Arabic fixture | `multilingual.py`, `extractor.py` | `test_arabic_pdf_extraction_matches_source_text_exactly` |
| Chinese respects `max_tokens` (was one 960-token chunk at a 100-token limit) | `chunker.py` | `test_unspaced_chinese_respects_max_tokens_and_adds_no_spaces` |
| Arabic gets real vectors in `feature_hash` (was a placeholder), independent of tashkeel | `embeddings.py` | `test_arabic_gets_a_real_vector_and_tashkeel_does_not_change_it` |
| ES analyzer: Arabic normalization, Arabic-Indic digits, Han bigrams | `es_client.py` | `test_live_es_analyzer_*` |
| Scanned PDF pages via OCR port; warning when no engine is configured | `ocr.py` | `test_scanned_page_*`; the PaddleOCR adapter was also run by hand against `arabic_scanned_policy.pdf` |
| Semantic chunking with overlap, size limits, latency telemetry; adaptive routing | `chunker.py` | `test_semantic_chunker_*`, `test_adaptive_*` |
| Expiry filter, trust-tier re-ranking, conflict detection, precedence prompt | `governance.py`, `retriever.py`, `synthesizer.py` | `test_expired_*`, `test_trust_tier_*`, `test_conflict_*`, `test_llm_prompt_*`, `test_live_es_filters_*` |
| Erasure across all versions, files, tables, concepts and registry; verification; tombstones | `erasure.py`, `repository.py` | `test_erasure_*`, `test_tombstone_*`, `test_live_es_erasure_*` |
| Physical purge tracked separately because ES retains soft-deleted text | `confirm_physical_purge` | `test_live_es_erasure_*`, `test_physical_purge_timeout_*` |
| Source-system delete and archive events, idempotent; fail-closed API token | `SourceSyncHandler`, `app.py` | `test_source_*`, `test_destructive_endpoints_*` |
| CI gate: golden set, Context Precision, Faithfulness, deterministic stale-fact check; inconclusive never passes | `src/eval/`, `rag-eval.yml` | `test_gate_*`, `test_dead_judge_*`, `test_silent_fallback_*` |

## Claims → code / tests

| Claim | Where |
| --- | --- |
| Soft `SUPERSEDED` versioning (`source_key` + content hash / `doc_id`) | `src/rag/ingestor.py`, `src/rag/models.py`, `src/rag/repository.py`, `src/rag/retriever.py`; `tests/test_episode2_acceptance.py::test_revised_pdf_same_source_key_soft_supersedes_v1_out_of_search` |
| Retry same bytes → no duplicate chunks | `PdfIngestor.ingest` idempotent replace; `test_retry_same_bytes_still_idempotent` |
| Durable document registry | JSON sidecar `workspace/document_registry.json` via `DocumentRepository` (`persist_registry=True` in `service.py`) |
| Layout-aware PDF parser + offline fallback | Primary: **pdfplumber** (`PdfPlumberDocumentParser`). Offline fallback: **PyPDF** (`PyPdfDocumentParser`). Docling deferred (heavier deps / Docker). PyMuPDF deferred (AGPL). |
| Table + multi-column fixtures | `tests/fixtures/pdfs/*`; `scripts/generate_ep2_fixtures.py` |
| Before/after table extract (presence + blocks) | `test_before_after_table_retrieval_structure_beats_flat_pypdf` (v1 fixture — both paths may still contain `450,000,000`) |
| Before/after **label↔value mis-pair** (Short T2/T3) | Fixture `policy_mispaired_limits_table.pdf` (column-major paint); `test_flat_pypdf_mispairs_limit_label_layout_aware_keeps_row` — flat next-line neighbor is `FX forward approval floor`; layout-aware row keeps `Intraday cash buffer` + `HKD 450,000,000` |
| Chunk metadata `block_type`, `page_number`, stable `chunk_id`; ACL preserved | `ingestor._build_chunks_from_blocks`; `test_ingestion_preserves_acl_tags_on_chunks` |
| Embedding contract: Ollama `bge-m3` + remote API + CI `feature_hash` | `create_embedding_provider` in `src/rag/embeddings.py` |
| Manifest mismatch fails fast at ingest | `PdfIngestor(expected_manifest=...)` + index manifest lock; `test_embedding_manifest_mismatch_fails_fast_at_ingest` |
| PENDING recorded before PROCESSING | `PdfIngestor.ingest`; `test_pending_state_actually_used_or_documented` |
| Tenant/scope in identity | `source_key = {tenant}:{filename}`; non-default tenant also prefixes `doc_id`. Default tenant keeps Ep1 `doc-{stem}-{12hex}` shape. |
| Languages fully config/data-driven (no EN/ZH hardcode in orchestration) | Runtime JSON only: `config/language_packs.defaults.json` (+ workspace override). Loader `src/rag/language_packs.py` has no pack dicts. UI demo prompts/lang from `config/ui_demo.json` via `/api/ui-demo-config`. Tests in `tests/test_episode2_amendment_o1_o3.py`. |
| Explicit PDF parser degraded policy | `RAG_PARSER_ALLOW_DEGRADED`; registry records `parser_name` / `parser_degraded` / reason; same-bytes refuses flat overwrite of pdfplumber INDEXED |
| Stage-then-publish supersede | `DocumentRepository.publish_new_current`; v2 staged then INDEXED before v1→SUPERSEDED; exactly one current; v2 fail leaves v1 |
| Upload accepts PDF + Markdown + txt + xlsx | `/api/upload` in `src/web/app.py`. Excel added in the ingestion challenges increment. |
| Rollback on embed / bulk failure | `test_embed_failure_rolls_back_no_partial_chunks`, `test_bulk_index_failure_rolls_back_no_dual_current` |

## Exact G1 surface answers

1. **Layout parser:** pdfplumber (primary). **Offline fallback:** PyPDF plain `extract_text()`.
2. **Remote embedding API:** OpenAI-compatible HTTP embeddings via `RemoteApiEmbeddingProvider`. **Env vars:** `RAG_EMBEDDING_API_KEY` (or `OPENAI_API_KEY`), `RAG_EMBEDDING_API_BASE` (default `https://api.openai.com/v1`), `RAG_EMBEDDING_PROVIDER=remote_api`, `RAG_EMBEDDING_MODEL` (default `bge-m3`). Ollama path: `RAG_EMBEDDING_PROVIDER=ollama`. CI: `feature_hash`.
3. **md/txt upload this episode:** **yes** (via existing registry + upload API).
4. **Durable registry storage:** `workspace/document_registry.json` (documents, `current_by_source_key`, `index_embedding_manifest`).
5. **Run locally branch:** `cursor/episode-2-ingestion-b42f`.

## Explicitly not shipped / unclaimed

- Handwriting OCR; mixed-script OCR accuracy is limited (see README)
- Legacy `.xls` files (re-save as `.xlsx`)
- Docling / PyMuPDF as default
- Invented benchmark numbers
- **ZH query → EN table cell end-to-end retrieval demo:** **explicitly unclaimed this episode.** Language packs + JA detect are tested; a live ZH→EN table-cell retrieval under Finance + bge-m3 is not covered by a dedicated acceptance test (would need a live embedding model or a canned expansion that overclaims). Do not show it in Short/Hashnode as shipped proof.
- **v1 liquidity fixture alone does not prove “flat missed the number”** — on `policy_v1_liquidity_table.pdf` PyPDF still contains `450,000,000`. Use `policy_mispaired_limits_table.pdf` for the Short flattened→wrong-association beat.
- **Live Elasticsearch SUPERSEDED integration test:** not run in CI (no ES in unit env). Mapping fields + memory-filter tests ship; **docker-compose smoke** documented in README Episode 2 section.

## Claim-matrix R3 (Ep1 dual-version search on `main`)

**CLAIMED (backed):** `scripts/verify_main_dual_version_search.py` + `test_main_branch_search_query_returns_both_cash_buffer_markers` — git worktree of **`main`**, ingest v1+v2 with Ep1 `PdfIngestor`, run `HybridRetriever.search("intraday unencumbered HKD cash buffer threshold")`. Observed: `search_has_450=true`, `search_has_520=true`.

Ep2-branch storage sim (`test_ep1_baseline_without_shared_source_key_both_versions_stay_searchable`) remains a weaker companion (“searchable chunks contain both”) and is not sufficient alone for the Short/content claim.

## Dashboard / API (Short demo)

- Upload accepts `.pdf` / `.md` / `.txt`; optional `source_key` form field (UI + API).
- Document list shows lifecycle badges (`PENDING`/`PROCESSING`/`INDEXED`/`SUPERSEDED`/`FAILED`); SUPERSEDED rows remain visible.
- Chat citations surface `block_type` (+ page) when present on authorized chunks.
- Ollama / `remote_api` / Google embeddings **refuse silent pad/truncate** on native dimension mismatch (`EmbeddingManifestMismatchError`).

## Short fixture guidance (Manager / Shorts)

| Beat | Fixture | Observed failure mode |
| --- | --- | --- |
| Flattened table / wrong association | `tests/fixtures/pdfs/policy_mispaired_limits_table.pdf` | Flat next line after `Intraday cash buffer` = `FX forward approval floor`; value appears later in stream |
| Layout-aware fix | same file via pdfplumber | Table row `\| Intraday cash buffer \| HKD 450,000,000 \|` |
| Version supersede | `policy_v1` → `policy_v2` same `source_key` | Soft `SUPERSEDED` |
