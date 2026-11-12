# BeyondRAG Episode 2 — What shipped

**Run it locally:** `docker compose up -d --build` then open `http://localhost:8090`.

This file records the first Episode 2 increment (versioning, PDF layout, embedding contract), then the second (Excel, multilingual, semantic chunking, trust tiers, GDPR, CI evaluation) under "Ingestion challenges increment". Where the two disagree, the second is current.

## Cost reduction increment (Episode 7)

| Claim | Where | Test |
| --- | --- | --- |
| Embedding cache serves repeats, batches misses, returns the same vectors | `embedding_cache.py`, `embeddings.py` | `test_embedding_cache_serves_repeats_and_batches_misses`, `test_cached_vectors_match_uncached_ones` |
| Embedding cache is per tenant and purged with the document | `embedding_cache.py`, `service.py` | `test_embedding_cache_is_per_tenant_and_purged_with_the_document`, `test_deleting_a_document_deletes_the_answers_built_on_it` |
| An LLM cache hit is saved, not spent, and not charged to the budget | `llm_cache.py`, `telemetry.py` | `test_an_llm_cache_hit_is_saved_not_spent_and_not_charged_to_the_budget` |
| Answer cache needs the same tenant, roles and corpus generation | `answer_cache.py`, `repository.py` | `test_exact_hits_need_the_same_tenant_roles_and_generation`, `test_a_new_version_of_a_source_makes_the_cached_answer_unreachable` |
| Removals and index resets delete cached answers | `answer_cache.py`, `repository.py` | `test_a_removal_deletes_the_tenants_answers`, `test_an_index_reset_keeps_embeddings_but_drops_answers` |
| No cached answer outlives its sources' expiry; degraded answers aren't cached | `answer_cache.py`, `service.py` | `test_an_answer_never_outlives_its_sources_expiry`, `test_degraded_answers_are_not_cached` |
| Semantic hits respect the threshold and the guard | `answer_cache.py` | `test_semantic_hits_respect_the_threshold_and_the_guard`, `test_the_guard_rejects_near_misses` |
| Repeated questions are answered from the cache for free, even over budget | `service.py` | `test_a_repeated_question_is_answered_from_the_cache_for_free`, `test_a_cached_answer_is_served_even_over_budget` |
| Routing is explicit and labelled; the cascade escalates on failed checks | `routing.py`, `config.py`, `synthesizer.py` | `test_routing_is_off_unless_asked`, `test_the_cascade_*` |
| Compression keeps the answer sentence; compact headers and verification | `synthesizer.py`, `trust.py` | `test_compression_keeps_the_sentence_with_the_answer`, `test_compact_*` |
| Redis store and shared token ledger | `cache_store.py`, `guardrails.py` | `test_redis_store_keys_ttl_and_vector_scope` (needs Redis), `test_shared_ledger_counts_like_the_file_ledger` |
| Costs, caches, routing and tokens measured against answer quality | `scripts/ep7_benchmark.py` | Run by hand |

Not shipped: hosted prompt caching (local models only), per-entry purge of cached answers by document (a removal clears the tenant's answers), and a semantic cache that is safe without the guard.

## Observability and guardrails increment (Episode 6)

| Claim | Where | Test |
| --- | --- | --- |
| A model call inside a question is a child span with tokens, stage and tenant | `telemetry.py`, `config.py` | `test_a_model_call_inside_a_request_is_a_child_span_with_tokens`, `test_token_metrics_carry_stage_and_tenant` |
| Direct HTTP model calls (concept extraction, embeddings) report usage and errors | `cross_lingual.py`, `embeddings.py` | `test_direct_http_calls_report_usage_and_errors` |
| Every answer leaves with exactly one outcome; a missing one or an exception is `error` | `telemetry.py`, `service.py` | `test_outcome_classification`, `test_a_request_without_an_outcome_is_counted_as_an_error`, `test_every_answer_carries_a_trace_id_tokens_and_an_outcome` |
| A model failure and an empty model answer are degraded, not ok | `synthesizer.py` | `test_a_model_failure_is_degraded_not_ok`, `test_an_empty_model_answer_is_degraded_too` |
| Personal data is masked in telemetry, log lines and guarded responses | `pii.py`, `telemetry.py`, `guardrails.py` | `test_redaction_*`, `test_event_details_are_masked`, `test_the_wide_event_log_line_has_no_question_text`, `test_personal_data_is_masked_in_the_answer_and_the_returned_chunks` |
| Blocked questions never reach search or the model | `service.py`, `guardrails.py` | `test_a_blocked_question_never_reaches_search_or_the_model` |
| Planted instructions are dropped before the model reads them | `guardrails.py` | `test_a_planted_instruction_is_dropped_before_the_model_reads_it` |
| Daily token budget per tenant degrades to the fast answer | `guardrails.py` | `test_over_budget_tenants_get_the_fast_answer_and_no_model_call`, `test_model_tokens_are_added_to_the_tenant_ledger` |
| Grafana, Prometheus, Tempo and Loki as an opt-in compose profile, with a dashboard and alerts | `docker-compose.yml`, `observability/` | Run by hand: `docker compose --profile observability up -d lgtm` |
| Failures, tokens, guardrails, personal data, planted instructions, budget, overhead measured | `scripts/ep6_benchmark.py` | Run by hand |

Not shipped: guardrail rules beyond English, detection of names and addresses, a shared token ledger for more than one process, retention or erasure of tenant names in the telemetry backends, and a production-grade Grafana setup.

## Multi-tenant isolation increment (Episode 5)

| Claim | Where | Test |
| --- | --- | --- |
| Identity from a signed token; forged, expired, wildcard and two-tenant tokens refused | `auth.py`, `app.py` | `test_token_*`, `test_changing_the_tenant_inside_a_token_breaks_it`, `test_data_endpoints_refuse_a_request_without_a_valid_token` |
| Roles and tenant typed into a request are ignored | `app.py` | `test_roles_and_tenant_typed_into_the_request_are_ignored`, `test_an_upload_lands_in_the_token_tenant_whatever_the_form_says` |
| A read names exactly one tenant, with no default and no wildcard | `tenancy.py`, `retriever.py` | `test_a_read_needs_exactly_one_valid_tenant`, `test_search_without_a_tenant_does_not_run` |
| Tenant filter in BM25, kNN and the masked-documents audit | `retriever.py` | `test_tenant_filter_is_in_both_halves_of_the_hybrid_query`, `test_live_pool_filters_*` |
| Pool and silo layouts | `es_client.py`, `profiles.py` | `test_silo_mode_gives_each_tenant_a_concrete_index_and_keeps_the_filter` |
| Document ids, source keys and tables scoped by tenant | `repository.py`, `tabular_store.py` | `test_same_file_name_in_two_tenants_*`, `test_same_bytes_in_two_tenants_*`, `test_same_table_id_in_two_tenants_*` |
| A foreign document answers like a missing one | `repository.py`, `app.py` | `test_another_tenants_document_looks_exactly_like_a_missing_one` |
| Erasure, tombstones and sync events stay inside one tenant | `erasure.py` | `test_erasing_a_source_in_one_tenant_*`, `test_a_tombstone_blocks_only_the_tenant_that_erased`, `test_two_tenants_can_send_the_same_event_id` |
| Concepts and cached queries stay inside their tenant | `cross_lingual.py` | `test_concepts_and_cached_queries_stay_inside_their_tenant` |
| Writes need a write role in the caller's own tenant | `app.py` | `test_changing_documents_needs_a_write_role`, `test_an_admin_of_one_tenant_is_nobody_in_another` |
| Operator endpoints take one explicit tenant; offboarding removes one tenant only | `app.py`, `service.py`, `repository.py` | `test_operator_endpoints_need_*`, `test_offboarding_*` |
| Health reports liveness only; no CORS grant by default; no demo token in required mode | `app.py` | `test_health_says_nothing_about_the_data`, `test_browsers_on_other_sites_*`, `test_demo_token_exists_only_in_demo_mode` |
| 22 attacks against the code before and after, token forgery, restart | `scripts/ep5_attack_runner.py`, `scripts/ep5_benchmark.py` | Run by hand: `--sections attacks` |

Not shipped: tokens from an identity provider with public keys, per-tenant encryption keys, rate limits or quotas, and a defence against the shared BM25 statistics of the pool layout.

## Trustworthy answers increment (Episode 4)

| Claim | Where | Test |
| --- | --- | --- |
| Numbered sources and per-sentence citations, rendered as file names | `synthesizer.py`, `trust.py` | `test_citations_*`, `test_llm_answer_cites_*` |
| Value check: numbers and codes must appear in a cited source | `trust.py` | `test_value_check_*`, `test_values_cover_*`, `test_spelled_out_*` |
| LLM claim check, failures reported, verdicts matched even when numbered from 0 | `trust.py` | `test_llm_verdicts_*`, `test_llm_failure_*`, `test_verdicts_*` |
| `NOT_FOUND` abstention, including wrapped forms | `trust.py`, `synthesizer.py` | `test_not_found_*`, `test_wrapped_not_found_*`, `test_llm_saying_not_found_*` |
| Policies: annotate, strip, abstain | `trust.py` | `test_annotate_*`, `test_strip_*`, `test_abstain_*` |
| Relevance gate refuses before any LLM call | `service.py` | `test_relevance_gate_*` |
| Verification only sees authorized sources | `service.py` | `test_verification_never_sees_restricted_*` |
| Trust report panel and controls in the Lab | `index.html`, `app.py` | Exercised by hand |

## Retrieval at scale increment (Episode 3)

| Claim | Where | Test |
| --- | --- | --- |
| Configurable vector index type, `int8_hnsw` by default | `es_client.py` | `test_vector_index_type_*`, `test_unknown_vector_index_type_*` |
| Codes stay whole in a `content.codes` subfield | `es_client.py`, `retriever.py` | `test_live_standard_analyzer_*`, `test_live_sentence_end_*`, `test_live_exact_code_*` |
| Reciprocal Rank Fusion over separate BM25 and dense searches, with the ACL on both | `fusion.py`, `retriever.py` | `test_rrf_*`, `test_live_rrf_*`, `test_live_acl_*`, `test_acl_filters_*` |
| Query rewrites and HyDE, original always kept, HyDE dense only, failures reported | `query_transform.py`, `pipeline.py` | `test_rewrites_*`, `test_hyde_*`, `test_llm_failure_*`, `test_every_variant_*` |
| Parent-child: only children embedded, parent stored on the child, covered by ACL, versioning and erasure | `parent_child.py`, `ingestor.py` | `test_children_*`, `test_expansion_*`, `test_retrieval_returns_the_parent_*`, `test_parent_inherits_*`, `test_erasure_removes_parents_*`, `test_a_new_version_*` |
| Two-stage re-ranking; failure keeps first-stage order and is reported; ACL applies before it | `reranker.py`, `pipeline.py` | `test_reranker_*`, `test_rerank_scores_*`, `test_missing_reranker_*`, `test_acl_is_enforced_before_*` |
| Multi-hop: dependent searches, capped hops, stops when a hop finds nothing, ACL on every hop | `multi_hop.py`, `service.py` | `test_second_hop_*`, `test_chain_stops_*`, `test_hops_are_capped`, `test_restricted_second_hop_*` |
| Context reordering for the LLM prompt | `fusion.py`, `synthesizer.py` | `test_reorder_*`, `test_llm_prompt_uses_*` |
| Retrieval Lab controls and a retrieval trace in the UI | `index.html`, `app.py`, `seeding.py` | Exercised by hand, not by an automated test |

Not implemented: GraphRAG. Measurements are in the README, and they include techniques that showed no clear benefit in our tests.

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
