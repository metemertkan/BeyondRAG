"""Thin Domain Facade for BeyondRAG (`RBACKnowledgeBaseService`).

Delegates each responsibility to a dedicated, single-responsibility component via
domain ports (zero direct dependency on Elasticsearch, Gemini, Ollama, PyPDF,
ReportLab, or FastAPI):
  1. Index & Connection Management -> `IndexManagerPort` (`es_client.py`)
  2. PDF Ingestion                 -> `PdfIngestorPort` (`ingestor.py`)
  3. Text Extraction               -> `PdfTextExtractorPort` (`extractor.py`)
  4. Chunking                      -> `ChunkerPort` (`chunker.py`)
  5. Embedding                     -> `EmbedderPort` (`embeddings.py`)
  6. Document Persistence          -> `DocumentRepositoryPort` (`repository.py`)
  7. ACL Management                -> `acl.py`
  8. Hybrid Retrieval              -> `HybridRetrieverPort` (`retriever.py`)
  9. Cross-Lingual Query Handling  -> `CrossLingualPort` (`cross_lingual.py`)
  10. Answer Synthesis             -> `AnswerSynthesizerPort` (`synthesizer.py`)
  11. Demo Document Seeding        -> `DemoSeederPort` (`seeding.py`)
  12. Erasure & Source Sync        -> `ErasureService`, `SourceSyncHandler` (`erasure.py`)
  13. Retrieval Pipeline           -> `RetrievalPipeline` (`pipeline.py`), `MultiHopPlanner` (`multi_hop.py`)
  14. Tenant Scope                 -> `tenancy.py` (every read below names exactly one tenant)

Every method that reads or changes stored documents takes ``tenant`` as a required keyword.
There is deliberately no default: an endpoint that forgets it fails with a ``TypeError``
instead of answering from another tenant's data. Ingestion keeps its optional tenant
(``default``), because a write without one cannot spread across tenants.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, replace
from typing import Any, Dict, Final, List, Optional

from src.rag.acl import (
    PUBLIC_SENTINEL_TAG,
    build_effective_acl_tags,
    collect_available_tags,
    normalize_tags,
    resolve_document_acl,
)
from src.rag import telemetry
from src.rag.governance import DocumentGovernance, detect_conflicts
from src.rag.guardrails import GuardrailEngine, get_guardrails
from src.rag.multi_hop import MultiHopPlanner
from src.rag.pipeline import PipelineOptions, PipelineResult, RetrievalPipeline
from src.rag.profiles import DEFAULT_INDEX_PREFIX
from src.rag.tenancy import TenantNotFoundError, normalize_tenant, require_tenant
from src.rag.trust import ABSTAIN_MESSAGE, AnswerVerifier, LlmClaimVerifier, TrustReport, apply_trust_policy
from src.rag.ports import (
    AnswerSynthesizerPort,
    CrossLingualPort,
    DemoSeederPort,
    DocumentRepositoryPort,
    EmbedderPort,
    HybridRetrieverPort,
    IndexManagerPort,
    PdfIngestorPort,
    RerankerPort,
)

KB_INDEX_NAME: Final[str] = DEFAULT_INDEX_PREFIX


def _wire_default_components(
    es_url: str | None = None,
) -> tuple[
    IndexManagerPort,
    DocumentRepositoryPort,
    PdfIngestorPort,
    CrossLingualPort,
    EmbedderPort,
    HybridRetrieverPort,
    AnswerSynthesizerPort,
    DemoSeederPort,
]:
    """Composition root wiring the concrete infrastructure adapters."""
    from src.config import ELASTICSEARCH_URL
    from src.rag.chunker import create_chunker
    from src.rag.cross_lingual import DefaultCrossLingualAdapter
    from src.rag.embeddings import CrossLingualEmbedder
    from src.rag.es_client import ElasticsearchIndexManager
    from src.rag.extractor import DocumentParserRegistry
    from src.rag.ingestor import PdfIngestor
    from src.rag.repository import DEFAULT_REGISTRY_PATH, DEFAULT_TABULAR_PATH, DocumentRepository
    from src.rag.tabular_store import TabularStore
    from src.rag.parent_child import ParentChildSplitter, parent_child_enabled
    from src.rag.retriever import HybridRetriever
    from src.rag.seeding import DemoDocumentSeeder
    from src.rag.synthesizer import GroundedAnswerSynthesizer
    from src.rag.language_packs import get_language_pack_registry

    get_language_pack_registry().ensure_workspace_defaults()

    es_manager = ElasticsearchIndexManager(
        es_url=es_url or ELASTICSEARCH_URL,
        index_name=KB_INDEX_NAME,
    )
    extractor = DocumentParserRegistry()
    embedder = CrossLingualEmbedder()
    chunker = create_chunker(embedder=embedder)
    cross_lingual = DefaultCrossLingualAdapter()
    repository = DocumentRepository(
        es_manager=es_manager,
        registry_path=DEFAULT_REGISTRY_PATH,
        persist_registry=True,
        tabular_store=TabularStore(DEFAULT_TABULAR_PATH),
    )
    ingestor = PdfIngestor(
        extractor=extractor,
        chunker=chunker,
        cross_lingual=cross_lingual,
        embedder=embedder,
        repository=repository,
        parent_child=ParentChildSplitter() if parent_child_enabled() else None,
    )
    retriever = HybridRetriever(es_manager=es_manager)
    synthesizer = GroundedAnswerSynthesizer()
    seeder = DemoDocumentSeeder()
    return (
        es_manager,
        repository,
        ingestor,
        cross_lingual,
        embedder,
        retriever,
        synthesizer,
        seeder,
    )


def _create_pdf_bytes(title: str, pages: List[List[str]]) -> bytes:
    """Backward-compatible helper delegating PDF byte generation to `seeding.py`."""
    from src.rag.seeding import create_pdf_bytes

    return create_pdf_bytes(title=title, pages=pages)


class RBACKnowledgeBaseService:
    """Thin domain facade coordinating injected RAG ports."""

    def __init__(
        self,
        es_url: str | None = None,
        *,
        index_manager: IndexManagerPort | None = None,
        repository: DocumentRepositoryPort | None = None,
        ingestor: PdfIngestorPort | None = None,
        cross_lingual: CrossLingualPort | None = None,
        embedder: EmbedderPort | None = None,
        retriever: HybridRetrieverPort | None = None,
        synthesizer: AnswerSynthesizerPort | None = None,
        seeder: DemoSeederPort | None = None,
        reranker: RerankerPort | None = None,
        multi_hop_planner: MultiHopPlanner | None = None,
        answer_verifier: AnswerVerifier | None = None,
        guardrails: GuardrailEngine | None = None,
    ) -> None:
        # On the service, not on the HTTP route, so every caller goes through the same checks.
        self._guardrails = guardrails or get_guardrails()
        self._reranker_override = reranker
        self._answer_verifier = answer_verifier or AnswerVerifier(LlmClaimVerifier())
        self._multi_hop = multi_hop_planner or MultiHopPlanner()
        self._pipeline: RetrievalPipeline | None = None
        if any(
            c is None
            for c in (
                index_manager,
                repository,
                ingestor,
                cross_lingual,
                embedder,
                retriever,
                synthesizer,
                seeder,
            )
        ):
            (
                d_idx,
                d_repo,
                d_ing,
                d_xl,
                d_emb,
                d_ret,
                d_syn,
                d_seed,
            ) = _wire_default_components(es_url)
            self._index_manager: IndexManagerPort = index_manager or d_idx
            self._repository: DocumentRepositoryPort = repository or d_repo
            self._ingestor: PdfIngestorPort = ingestor or d_ing
            self._cross_lingual: CrossLingualPort = cross_lingual or d_xl
            self._embedder: EmbedderPort = embedder or d_emb
            self._retriever: HybridRetrieverPort = retriever or d_ret
            self._synthesizer: AnswerSynthesizerPort = synthesizer or d_syn
            self._seeder: DemoSeederPort = seeder or d_seed
        else:
            assert index_manager is not None
            assert repository is not None
            assert ingestor is not None
            assert cross_lingual is not None
            assert embedder is not None
            assert retriever is not None
            assert synthesizer is not None
            assert seeder is not None
            self._index_manager = index_manager
            self._repository = repository
            self._ingestor = ingestor
            self._cross_lingual = cross_lingual
            self._embedder = embedder
            self._retriever = retriever
            self._synthesizer = synthesizer
            self._seeder = seeder

    @property
    def es_url(self) -> str:
        return self._index_manager.es_url

    def _is_es_online(self) -> bool:
        return self._index_manager.is_online()

    def ingest_pdf_bytes(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: str | None = None,
        tenant: str | None = None,
        source_key: str | None = None,
        governance: DocumentGovernance | None = None,
        **ingest_options: Any,
    ) -> Dict[str, Any]:
        with telemetry.stage("ingest", tenant=normalize_tenant(tenant), bytes=len(pdf_bytes)):
            return self._ingestor.ingest(
                filename=filename,
                pdf_bytes=pdf_bytes,
                tags=tags,
                doc_id=doc_id,
                tenant=tenant,
                source_key=source_key,
                governance=governance,
                **ingest_options,
            )

    def list_documents(self, *, tenant: str, include_superseded: bool = True) -> List[Dict[str, Any]]:
        return self._repository.list_documents(
            tenant=require_tenant(tenant), include_superseded=include_superseded
        )

    def list_available_tags(self, *, tenant: str) -> List[str]:
        return collect_available_tags(self.list_documents(tenant=tenant))

    def _owned_document(self, doc_id: str, tenant: str) -> str:
        """Return the tenant scope, or raise not-found.

        A document that belongs to another tenant and a document that does not exist give
        the same error, so document ids cannot be probed across tenants.
        """
        scope = require_tenant(tenant)
        if self._repository.tenant_of(doc_id) != scope:  # type: ignore[attr-defined]
            raise TenantNotFoundError(f"Document not found: {doc_id}")
        return scope

    def update_document_tags(
        self,
        doc_id: str,
        tags: List[str] | str | None,
        *,
        tenant: str,
    ) -> Dict[str, Any]:
        scope = self._owned_document(doc_id, tenant)
        acl = resolve_document_acl(tags)
        self._repository.update_document_acl(
            doc_id=doc_id,
            acl_tags=acl.acl_tags,
            is_public=acl.is_public,
            tenant=scope,
        )
        return {
            "doc_id": doc_id,
            "tenant": scope,
            "is_public": acl.is_public,
            "tags": acl.visible_tags,
            "acl_tags": acl.acl_tags,
        }

    def delete_document(self, doc_id: str, *, tenant: str) -> Dict[str, Any]:
        scope = self._owned_document(doc_id, tenant)
        return self._repository.delete_document(doc_id, tenant=scope)

    def offboard_tenant(self, tenant: str) -> Dict[str, Any]:
        """Remove everything one tenant stored: chunks, files, tables, registry, concepts."""
        from src.rag.cross_lingual import get_concept_registry

        scope = require_tenant(tenant)
        receipt = self._repository.purge_tenant(scope)  # type: ignore[attr-defined]
        receipt["concept_entries"] = get_concept_registry().forget_tenant(scope)
        # The local count only. Metrics, traces and logs already sent to a backend keep the
        # tenant's name until that backend's retention removes them.
        receipt["token_ledger_entries"] = self._guardrails.ledger.forget_tenant(scope)
        return receipt

    def erasure_service(self) -> Any:
        """Hard-deletion coordinator wired to this service's stores."""
        from src.rag.cross_lingual import get_concept_registry
        from src.rag.erasure import ErasureService

        from src.config import WORKSPACE_DIR

        return ErasureService(
            self._repository,
            concept_registry=get_concept_registry(),
            audit_log_path=WORKSPACE_DIR / "erasure_log.jsonl",
        )

    def source_sync_handler(self) -> Any:
        from src.rag.erasure import SourceSyncHandler

        return SourceSyncHandler(self._repository, self.erasure_service())

    def seed_sample_pdfs(self) -> List[Dict[str, Any]]:
        return self._seeder.seed(self)

    def reindex_all_sample_pdfs(self) -> List[Dict[str, Any]]:
        self._index_manager.reset_index()
        # The registry is on disk and outlives the index. Left alone, it still lists the
        # samples, the seeder skips them as already loaded, and nothing can be searched.
        self._repository.forget_all_documents()
        return self.seed_sample_pdfs()

    def query_and_chat(
        self,
        question: str,
        user_roles: List[str] | str | None = None,
        top_k: int = 4,
        synthesis_mode: str = "fast",
        retrieval_profile: str | None = None,
        pipeline_options: Optional[PipelineOptions] = None,
        *,
        tenant: str,
    ) -> Dict[str, Any]:
        # First line on purpose: nothing below runs without exactly one valid tenant.
        tenant = require_tenant(tenant)
        with telemetry.request_scope(tenant, synthesis_mode, question) as record:
            response = self._answer(
                question, user_roles, top_k, synthesis_mode, retrieval_profile, pipeline_options, tenant
            )
            # Every request leaves with exactly one label. "It returned 200" is not one of them.
            record.outcome = telemetry.classify_outcome(record, synthesis_mode, response.get("trust_report"))
        # Read after the scope closes, so the duration and the reason are final.
        response["telemetry"] = record.to_dict()
        response["guardrails"]["decisions"] = list(record.guardrails)
        return response

    def _answer(
        self,
        question: str,
        user_roles: List[str] | str | None,
        top_k: int,
        synthesis_mode: str,
        retrieval_profile: str | None,
        pipeline_options: Optional[PipelineOptions],
        tenant: str,
    ) -> Dict[str, Any]:
        guard = self._guardrails
        guarded = guard.policy.enabled
        cleaned_roles, effective_acl_tags = build_effective_acl_tags(user_roles)
        options = pipeline_options or PipelineOptions.from_env()
        dropped_chunks: List[Dict[str, Any]] = []
        guardrail_report: Dict[str, Any] = {"enabled": guarded, "decisions": [], "dropped_chunks": dropped_chunks}

        if guarded:
            with telemetry.stage("guardrail_input"):
                verdict = guard.check_input(question)
            # From here on only the checked question exists. Personal data found in it is masked.
            question = verdict.question
            if verdict.blocked:
                telemetry.note("blocked_input", verdict.guardrail)
                return {
                    "question": question,
                    "tenant": tenant,
                    "is_cross_lingual": False,
                    "matched_concepts": [],
                    "expanded_query": "",
                    "user_roles": cleaned_roles,
                    "effective_acl_filter": effective_acl_tags,
                    "latency_ms": 0.0,
                    "authorized_chunks": [],
                    "masked_restricted_docs": [],
                    "es_query_dsl": {},
                    "conflicts": [],
                    "retrieval_trace": {},
                    "trust_report": None,
                    "answer": verdict.message,
                    "guardrails": guardrail_report,
                }
            uses_model = synthesis_mode == "llm" or options.query_mode != "off" or options.multi_hop
            if uses_model and guard.over_budget(tenant):
                # Still answers, from the documents alone. Search and its embedding call go on.
                telemetry.note("budget_exceeded", "the daily token budget is used up; answered without the model")
                synthesis_mode = "fast"
                options = replace(options, query_mode="off", multi_hop=False, verify=False)

        is_cjk_query = self._cross_lingual.is_cross_lingual(question)
        with telemetry.stage("query_understanding"):
            expanded_query, matched_concepts = self._cross_lingual.prepare_query(
                question=question,
                allow_llm_fallback=(synthesis_mode == "llm"),
                tenant=tenant,
            )

        pipeline = self._get_pipeline()
        # Only where a model reads the chunks. The fast answer quotes one chunk and obeys nothing.
        filter_context = guarded and (synthesis_mode == "llm" or options.multi_hop)

        def retrieve(search_question: str, search_text: str) -> PipelineResult:
            # Every search, on every hop, is pinned to this user's tenant and ACL tags.
            result = pipeline.run(
                question=search_question,
                search_text=search_text,
                acl_tags=effective_acl_tags,
                top_k=top_k,
                profile_name=retrieval_profile,
                options=options,
                tenant=tenant,
            )
            if filter_context:
                with telemetry.stage("guardrail_context"):
                    kept, dropped = guard.filter_context(list(result.chunks))
                if dropped:
                    result.chunks = kept
                    dropped_chunks.extend(dropped)
            return result

        if options.multi_hop:
            with telemetry.stage("multi_hop", max_hops=options.max_hops):
                hop_result = self._multi_hop.run(
                    question, lambda step: retrieve(step, step), max_hops=options.max_hops
                )
            latest = hop_result.latest or retrieve(question, expanded_query)
            authorized_chunks = hop_result.chunks or list(latest.chunks)
            masked_docs = hop_result.masked_restricted_docs
            trace = {
                **latest.trace,
                "hops": [asdict(hop) for hop in hop_result.hops],
                "stopped_reason": hop_result.stopped_reason,
            }
            search_text = f"{expanded_query} {' '.join(hop.search for hop in hop_result.hops)}"
        else:
            latest = retrieve(question, expanded_query)
            authorized_chunks = latest.chunks
            masked_docs = latest.masked_restricted_docs
            trace = latest.trace
            search_text = latest.search_text or expanded_query

        allowed_tables = {
            chunk["metadata"]["table_id"]
            for chunk in authorized_chunks
            if isinstance(chunk.get("metadata"), dict) and chunk["metadata"].get("table_id")
        }

        def run_table_query(table_query: Any) -> Dict[str, Any]:
            # Only tables whose summary this user just retrieved: that set has already
            # passed the ACL, current-version and expiry filters.
            if table_query.table_id not in allowed_tables:
                raise PermissionError(f"Table {table_query.table_id} was not retrieved for this query")
            return self._repository.tabular_store.query(  # type: ignore[attr-defined]
                table_query, effective_acl_tags, tenant=tenant
            )

        extra_args: Dict[str, Any] = {}
        synthesize_params = inspect.signature(self._synthesizer.synthesize).parameters
        if allowed_tables and "table_query_runner" in synthesize_params:
            extra_args["table_query_runner"] = run_table_query
        reranked = trace.get("rerank", {}).get("status") == "applied"
        if (reranked or options.multi_hop) and "ranked_by_model" in synthesize_params:
            extra_args["ranked_by_model"] = True

        trust_report: Optional[Dict[str, Any]] = None
        best = max((float(c.get("score") or 0.0) for c in authorized_chunks), default=0.0)
        if options.min_relevance is not None and reranked and authorized_chunks and best < options.min_relevance:
            # Refuse before generating: nothing retrieved looks like an answer, so no LLM call.
            answer = ABSTAIN_MESSAGE
            reason = f"The best match scored {best:.3f}, below the {options.min_relevance:.3f} threshold."
            telemetry.note("low_relevance", reason)
            trust_report = TrustReport(
                "abstained", [], ["relevance"], abstained=True, abstain_reason=reason
            ).to_dict()
        else:
            with telemetry.stage("generation", mode=synthesis_mode, chunks=len(authorized_chunks)):
                answer = self._synthesizer.synthesize(
                    question=question,
                    expanded_query=search_text,
                    matched_concepts=matched_concepts,
                    is_cjk_query=is_cjk_query,
                    user_roles=cleaned_roles,
                    authorized_chunks=authorized_chunks,
                    masked_restricted_docs=masked_docs,
                    synthesis_mode=synthesis_mode,
                    **extra_args,
                )
            if options.verify:
                with telemetry.stage("verification"):
                    trust_report, answer = self._verify_answer(answer, options)

        conflicts = detect_conflicts(authorized_chunks)
        if guarded:
            with telemetry.stage("guardrail_output"):
                checked = guard.check_output(answer)
                answer = checked.answer
                if checked.blocked:
                    telemetry.note("blocked_output", checked.guardrail)
                # The answer is one of four places in this response that carry document text.
                authorized_chunks, trace, conflicts = guard.redact_sources([authorized_chunks, trace, conflicts])

        return {
            "question": question,
            "tenant": tenant,
            "is_cross_lingual": is_cjk_query,
            "matched_concepts": matched_concepts,
            "expanded_query": expanded_query,
            "user_roles": cleaned_roles,
            "effective_acl_filter": effective_acl_tags,
            "latency_ms": latest.latency_ms,
            "authorized_chunks": authorized_chunks,
            "masked_restricted_docs": masked_docs,
            "es_query_dsl": latest.query_dsl,
            "conflicts": conflicts,
            "retrieval_trace": trace,
            "trust_report": trust_report,
            "answer": answer,
            "guardrails": guardrail_report,
        }

    def _verify_answer(self, answer: str, options: PipelineOptions) -> tuple[Dict[str, Any], str]:
        synthesizer = self._synthesizer
        if getattr(synthesizer, "last_mode_used", "fast") != "llm":
            # The fast answer quotes a retrieved chunk word for word, so there is nothing to verify.
            return {"verdict": "extractive", "sentences": [], "checks": [], "abstained": False}, answer
        sources = getattr(synthesizer, "last_sources", {})
        report = self._answer_verifier.report(
            getattr(synthesizer, "last_raw_answer", answer), sources, use_llm=options.verify_with_llm
        )
        answer = apply_trust_policy(report, answer, sources, options.trust_mode)
        return report.to_dict(), answer

    def _get_pipeline(self) -> RetrievalPipeline:
        if self._pipeline is None:
            # Loads lazily: a missing optional install shows up in the trace only if re-ranking is asked for.
            from src.rag.reranker import FastEmbedCrossEncoder

            reranker = self._reranker_override or FastEmbedCrossEncoder()
            self._pipeline = RetrievalPipeline(self._retriever, self._embedder, reranker=reranker)
        return self._pipeline


_KB_SERVICE_INSTANCE: RBACKnowledgeBaseService | None = None


def get_kb_service() -> RBACKnowledgeBaseService:
    """Return the singleton service. In demo mode it starts from freshly seeded sample tenants.

    ``RAG_AUTH_MODE=required`` never resets the index and never loads demo documents: a
    restart must not delete tenants' data or add documents nobody uploaded.
    """
    global _KB_SERVICE_INSTANCE
    if _KB_SERVICE_INSTANCE is None:
        from src.rag.auth import auth_mode

        service = RBACKnowledgeBaseService()
        with telemetry.stage("startup", mode=auth_mode()):
            if auth_mode() == "demo":
                service.reindex_all_sample_pdfs()
            else:
                service._index_manager.ensure_index()
        _KB_SERVICE_INSTANCE = service
    return _KB_SERVICE_INSTANCE
