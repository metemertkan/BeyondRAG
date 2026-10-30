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
"""

from __future__ import annotations

from typing import Any, Dict, Final, List

from src.rag.acl import (
    PUBLIC_SENTINEL_TAG,
    build_effective_acl_tags,
    collect_available_tags,
    normalize_tags,
    resolve_document_acl,
)
from src.rag.ports import (
    AnswerSynthesizerPort,
    CrossLingualPort,
    DemoSeederPort,
    DocumentRepositoryPort,
    EmbedderPort,
    HybridRetrieverPort,
    IndexManagerPort,
    PdfIngestorPort,
)

KB_INDEX_NAME: Final[str] = "hk-rbac-rag-kb"


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
    from src.rag.repository import DocumentRepository
    from src.rag.retriever import HybridRetriever
    from src.rag.seeding import DemoDocumentSeeder
    from src.rag.synthesizer import GroundedAnswerSynthesizer

    es_manager = ElasticsearchIndexManager(
        es_url=es_url or ELASTICSEARCH_URL,
        index_name=KB_INDEX_NAME,
    )
    extractor = DocumentParserRegistry()
    chunker = create_chunker()
    cross_lingual = DefaultCrossLingualAdapter()
    embedder = CrossLingualEmbedder()
    repository = DocumentRepository(es_manager=es_manager)
    ingestor = PdfIngestor(
        extractor=extractor,
        chunker=chunker,
        cross_lingual=cross_lingual,
        embedder=embedder,
        repository=repository,
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
    ) -> None:
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
    ) -> Dict[str, Any]:
        return self._ingestor.ingest(
            filename=filename,
            pdf_bytes=pdf_bytes,
            tags=tags,
            doc_id=doc_id,
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        return self._repository.list_documents()

    def list_available_tags(self) -> List[str]:
        return collect_available_tags(self.list_documents())

    def update_document_tags(
        self,
        doc_id: str,
        tags: List[str] | str | None,
    ) -> Dict[str, Any]:
        acl = resolve_document_acl(tags)
        self._repository.update_document_acl(
            doc_id=doc_id,
            acl_tags=acl.acl_tags,
            is_public=acl.is_public,
        )
        return {
            "doc_id": doc_id,
            "is_public": acl.is_public,
            "tags": acl.visible_tags,
            "acl_tags": acl.acl_tags,
        }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        return self._repository.delete_document(doc_id)

    def seed_sample_pdfs(self) -> List[Dict[str, Any]]:
        return self._seeder.seed(self)

    def reindex_all_sample_pdfs(self) -> List[Dict[str, Any]]:
        self._index_manager.reset_index()
        return self.seed_sample_pdfs()

    def query_and_chat(
        self,
        question: str,
        user_roles: List[str] | str | None = None,
        top_k: int = 4,
        synthesis_mode: str = "fast",
        retrieval_profile: str | None = None,
    ) -> Dict[str, Any]:
        cleaned_roles, effective_acl_tags = build_effective_acl_tags(user_roles)
        is_cjk_query = self._cross_lingual.is_cross_lingual(question)

        expanded_query, matched_concepts = self._cross_lingual.prepare_query(
            question=question,
            allow_llm_fallback=(synthesis_mode == "llm"),
        )
        query_vector = self._embedder.embed(expanded_query)

        search_outcome = self._retriever.search(
            expanded_query=expanded_query,
            query_vector=query_vector,
            effective_acl_tags=effective_acl_tags,
            top_k=top_k,
            profile_name=retrieval_profile,
        )

        answer = self._synthesizer.synthesize(
            question=question,
            expanded_query=expanded_query,
            matched_concepts=matched_concepts,
            is_cjk_query=is_cjk_query,
            user_roles=cleaned_roles,
            authorized_chunks=search_outcome.authorized_chunks,
            masked_restricted_docs=search_outcome.masked_restricted_docs,
            synthesis_mode=synthesis_mode,
        )

        return {
            "question": question,
            "is_cross_lingual": is_cjk_query,
            "matched_concepts": matched_concepts,
            "expanded_query": expanded_query,
            "user_roles": cleaned_roles,
            "effective_acl_filter": effective_acl_tags,
            "latency_ms": search_outcome.latency_ms,
            "authorized_chunks": search_outcome.authorized_chunks,
            "masked_restricted_docs": search_outcome.masked_restricted_docs,
            "es_query_dsl": search_outcome.query_dsl,
            "answer": answer,
        }


_KB_SERVICE_INSTANCE: RBACKnowledgeBaseService | None = None


def get_kb_service() -> RBACKnowledgeBaseService:
    """Return the singleton RBACKnowledgeBaseService instance and ensure sample PDFs are indexed."""
    global _KB_SERVICE_INSTANCE
    if _KB_SERVICE_INSTANCE is None:
        _KB_SERVICE_INSTANCE = RBACKnowledgeBaseService()
        _KB_SERVICE_INSTANCE.reindex_all_sample_pdfs()
    return _KB_SERVICE_INSTANCE
