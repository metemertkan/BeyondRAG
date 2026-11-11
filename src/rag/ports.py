"""Pure Domain Ports (Protocols) for BeyondRAG's Decoupled Component Architecture.

Defines the hexagonal boundary interfaces between the domain layer (`service.py`,
`ingestor.py`, `acl.py`, `chunker.py`) and infrastructure adapters (Elasticsearch,
Ollama, Gemini, pdfplumber, PyPDF, ReportLab). Contains zero external library dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from src.rag.models import IngestionReport, ParsedDocument


@dataclass(frozen=True)
class HybridSearchOutcome:
    """Result returned by a HybridRetrieverPort execution."""

    authorized_chunks: List[Dict[str, Any]]
    masked_restricted_docs: List[Dict[str, Any]]
    latency_ms: float
    query_dsl: Dict[str, Any]
    # Chunk ids per ranking ("lexical", "dense") before fusion, for tracing.
    rankings: Dict[str, List[str]] = field(default_factory=dict)


class RerankerPort(Protocol):
    """Cross-encoder scoring: how well each text answers this query, read together."""

    @property
    def model_name(self) -> str:
        ...

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        ...


class IndexManagerPort(Protocol):
    """Port 1: Elasticsearch connection & index lifecycle management."""

    @property
    def es_url(self) -> str:
        ...

    def is_online(self) -> bool:
        ...

    def index_for(self, tenant: str | None) -> str:
        ...

    def ensure_index(self, tenant: str | None = None) -> str:
        ...

    def reset_index(self) -> None:
        ...


class PdfTextExtractorPort(Protocol):
    """Port 2: Per-page text extraction from raw PDF bytes."""

    def extract_pages(self, pdf_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        ...


class DocumentParserPort(Protocol):
    """Port 2b: Pluggable document parser supporting multiple formats.

    Implementations may also expose ``extract_document`` returning a
    :class:`ParsedDocument` with structure-preserving blocks (tables, columns).
    """

    def extract_pages(self, filename: str, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        ...


class StructuredDocumentParserPort(DocumentParserPort, Protocol):
    """Optional richer parse path used when structure metadata is available."""

    def extract_document(self, filename: str, file_bytes: bytes) -> ParsedDocument:
        ...


class ChunkerPort(Protocol):
    """Port 3: Text segmentation and chunking."""

    def split(self, text: str) -> List[str]:
        ...


class EmbedderPort(Protocol):
    """Port 4: Dense vector embedding."""

    def embed(self, text: str) -> List[float]:
        ...


class DocumentRepositoryPort(Protocol):
    """Port 5: Document and chunk persistence."""

    def save_upload_file(self, doc_id: str, filename: str, pdf_bytes: bytes) -> Path:
        ...

    def delete_upload_file(self, doc_id: str) -> None:
        ...

    def delete_chunks(self, doc_id: str, *, tenant: str | None = None) -> None:
        ...

    def save_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        ...

    def list_documents(self, *, tenant: str, include_superseded: bool = True) -> List[Dict[str, Any]]:
        """One tenant's documents. The tenant is required: there is no cross-tenant listing."""
        ...

    def update_document_acl(
        self,
        doc_id: str,
        acl_tags: List[str],
        is_public: bool,
        *,
        tenant: str | None = None,
    ) -> None:
        ...

    def delete_document(self, doc_id: str, *, tenant: str | None = None) -> Dict[str, Any]:
        ...

    def forget_all_documents(self) -> int:
        """Drop every registry entry, saved copy and table. Call after the index was reset."""
        ...


class DocumentRegistryPort(Protocol):
    """Durable document registry for lifecycle + version tracking."""

    def get_document_state(self, doc_id: str) -> Optional[IngestionReport]:
        ...

    def set_document_state(self, report: IngestionReport) -> None:
        ...

    def get_current_doc_id(self, source_key: str, tenant: str | None = None) -> Optional[str]:
        """A logical document is (tenant, source_key), never source_key alone."""
        ...

    def mark_superseded(self, doc_id: str) -> None:
        ...

    def get_index_embedding_manifest(self) -> Optional[str]:
        ...

    def set_index_embedding_manifest(self, manifest_id: str) -> None:
        ...

    def list_registry_documents(
        self, *, current_only: bool = False, tenant: str | None = None
    ) -> List[IngestionReport]:
        ...


class PdfIngestorPort(Protocol):
    """Port 6: End-to-end document ingestion pipeline."""

    def ingest(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: str | None = None,
        tenant: str | None = None,
        source_key: str | None = None,
        governance: Any = None,
    ) -> Dict[str, Any]:
        ...


class CrossLingualPort(Protocol):
    """Port 7: Cross-lingual query handling & concept extraction.

    Language-agnostic at the port boundary — adapters may add languages
    without changing callers.
    """

    def is_cross_lingual(self, text: str) -> bool:
        ...

    def extract_concepts_at_ingestion(
        self,
        filename: str,
        full_text: str,
        tenant: str | None = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        ...

    def prepare_query(
        self,
        question: str,
        allow_llm_fallback: bool = False,
        tenant: str | None = None,
    ) -> Tuple[str, List[str]]:
        ...


class HybridRetrieverPort(Protocol):
    """Port 8: Pre-filtered hybrid (`BM25 + int8_hnsw`) retrieval & RBAC mask auditing.

    ``tenant`` is keyword-only and has no default. A search that forgets it fails with a
    ``TypeError`` instead of running across every tenant.
    """

    def search(
        self,
        expanded_query: str,
        query_vector: Optional[List[float]],
        effective_acl_tags: List[str],
        top_k: int = 4,
        profile_name: str | None = None,
        fusion: str | None = None,
        candidate_k: int | None = None,
        *,
        tenant: str,
    ) -> HybridSearchOutcome:
        ...


class AnswerSynthesizerPort(Protocol):
    """Port 9: Grounded answer and RBAC block notice synthesis."""

    def synthesize(
        self,
        question: str,
        expanded_query: str,
        matched_concepts: List[str],
        is_cjk_query: bool,
        user_roles: List[str],
        authorized_chunks: List[Dict[str, Any]],
        masked_restricted_docs: List[Dict[str, Any]],
        synthesis_mode: str = "fast",
    ) -> str:
        ...


class DemoSeederPort(Protocol):
    """Port 10: Demo document seeding."""

    def seed(self, service: Any) -> List[Dict[str, Any]]:
        ...
