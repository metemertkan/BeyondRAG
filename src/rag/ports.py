"""Pure Domain Ports (Protocols) for BeyondRAG's Decoupled Component Architecture.

Defines the hexagonal boundary interfaces between the domain layer (`service.py`,
`ingestor.py`, `acl.py`, `chunker.py`) and infrastructure adapters (Elasticsearch,
Ollama, Gemini, PyPDF, ReportLab). Contains zero external library dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class HybridSearchOutcome:
    """Result returned by a HybridRetrieverPort execution."""

    authorized_chunks: List[Dict[str, Any]]
    masked_restricted_docs: List[Dict[str, Any]]
    latency_ms: float
    query_dsl: Dict[str, Any]


class IndexManagerPort(Protocol):
    """Port 1: Elasticsearch connection & index lifecycle management."""

    @property
    def es_url(self) -> str:
        ...

    def is_online(self) -> bool:
        ...

    def ensure_index(self) -> None:
        ...

    def reset_index(self) -> None:
        ...


class PdfTextExtractorPort(Protocol):
    """Port 2: Per-page text extraction from raw PDF bytes."""

    def extract_pages(self, pdf_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        ...


class DocumentParserPort(Protocol):
    """Port 2b: Pluggable document parser supporting multiple formats."""

    def extract_pages(self, filename: str, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
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

    def save_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        ...

    def list_documents(self) -> List[Dict[str, Any]]:
        ...

    def update_document_acl(
        self,
        doc_id: str,
        acl_tags: List[str],
        is_public: bool,
    ) -> None:
        ...

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        ...


class PdfIngestorPort(Protocol):
    """Port 6: End-to-end PDF ingestion pipeline."""

    def ingest(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: str | None = None,
    ) -> Dict[str, Any]:
        ...


class CrossLingualPort(Protocol):
    """Port 7: Cross-lingual (EN <-> ZH) query handling & concept extraction."""

    def is_cross_lingual(self, text: str) -> bool:
        ...

    def extract_concepts_at_ingestion(
        self,
        filename: str,
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        ...

    def prepare_query(
        self,
        question: str,
        allow_llm_fallback: bool = False,
    ) -> Tuple[str, List[str]]:
        ...


class HybridRetrieverPort(Protocol):
    """Port 8: Pre-filtered hybrid (`BM25 + int8_hnsw`) retrieval & RBAC mask auditing."""

    def search(
        self,
        expanded_query: str,
        query_vector: List[float],
        effective_acl_tags: List[str],
        top_k: int = 4,
        profile_name: str | None = None,
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
