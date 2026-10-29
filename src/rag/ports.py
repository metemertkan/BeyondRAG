"""Pure Domain Ports (Protocols) and Text Chunking Rules for BeyondRAG.

Defines the hexagonal boundary interfaces between the domain service (`service.py`)
and external infrastructure adapters (Elasticsearch, Ollama, Gemini, PyPDF).
Contains zero external library dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class HybridSearchOutcome:
    """Result returned by a ChunkStorePort hybrid search execution."""

    authorized_chunks: List[Dict[str, Any]]
    masked_restricted_docs: List[Dict[str, Any]]
    latency_ms: float
    query_dsl: Dict[str, Any]


class PdfTextExtractorPort(Protocol):
    """Port for extracting per-page text and persisting uploaded PDF bytes."""

    def save_upload(self, doc_id: str, filename: str, pdf_bytes: bytes) -> None:
        ...

    def delete_upload(self, doc_id: str) -> None:
        ...

    def extract_pages(self, pdf_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        """Return (total_page_count, list of (page_number, extracted_text))."""
        ...


class CrossLingualPort(Protocol):
    """Port for cross-lingual (EN <-> ZH) concept extraction, query expansion, and embedding."""

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
        """Return (expanded_bilingual_query, matched_concept_ids)."""
        ...

    def embed(self, text: str) -> List[float]:
        ...


class ChunkStorePort(Protocol):
    """Port for storing, updating, deleting, and pre-filter searching document chunks."""

    @property
    def backend_url(self) -> str:
        ...

    def is_online(self) -> bool:
        ...

    def index_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        ...

    def list_all_chunk_metadata(self) -> List[Dict[str, Any]]:
        ...

    def update_document_acl(
        self,
        doc_id: str,
        acl_tags: List[str],
        is_public: bool,
    ) -> None:
        ...

    def delete_document_chunks(self, doc_id: str) -> None:
        ...

    def reset_store(self) -> None:
        ...

    def hybrid_search(
        self,
        expanded_query: str,
        query_vector: List[float],
        effective_acl_tags: List[str],
        top_k: int,
    ) -> HybridSearchOutcome:
        ...


class AnswerSynthesizerPort(Protocol):
    """Port for synthesizing grounded answers from authorized chunks and RBAC audit metadata."""

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


def chunk_text(text: str, chunk_size: int = 650, overlap: int = 100) -> List[str]:
    """Split page text into overlapping semantic paragraphs/chunks (pure domain logic)."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_pos = normalized.rfind(". ", start, end)
            if split_pos == -1 or split_pos <= start + chunk_size // 2:
                split_pos = normalized.rfind(" ", start, end)
            if split_pos != -1 and split_pos > start:
                end = split_pos + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks
