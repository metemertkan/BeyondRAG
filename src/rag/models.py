"""Domain models, configurations, and state machines for BeyondRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IngestionState(str, Enum):
    """Lifecycle state of an ingested document.

    PENDING is recorded as soon as the upload is accepted (before parse/embed).
    PROCESSING covers extract → embed → index work.
    INDEXED is the searchable current version.
    SUPERSEDED retains prior bytes/chunks but excludes them from retrieval.
    FAILED is a terminal quarantine state after rollback.
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ParsedBlock:
    """Structure-preserving unit emitted by a document parser."""

    block_type: str
    text: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """Full parse result: page texts for legacy paths plus structured blocks."""

    page_count: int
    pages: List[tuple[int, str]]
    blocks: List[ParsedBlock] = field(default_factory=list)
    parser_name: str = ""


@dataclass(frozen=True)
class DocumentChunk:
    """Strongly typed domain representation of an ingested document chunk."""

    chunk_id: str
    doc_id: str
    filename: str
    title: str
    page_number: int
    chunk_index: int
    content: str
    is_public: bool
    acl_tags: List[str]
    uploaded_at: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Concept:
    """Domain concept representation for cross-lingual and domain retrieval."""

    concept_id: str
    triggers: List[str]
    expansion: str
    source_language: str = "en"
    target_language: Optional[str] = None


@dataclass(frozen=True)
class IngestionReport:
    """Status report for document ingestion lifecycle tracking."""

    doc_id: str
    filename: str
    status: IngestionState
    page_count: int
    chunk_count: int
    is_public: bool
    tags: List[str]
    acl_tags: List[str]
    uploaded_at: str
    dynamic_concepts: List[str] = field(default_factory=list)
    concept_model: str = ""
    error_message: Optional[str] = None
    source_key: str = ""
    content_hash: str = ""
    doc_version: int = 1
    tenant: str = "default"
    is_current: bool = True
    supersedes_doc_id: Optional[str] = None
    embedding_manifest: str = ""
    parser_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "status": self.status.value if isinstance(self.status, IngestionState) else str(self.status),
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "is_public": self.is_public,
            "tags": list(self.tags),
            "acl_tags": list(self.acl_tags),
            "uploaded_at": self.uploaded_at,
            "dynamic_concepts": list(self.dynamic_concepts),
            "concept_model": self.concept_model,
            "error_message": self.error_message,
            "source_key": self.source_key,
            "content_hash": self.content_hash,
            "doc_version": self.doc_version,
            "tenant": self.tenant,
            "is_current": self.is_current,
            "supersedes_doc_id": self.supersedes_doc_id,
            "embedding_manifest": self.embedding_manifest,
            "parser_name": self.parser_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionReport":
        status_raw = data.get("status", IngestionState.FAILED.value)
        try:
            status = IngestionState(status_raw)
        except ValueError:
            status = IngestionState.FAILED
        return cls(
            doc_id=str(data.get("doc_id", "")),
            filename=str(data.get("filename", "")),
            status=status,
            page_count=int(data.get("page_count", 0) or 0),
            chunk_count=int(data.get("chunk_count", 0) or 0),
            is_public=bool(data.get("is_public", False)),
            tags=list(data.get("tags") or []),
            acl_tags=list(data.get("acl_tags") or []),
            uploaded_at=str(data.get("uploaded_at", "")),
            dynamic_concepts=list(data.get("dynamic_concepts") or []),
            concept_model=str(data.get("concept_model") or ""),
            error_message=data.get("error_message"),
            source_key=str(data.get("source_key") or ""),
            content_hash=str(data.get("content_hash") or ""),
            doc_version=int(data.get("doc_version", 1) or 1),
            tenant=str(data.get("tenant") or "default"),
            is_current=bool(data.get("is_current", True)),
            supersedes_doc_id=data.get("supersedes_doc_id"),
            embedding_manifest=str(data.get("embedding_manifest") or ""),
            parser_name=str(data.get("parser_name") or ""),
        )
