"""Domain models, configurations, and state machines for BeyondRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IngestionState(str, Enum):
    """Lifecycle state of an ingested document."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


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
