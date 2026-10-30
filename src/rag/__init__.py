"""BeyondRAG: RBAC + Multilingual Retrieval-Augmented Generation."""

from src.rag.embeddings import (
    EMBEDDING_DIMS,
    embed_text,
    has_cjk,
)
from src.rag.cross_lingual import MultilingualQueryAdapter
from src.rag.service import RBACKnowledgeBaseService, get_kb_service

__all__ = [
    "EMBEDDING_DIMS",
    "embed_text",
    "has_cjk",
    "MultilingualQueryAdapter",
    "RBACKnowledgeBaseService",
    "get_kb_service",
]
