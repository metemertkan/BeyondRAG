"""BeyondRAG: RBAC + Cross-Lingual (English <-> Chinese) Retrieval-Augmented Generation."""

from src.rag.embeddings import (
    EMBEDDING_DIMS,
    embed_text,
    expand_cross_lingual_query,
    has_cjk,
)
from src.rag.service import RBACKnowledgeBaseService, get_kb_service

__all__ = [
    "EMBEDDING_DIMS",
    "embed_text",
    "expand_cross_lingual_query",
    "has_cjk",
    "RBACKnowledgeBaseService",
    "get_kb_service",
]
