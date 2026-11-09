"""Small-to-big retrieval: search on small child chunks, read their large parent.

Small chunks embed sharply but lack the surrounding context an LLM needs. Large chunks have
the context, but their embeddings are diluted. So only children are embedded and searched,
and each child stores its parent's text. Keeping the parent on the child means the ACL,
versioning and GDPR deletion that already apply to chunks cover it with no extra store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Sequence

from src.rag.chunker import ChunkingUnit, create_chunker

DEFAULT_PARENT_TOKENS: Final[int] = int(os.getenv("RAG_PARENT_TOKENS", "1000"))
DEFAULT_CHILD_TOKENS: Final[int] = int(os.getenv("RAG_CHILD_TOKENS", "200"))
DEFAULT_CHILD_OVERLAP: Final[int] = int(os.getenv("RAG_CHILD_OVERLAP_TOKENS", "30"))


@dataclass(frozen=True)
class ChildChunk:
    parent_index: int
    parent_text: str
    text: str


class ParentChildSplitter:
    def __init__(
        self,
        parent_tokens: int = DEFAULT_PARENT_TOKENS,
        child_tokens: int = DEFAULT_CHILD_TOKENS,
        child_overlap: int = DEFAULT_CHILD_OVERLAP,
    ) -> None:
        if child_tokens >= parent_tokens:
            raise ValueError(f"child_tokens ({child_tokens}) must be smaller than parent_tokens ({parent_tokens})")
        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens
        self._parents = create_chunker("paragraph", max_tokens=parent_tokens, overlap_tokens=0, unit=ChunkingUnit.TOKENS)
        self._children = create_chunker("sentence", max_tokens=child_tokens, overlap_tokens=child_overlap, unit=ChunkingUnit.TOKENS)

    def split(self, text: str) -> List[ChildChunk]:
        children: List[ChildChunk] = []
        for index, parent in enumerate(self._parents.split(text)):
            children += [ChildChunk(index, parent, child) for child in self._children.split(parent)]
        return children


def parent_child_enabled() -> bool:
    return os.getenv("RAG_PARENT_CHILD", "false").strip().lower() == "true"


def expand_to_parents(ranked: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """Swap each child for its parent, keeping the best rank when several children of one parent hit.

    Returns the expanded chunks and how many duplicate children were merged. Chunks without a
    parent (tables, documents ingested without parent-child) pass through unchanged.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    expanded: List[Dict[str, Any]] = []
    merged = 0
    for chunk in ranked:
        parent_id = chunk.get("parent_id")
        parent_text = chunk.get("parent_content")
        if not parent_id or not parent_text:
            expanded.append(chunk)
            continue
        if parent_id in seen:
            seen[parent_id]["matched_children"].append(chunk["content"])
            merged += 1
            continue
        parent = {**chunk, "content": parent_text, "matched_children": [chunk["content"]]}
        parent.pop("parent_content", None)
        seen[parent_id] = parent
        expanded.append(parent)
    return expanded, merged
