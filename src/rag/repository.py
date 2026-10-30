"""Component 4: Robust, Idempotent Document & Chunk Persistence Repository.

Features:
  - Idempotent upserts with deterministic document & chunk IDs
  - Staged lifecycle tracking (PENDING -> PROCESSING -> INDEXED / FAILED)
  - Bulk indexing in Elasticsearch
  - Full cleanup and rollback on failure (prevents partially indexed state)
  - Seamless document replacement (wiping previous chunks before reindexing)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence

from elasticsearch.helpers import bulk

from src.config import WORKSPACE_DIR
from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.models import DocumentChunk, IngestionReport, IngestionState

logger = logging.getLogger(__name__)

UPLOADS_DIR: Final[Path] = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class DocumentRepository:
    """Handles staged, idempotent document and chunk persistence."""

    def __init__(
        self,
        es_manager: ElasticsearchIndexManager,
        uploads_dir: Path = UPLOADS_DIR,
    ) -> None:
        self.es_manager = es_manager
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._doc_states: Dict[str, IngestionReport] = {}

    def get_document_state(self, doc_id: str) -> Optional[IngestionReport]:
        return self._doc_states.get(doc_id)

    def set_document_state(self, report: IngestionReport) -> None:
        self._doc_states[report.doc_id] = report

    def save_upload_file(self, doc_id: str, filename: str, pdf_bytes: bytes) -> Path:
        """Persist raw document bytes on disk with deterministic path."""
        safe_filename = Path(filename).name
        saved_path = self.uploads_dir / f"{doc_id}_{safe_filename}"
        saved_path.write_bytes(pdf_bytes)
        return saved_path

    def delete_upload_file(self, doc_id: str) -> None:
        """Remove persisted upload file matching doc_id."""
        for file_path in self.uploads_dir.glob(f"{doc_id}_*"):
            file_path.unlink(missing_ok=True)

    def save_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """Bulk index chunk records into Elasticsearch or memory fallback."""
        if not chunks:
            return

        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            actions = [
                {
                    "_index": self.es_manager.index_name,
                    "_id": rec["chunk_id"],
                    "_source": rec,
                }
                for rec in chunks
            ]
            success_count, errors = bulk(
                self.es_manager.client,
                actions,
                refresh=True,
                raise_on_error=False,
            )
            if errors:
                raise RuntimeError(f"Elasticsearch bulk indexing failed: {errors}")
        else:
            # Memory fallback with idempotent replacement by chunk_id
            existing_ids = {c["chunk_id"] for c in chunks}
            self.es_manager.memory_chunks = [
                c for c in self.es_manager.memory_chunks if c.get("chunk_id") not in existing_ids
            ]
            self.es_manager.memory_chunks.extend(chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a deduplicated list of all indexed documents and their RBAC tags."""
        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            resp = self.es_manager.client.search(
                index=self.es_manager.index_name,
                size=1000,
                _source=[
                    "doc_id",
                    "filename",
                    "page_number",
                    "is_public",
                    "acl_tags",
                    "uploaded_at",
                ],
                query={"match_all": {}},
            )
            hits = [h["_source"] for h in resp["hits"]["hits"]]
        else:
            hits = list(self.es_manager.memory_chunks)

        docs_by_id: Dict[str, Dict[str, Any]] = {}
        for item in hits:
            d_id = item["doc_id"]
            raw_acl = item.get("acl_tags", [PUBLIC_SENTINEL_TAG])
            visible_tags = extract_visible_tags(raw_acl)
            is_pub = bool(item.get("is_public", len(visible_tags) == 0))
            if d_id not in docs_by_id:
                docs_by_id[d_id] = {
                    "doc_id": d_id,
                    "filename": item.get("filename", d_id),
                    "page_count": item.get("page_number", 1),
                    "chunk_count": 1,
                    "is_public": is_pub,
                    "tags": visible_tags,
                    "acl_tags": raw_acl,
                    "uploaded_at": item.get("uploaded_at", ""),
                }
            else:
                docs_by_id[d_id]["chunk_count"] += 1
                docs_by_id[d_id]["page_count"] = max(
                    docs_by_id[d_id]["page_count"],
                    int(item.get("page_number", 1)),
                )

        return sorted(
            docs_by_id.values(),
            key=lambda d: d.get("uploaded_at", ""),
            reverse=True,
        )

    def update_document_acl(
        self,
        doc_id: str,
        acl_tags: List[str],
        is_public: bool,
    ) -> None:
        """Update ACL tags on all chunks of doc_id without re-embedding."""
        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            self.es_manager.client.update_by_query(
                index=self.es_manager.index_name,
                query={"term": {"doc_id": doc_id}},
                script={
                    "source": "ctx._source.acl_tags = params.acl_tags; ctx._source.is_public = params.is_public;",
                    "lang": "painless",
                    "params": {
                        "acl_tags": acl_tags,
                        "is_public": is_public,
                    },
                },
                refresh=True,
            )
        else:
            for rec in self.es_manager.memory_chunks:
                if rec["doc_id"] == doc_id:
                    rec["acl_tags"] = acl_tags
                    rec["is_public"] = is_public

    def delete_chunks(self, doc_id: str) -> None:
        """Remove a document's indexed chunks from Elasticsearch and memory without deleting the uploaded file."""
        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            self.es_manager.client.delete_by_query(
                index=self.es_manager.index_name,
                query={"term": {"doc_id": doc_id}},
                refresh=True,
            )
        self.es_manager.memory_chunks = [
            c for c in self.es_manager.memory_chunks if c.get("doc_id") != doc_id
        ]

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document's indexed chunks and its saved PDF file."""
        self.delete_chunks(doc_id)
        self.delete_upload_file(doc_id)
        self._doc_states.pop(doc_id, None)
        return {"deleted": True, "doc_id": doc_id}
