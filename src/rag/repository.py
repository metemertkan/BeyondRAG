"""Component 4: Document & Chunk Persistence Repository.

Manages persisting raw PDF uploads on disk (`workspace/uploads`), indexing chunk
documents into Elasticsearch (or in-memory fallback), listing deduplicated
documents, updating ACL tags in-place (`_update_by_query`), and deleting documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final, List, Sequence

from src.config import WORKSPACE_DIR
from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.es_client import ElasticsearchIndexManager

UPLOADS_DIR: Final[Path] = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class DocumentRepository:
    """Handles document and chunk persistence across Elasticsearch and local file storage."""

    def __init__(
        self,
        es_manager: ElasticsearchIndexManager,
        uploads_dir: Path = UPLOADS_DIR,
    ) -> None:
        self.es_manager = es_manager
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def save_upload_file(self, doc_id: str, filename: str, pdf_bytes: bytes) -> Path:
        """Persist raw PDF bytes in the workspace uploads directory."""
        safe_filename = Path(filename).name
        saved_path = self.uploads_dir / f"{doc_id}_{safe_filename}"
        saved_path.write_bytes(pdf_bytes)
        return saved_path

    def delete_upload_file(self, doc_id: str) -> None:
        """Remove persisted PDF upload files matching `doc_id`."""
        for file_path in self.uploads_dir.glob(f"{doc_id}_*"):
            file_path.unlink(missing_ok=True)

    def save_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """Index chunk records into Elasticsearch (or in-memory fallback)."""
        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            for rec in chunks:
                self.es_manager.client.index(
                    index=self.es_manager.index_name,
                    id=rec["chunk_id"],
                    document=rec,
                )
            self.es_manager.client.indices.refresh(index=self.es_manager.index_name)
        else:
            self.es_manager.memory_chunks.extend(chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a deduplicated list of all indexed PDF documents and their RBAC tags."""
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
        """Update ACL tags on all chunks of `doc_id` without re-embedding."""
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

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document's indexed chunks and its saved PDF file."""
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
        self.delete_upload_file(doc_id)
        return {"deleted": True, "doc_id": doc_id}
