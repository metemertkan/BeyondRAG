"""Component 4: Robust, Idempotent Document & Chunk Persistence Repository.

Features:
  - Idempotent upserts with deterministic document & chunk IDs
  - Staged lifecycle tracking (PENDING -> PROCESSING -> INDEXED / SUPERSEDED / FAILED)
  - Durable JSON document registry under workspace/ (survives process restart)
  - Soft supersession: prior versions retained, marked non-current for retrieval
  - Bulk indexing in Elasticsearch + memory fallback
  - Full cleanup and rollback on failure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence

from elasticsearch.helpers import bulk

from src.config import WORKSPACE_DIR
from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.models import IngestionReport, IngestionState

logger = logging.getLogger(__name__)

UPLOADS_DIR: Final[Path] = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_REGISTRY_PATH: Final[Path] = WORKSPACE_DIR / "document_registry.json"


class DocumentRepository:
    """Handles staged, idempotent document and chunk persistence + durable registry."""

    def __init__(
        self,
        es_manager: ElasticsearchIndexManager,
        uploads_dir: Path = UPLOADS_DIR,
        registry_path: Path | None = None,
        *,
        persist_registry: bool | None = None,
    ) -> None:
        self.es_manager = es_manager
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        # Durable JSON registry under workspace/ when enabled. Unit tests that omit
        # registry_path stay in-process-isolated; production wiring passes the path.
        if registry_path is None:
            self.registry_path = DEFAULT_REGISTRY_PATH
            self._persist_enabled = bool(persist_registry) if persist_registry is not None else False
        else:
            self.registry_path = registry_path
            self._persist_enabled = True if persist_registry is None else bool(persist_registry)
        self._doc_states: Dict[str, IngestionReport] = {}
        self._current_by_source_key: Dict[str, str] = {}
        self._index_embedding_manifest: Optional[str] = None
        if self._persist_enabled:
            self._load_registry()

    # --- Durable registry -------------------------------------------------

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load document registry %s: %s", self.registry_path, exc)
            return
        docs = raw.get("documents") or {}
        for doc_id, payload in docs.items():
            if isinstance(payload, dict):
                self._doc_states[str(doc_id)] = IngestionReport.from_dict(payload)
        self._current_by_source_key = {
            str(k): str(v) for k, v in (raw.get("current_by_source_key") or {}).items()
        }
        manifest = raw.get("index_embedding_manifest")
        self._index_embedding_manifest = str(manifest) if manifest else None

    def _persist_registry(self) -> None:
        if not self._persist_enabled:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": {doc_id: report.to_dict() for doc_id, report in self._doc_states.items()},
            "current_by_source_key": dict(self._current_by_source_key),
            "index_embedding_manifest": self._index_embedding_manifest,
        }
        tmp_path = self.registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def get_document_state(self, doc_id: str) -> Optional[IngestionReport]:
        return self._doc_states.get(doc_id)

    def set_document_state(self, report: IngestionReport) -> None:
        self._doc_states[report.doc_id] = report
        if report.status == IngestionState.INDEXED and report.is_current and report.source_key:
            self._current_by_source_key[report.source_key] = report.doc_id
        self._persist_registry()

    def get_current_doc_id(self, source_key: str) -> Optional[str]:
        return self._current_by_source_key.get(source_key)

    def mark_superseded(self, doc_id: str) -> None:
        """Soft-supersede: retain bytes/chunks, exclude from retrieval."""
        existing = self._doc_states.get(doc_id)
        if existing is None:
            return
        superseded = IngestionReport(
            doc_id=existing.doc_id,
            filename=existing.filename,
            status=IngestionState.SUPERSEDED,
            page_count=existing.page_count,
            chunk_count=existing.chunk_count,
            is_public=existing.is_public,
            tags=list(existing.tags),
            acl_tags=list(existing.acl_tags),
            uploaded_at=existing.uploaded_at,
            dynamic_concepts=list(existing.dynamic_concepts),
            concept_model=existing.concept_model,
            error_message=existing.error_message,
            source_key=existing.source_key,
            content_hash=existing.content_hash,
            doc_version=existing.doc_version,
            tenant=existing.tenant,
            is_current=False,
            supersedes_doc_id=existing.supersedes_doc_id,
            embedding_manifest=existing.embedding_manifest,
            parser_name=existing.parser_name,
        )
        self._doc_states[doc_id] = superseded
        self._set_chunks_current(doc_id, is_current=False, status=IngestionState.SUPERSEDED.value)
        self._persist_registry()

    def get_index_embedding_manifest(self) -> Optional[str]:
        return self._index_embedding_manifest

    def set_index_embedding_manifest(self, manifest_id: str) -> None:
        self._index_embedding_manifest = manifest_id
        self._persist_registry()

    def clear_index_embedding_manifest(self) -> None:
        self._index_embedding_manifest = None
        self._persist_registry()

    def list_registry_documents(self, *, current_only: bool = False) -> List[IngestionReport]:
        reports = list(self._doc_states.values())
        if current_only:
            reports = [r for r in reports if r.is_current and r.status == IngestionState.INDEXED]
        return sorted(reports, key=lambda r: r.uploaded_at, reverse=True)

    def next_doc_version(self, source_key: str) -> int:
        versions = [
            r.doc_version
            for r in self._doc_states.values()
            if r.source_key == source_key
        ]
        return (max(versions) + 1) if versions else 1

    # --- File + chunk persistence ----------------------------------------

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

    def _set_chunks_current(self, doc_id: str, *, is_current: bool, status: str) -> None:
        if self.es_manager.is_online():
            self.es_manager.ensure_index()
            try:
                self.es_manager.client.update_by_query(
                    index=self.es_manager.index_name,
                    query={"term": {"doc_id": doc_id}},
                    script={
                        "source": (
                            "ctx._source.is_current = params.is_current; "
                            "ctx._source.ingestion_status = params.status;"
                        ),
                        "lang": "painless",
                        "params": {"is_current": is_current, "status": status},
                    },
                    refresh=True,
                )
            except Exception as exc:
                logger.warning("ES supersede update_by_query failed for %s: %s", doc_id, exc)
        for rec in self.es_manager.memory_chunks:
            if rec.get("doc_id") == doc_id:
                rec["is_current"] = is_current
                rec["ingestion_status"] = status

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
            existing_ids = {c["chunk_id"] for c in chunks}
            self.es_manager.memory_chunks = [
                c for c in self.es_manager.memory_chunks if c.get("chunk_id") not in existing_ids
            ]
            self.es_manager.memory_chunks.extend(chunks)

    def list_documents(self, *, include_superseded: bool = False) -> List[Dict[str, Any]]:
        """Return documents from the durable registry (preferred) or indexed chunks."""
        if self._doc_states:
            if include_superseded:
                reports = self.list_registry_documents(current_only=False)
            else:
                # Default UI list: current INDEXED (+ FAILED for quarantine visibility)
                reports = [
                    r
                    for r in self.list_registry_documents(current_only=False)
                    if (r.is_current and r.status == IngestionState.INDEXED)
                    or r.status == IngestionState.FAILED
                ]
            return [r.to_dict() for r in reports]

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
                    "is_current",
                    "source_key",
                    "ingestion_status",
                ],
                query={"match_all": {}},
            )
            hits = [h["_source"] for h in resp["hits"]["hits"]]
        else:
            hits = list(self.es_manager.memory_chunks)

        docs_by_id: Dict[str, Dict[str, Any]] = {}
        for item in hits:
            if not include_superseded and item.get("is_current") is False:
                continue
            if not include_superseded and item.get("ingestion_status") == IngestionState.SUPERSEDED.value:
                continue
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
                    "source_key": item.get("source_key", ""),
                    "is_current": item.get("is_current", True),
                    "status": item.get("ingestion_status", IngestionState.INDEXED.value),
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

        existing = self._doc_states.get(doc_id)
        if existing:
            self.set_document_state(
                IngestionReport(
                    doc_id=existing.doc_id,
                    filename=existing.filename,
                    status=existing.status,
                    page_count=existing.page_count,
                    chunk_count=existing.chunk_count,
                    is_public=is_public,
                    tags=extract_visible_tags(acl_tags),
                    acl_tags=acl_tags,
                    uploaded_at=existing.uploaded_at,
                    dynamic_concepts=list(existing.dynamic_concepts),
                    concept_model=existing.concept_model,
                    error_message=existing.error_message,
                    source_key=existing.source_key,
                    content_hash=existing.content_hash,
                    doc_version=existing.doc_version,
                    tenant=existing.tenant,
                    is_current=existing.is_current,
                    supersedes_doc_id=existing.supersedes_doc_id,
                    embedding_manifest=existing.embedding_manifest,
                    parser_name=existing.parser_name,
                )
            )

    def delete_chunks(self, doc_id: str) -> None:
        """Remove a document's indexed chunks without deleting the uploaded file."""
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
        """Remove a document's indexed chunks and its saved upload file."""
        report = self._doc_states.get(doc_id)
        self.delete_chunks(doc_id)
        self.delete_upload_file(doc_id)
        self._doc_states.pop(doc_id, None)
        if report and report.source_key and self._current_by_source_key.get(report.source_key) == doc_id:
            self._current_by_source_key.pop(report.source_key, None)
        self._persist_registry()
        return {"deleted": True, "doc_id": doc_id}
