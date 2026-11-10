"""Component 4: Robust, Idempotent Document & Chunk Persistence Repository.

Features:
  - Idempotent upserts with deterministic document & chunk IDs
  - Staged lifecycle tracking (PENDING -> PROCESSING -> INDEXED / SUPERSEDED / FAILED)
  - Durable JSON document registry under workspace/ (survives process restart)
  - Soft supersession: prior versions retained, marked non-current for retrieval
  - Bulk indexing in Elasticsearch + memory fallback
  - Full cleanup and rollback on failure
  - Lineage queries, hard deletion across stores, and erasure tombstones
  - Tenant scope on every operation

Tenant rules in this module:

  - A logical document is identified by (tenant, source_key), never by source_key alone.
    Two tenants may both use the key ``handbook`` and neither can supersede, archive or
    erase the other's document.
  - Every Elasticsearch call targets the tenant's own index and carries a tenant filter,
    so a doc_id guessed from another tenant matches nothing.
  - Calls made for a user (list, retag, delete) pass the caller's tenant. A document that
    belongs to someone else raises ``TenantNotFoundError``, the same as a missing one.
  - Internal calls by doc_id (supersede, rollback) omit the tenant and act on the tenant
    the registry recorded for that document.
  - Operator calls without a tenant (erasure, archive) act on the ``default`` tenant only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Final, Iterable, List, Optional, Sequence, Tuple

from elasticsearch.helpers import bulk

from src.config import WORKSPACE_DIR
from src.rag.acl import PUBLIC_SENTINEL_TAG, extract_visible_tags
from src.rag.erasure import ErasedSourceError, hash_with_key, load_erasure_key
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.governance import normalize_subject_id
from src.rag.models import IngestionReport, IngestionState, ParsedTable
from src.rag.tabular_store import TabularStore
from src.rag.tenancy import DEFAULT_TENANT, TenantNotFoundError, normalize_tenant, require_tenant, tenant_scoped

logger = logging.getLogger(__name__)

UPLOADS_DIR: Final[Path] = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_REGISTRY_PATH: Final[Path] = WORKSPACE_DIR / "document_registry.json"
DEFAULT_TABULAR_PATH: Final[Path] = WORKSPACE_DIR / "tabular_store.sqlite"
DEFAULT_ERASURE_KEY_PATH: Final[Path] = WORKSPACE_DIR / ".erasure_key"
MAX_PROCESSED_EVENTS: Final[int] = 5000


class DocumentRepository:
    """Handles staged, idempotent document and chunk persistence + durable registry."""

    def __init__(
        self,
        es_manager: ElasticsearchIndexManager,
        uploads_dir: Path = UPLOADS_DIR,
        registry_path: Path | None = None,
        *,
        persist_registry: bool | None = None,
        tabular_store: TabularStore | None = None,
        erasure_key_path: Path | None = None,
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
        # (tenant, source_key) -> current doc_id.
        self._current_by_source_key: Dict[Tuple[str, str], str] = {}
        self._index_embedding_manifest: Optional[str] = None
        self.tabular_store = tabular_store or TabularStore()
        self._erasure_key_path = erasure_key_path or self.registry_path.parent / DEFAULT_ERASURE_KEY_PATH.name
        self._erasure_key: Optional[bytes] = None
        # Tombstones hold keyed hashes only and outlive the data they refer to.
        self._tombstones: Dict[str, str] = {}
        self._processed_events: List[str] = []
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
        # "current_by_source_key" is the pre-tenant format. Entries there take their tenant
        # from the document they point at, so an older registry loads into the right scope.
        self._current_by_source_key = {}
        for key, doc_id in (raw.get("current_by_source_key") or {}).items():
            report = self._doc_states.get(str(doc_id))
            scope = normalize_tenant(report.tenant) if report else DEFAULT_TENANT
            self._current_by_source_key[(scope, str(key))] = str(doc_id)
        for scope, entries in (raw.get("current_by_tenant") or {}).items():
            for key, doc_id in (entries or {}).items():
                self._current_by_source_key[(normalize_tenant(scope), str(key))] = str(doc_id)
        manifest = raw.get("index_embedding_manifest")
        self._index_embedding_manifest = str(manifest) if manifest else None
        self._tombstones = {str(k): str(v) for k, v in (raw.get("tombstones") or {}).items()}
        self._processed_events = [str(e) for e in (raw.get("processed_events") or [])]

    def _persist_registry(self) -> None:
        if not self._persist_enabled:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": {doc_id: report.to_dict() for doc_id, report in self._doc_states.items()},
            "current_by_source_key": {
                key: doc_id for (scope, key), doc_id in self._current_by_source_key.items() if scope == DEFAULT_TENANT
            },
            "current_by_tenant": self._current_by_tenant(),
            "index_embedding_manifest": self._index_embedding_manifest,
            "tombstones": dict(self._tombstones),
            "processed_events": list(self._processed_events),
        }
        tmp_path = self.registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def _current_by_tenant(self) -> Dict[str, Dict[str, str]]:
        nested: Dict[str, Dict[str, str]] = {}
        for (scope, key), doc_id in self._current_by_source_key.items():
            if scope != DEFAULT_TENANT:
                nested.setdefault(scope, {})[key] = doc_id
        return nested

    # --- Tenant scope ------------------------------------------------------

    def tenant_of(self, doc_id: str) -> Optional[str]:
        """The tenant the registry recorded for a document, or None when it is unknown."""
        report = self._doc_states.get(doc_id)
        return normalize_tenant(report.tenant) if report else None

    def _scope_for_doc(self, doc_id: str, tenant: str | None) -> str:
        """Resolve the tenant an operation on ``doc_id`` runs in.

        With a tenant: it is the caller's, and the document must belong to it. Without
        one: an internal call, scoped to the document's own tenant.
        """
        owner = self.tenant_of(doc_id)
        if tenant is None:
            return owner or DEFAULT_TENANT
        scope = require_tenant(tenant)
        if owner is not None and owner != scope:
            raise TenantNotFoundError(f"Document not found: {doc_id}")
        return scope

    @staticmethod
    def _doc_query(doc_id: str, tenant: str) -> Dict[str, Any]:
        return {"bool": {"filter": [{"term": {"doc_id": doc_id}}, {"term": {"tenant": tenant}}]}}

    @staticmethod
    def _in_scope(rec: Dict[str, Any], tenant: str) -> bool:
        return normalize_tenant(rec.get("tenant")) == tenant

    def get_document_state(self, doc_id: str) -> Optional[IngestionReport]:
        return self._doc_states.get(doc_id)

    def set_document_state(self, report: IngestionReport) -> None:
        self._doc_states[report.doc_id] = report
        if report.status == IngestionState.INDEXED and report.is_current and report.source_key:
            self._current_by_source_key[(normalize_tenant(report.tenant), report.source_key)] = report.doc_id
        self._persist_registry()

    def get_current_doc_id(self, source_key: str, tenant: str | None = None) -> Optional[str]:
        return self._current_by_source_key.get((normalize_tenant(tenant), source_key))

    def mark_superseded(self, doc_id: str) -> None:
        """Soft-supersede: retain bytes/chunks, exclude from retrieval."""
        existing = self._doc_states.get(doc_id)
        if existing is None:
            return
        superseded = replace(existing, status=IngestionState.SUPERSEDED, is_current=False)
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

    def list_registry_documents(self, *, current_only: bool = False, tenant: str | None = None) -> List[IngestionReport]:
        """Registry entries, newest first. ``tenant=None`` is the operator view across tenants."""
        reports = list(self._doc_states.values())
        if tenant is not None:
            scope = require_tenant(tenant)
            reports = [r for r in reports if normalize_tenant(r.tenant) == scope]
        if current_only:
            reports = [r for r in reports if r.is_current and r.status == IngestionState.INDEXED]
        return sorted(reports, key=lambda r: r.uploaded_at, reverse=True)

    def next_doc_version(self, source_key: str, tenant: str | None = None) -> int:
        scope = normalize_tenant(tenant)
        versions = [
            r.doc_version
            for r in self._doc_states.values()
            if r.source_key == source_key and normalize_tenant(r.tenant) == scope
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

    def _set_chunks_current(self, doc_id: str, *, is_current: bool, status: str, tenant: str | None = None) -> None:
        scope = self._scope_for_doc(doc_id, tenant)
        if self.es_manager.is_online():
            try:
                self.es_manager.client.update_by_query(
                    index=self.es_manager.ensure_index(scope),
                    query=self._doc_query(doc_id, scope),
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
            if rec.get("doc_id") == doc_id and self._in_scope(rec, scope):
                rec["is_current"] = is_current
                rec["ingestion_status"] = status

    def save_chunks(self, chunks: Sequence[Dict[str, Any]]) -> None:
        """Bulk index chunk records into Elasticsearch or memory fallback.

        Every stored chunk carries a tenant. One without it is stamped ``default``, because
        a chunk with no tenant would match no tenant filter and be unreachable.
        """
        if not chunks:
            return
        chunks = [{**rec, "tenant": normalize_tenant(rec.get("tenant"))} for rec in chunks]

        if self.es_manager.is_online():
            index_by_tenant = {scope: self.es_manager.ensure_index(scope) for scope in {rec["tenant"] for rec in chunks}}
            actions = [
                {
                    "_index": index_by_tenant[rec["tenant"]],
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

    def list_documents(self, *, tenant: str, include_superseded: bool = True) -> List[Dict[str, Any]]:
        """Return one tenant's documents from the durable registry (preferred) or indexed chunks.

        Default includes SUPERSEDED so the dashboard can show retained prior versions
        (they remain excluded from retrieval via ``is_current`` / status filters).
        """
        scope = require_tenant(tenant)
        if self._doc_states:
            if include_superseded:
                reports = [
                    r
                    for r in self.list_registry_documents(current_only=False, tenant=scope)
                    if r.status
                    in {
                        IngestionState.INDEXED,
                        IngestionState.SUPERSEDED,
                        IngestionState.ARCHIVED,
                        IngestionState.FAILED,
                        IngestionState.PENDING,
                        IngestionState.PROCESSING,
                    }
                ]
            else:
                reports = [
                    r
                    for r in self.list_registry_documents(current_only=False, tenant=scope)
                    if (r.is_current and r.status == IngestionState.INDEXED)
                    or r.status == IngestionState.FAILED
                ]
            return [r.to_dict() for r in reports]

        if self.es_manager.is_online():
            resp = self.es_manager.client.search(
                index=self.es_manager.ensure_index(scope),
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
                query={"term": {"tenant": scope}},
            )
            hits = [h["_source"] for h in resp["hits"]["hits"]]
        else:
            hits = [rec for rec in self.es_manager.memory_chunks if self._in_scope(rec, scope)]

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
        *,
        tenant: str | None = None,
    ) -> None:
        """Update ACL tags on all chunks of doc_id without re-embedding."""
        scope = self._scope_for_doc(doc_id, tenant)
        if self.es_manager.is_online():
            self.es_manager.client.update_by_query(
                index=self.es_manager.ensure_index(scope),
                query=self._doc_query(doc_id, scope),
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
                if rec["doc_id"] == doc_id and self._in_scope(rec, scope):
                    rec["acl_tags"] = acl_tags
                    rec["is_public"] = is_public

        existing = self._doc_states.get(doc_id)
        if existing:
            self.set_document_state(
                replace(existing, is_public=is_public, tags=extract_visible_tags(acl_tags), acl_tags=acl_tags)
            )
        self.tabular_store.update_acl(doc_id, acl_tags, tenant=scope)

    def publish_new_current(
        self,
        report: IngestionReport,
        previous_doc_id: Optional[str] = None,
    ) -> None:
        """Atomically publish ``report`` as the sole current INDEXED doc for its source_key.

        Stage-then-publish: caller indexes the new version's chunks first (typically
        with ``is_current=False``), then calls this to flip currentness. Previous
        current (if any) is soft-superseded in the same step so observers never see
        zero currents and never see two currents for the same source_key.
        """
        if report.status != IngestionState.INDEXED or not report.is_current:
            raise ValueError("publish_new_current requires an INDEXED is_current report")

        prev_id = previous_doc_id
        if prev_id and prev_id != report.doc_id:
            existing = self._doc_states.get(prev_id)
            if existing is not None:
                self._doc_states[prev_id] = replace(existing, status=IngestionState.SUPERSEDED, is_current=False)
                self._set_chunks_current(
                    prev_id, is_current=False, status=IngestionState.SUPERSEDED.value
                )

        self._doc_states[report.doc_id] = report
        if report.source_key:
            self._current_by_source_key[(normalize_tenant(report.tenant), report.source_key)] = report.doc_id
        self._set_chunks_current(
            report.doc_id, is_current=True, status=IngestionState.INDEXED.value
        )
        self._persist_registry()

    def delete_chunks(self, doc_id: str, *, tenant: str | None = None) -> None:
        """Remove a document's indexed chunks without deleting the uploaded file."""
        scope = self._scope_for_doc(doc_id, tenant)
        if self.es_manager.is_online():
            self.es_manager.client.delete_by_query(
                index=self.es_manager.ensure_index(scope),
                query=self._doc_query(doc_id, scope),
                refresh=True,
            )
        self.es_manager.memory_chunks = [
            c for c in self.es_manager.memory_chunks if not (c.get("doc_id") == doc_id and self._in_scope(c, scope))
        ]

    def delete_document(self, doc_id: str, *, tenant: str | None = None) -> Dict[str, Any]:
        """Remove a document's indexed chunks and its saved upload file."""
        scope = self._scope_for_doc(doc_id, tenant)
        report = self._doc_states.get(doc_id)
        self.delete_chunks(doc_id, tenant=scope)
        if report is not None or tenant is None:
            # The upload file is named by doc_id alone. Only remove it for a document this
            # tenant is known to own, so a guessed id cannot delete another tenant's file.
            self.delete_upload_file(doc_id)
        self.tabular_store.delete_tables(doc_id=doc_id, tenant=scope)
        self._doc_states.pop(doc_id, None)
        if report and report.source_key and self._current_by_source_key.get((scope, report.source_key)) == doc_id:
            self._current_by_source_key.pop((scope, report.source_key), None)
        self._persist_registry()
        return {"deleted": True, "doc_id": doc_id}

    def save_tables(
        self,
        tables: Sequence[ParsedTable],
        *,
        doc_id: str,
        source_key: str,
        acl_tags: Sequence[str],
        tenant: str | None = None,
    ) -> None:
        """Store large-table rows for structured queries, replacing any earlier copy."""
        scope = normalize_tenant(tenant)
        self.tabular_store.delete_tables(doc_id=doc_id, tenant=scope)
        for table in tables:
            self.tabular_store.save_table(table, doc_id=doc_id, source_key=source_key, acl_tags=acl_tags, tenant=scope)

    # --- Lineage and hard deletion ---------------------------------------
    # Every lineage operation runs inside one tenant. A subject id or source key is only
    # unique within a tenant: "employee-1042" at two companies is two different people.

    @staticmethod
    def _lineage_query(source_keys: Sequence[str], subject_id: Optional[str], tenant: str) -> Dict[str, Any]:
        should: List[Dict[str, Any]] = []
        if source_keys:
            should.append({"terms": {"source_key": list(source_keys)}})
        if subject_id:
            should.append({"term": {"subject_ids": subject_id}})
        return {"bool": {"should": should, "minimum_should_match": 1, "filter": [{"term": {"tenant": tenant}}]}}

    @classmethod
    def _memory_matches(cls, rec: Dict[str, Any], source_keys: Sequence[str], subject_id: Optional[str], tenant: str) -> bool:
        if not cls._in_scope(rec, tenant):
            return False
        return rec.get("source_key") in source_keys or bool(subject_id and subject_id in (rec.get("subject_ids") or []))

    def _tenant_reports(self, tenant: str) -> List[IngestionReport]:
        return [report for report in self._doc_states.values() if normalize_tenant(report.tenant) == tenant]

    def find_sources_by_subject(self, subject_id: str, *, tenant: str | None = None) -> Tuple[List[str], List[str]]:
        """Return (declared, detected) source keys linked to a person, within one tenant."""
        scope = normalize_tenant(tenant)
        declared: set[str] = set()
        detected: set[str] = set()
        for report in self._tenant_reports(scope):
            if subject_id in report.subject_ids:
                declared.add(report.source_key)
            elif subject_id in report.detected_subject_ids:
                detected.add(report.source_key)
        # Registry and index can drift, so also ask the index directly.
        for key in self._index_source_keys_for_subject(subject_id, scope):
            if key not in declared:
                detected.add(key)
        return sorted(declared), sorted(detected)

    def _index_source_keys_for_subject(self, subject_id: str, tenant: str) -> List[str]:
        if self.es_manager.is_online():
            resp = self.es_manager.client.search(
                index=self.es_manager.ensure_index(tenant),
                size=0,
                query={"bool": {"filter": [{"term": {"subject_ids": subject_id}}, {"term": {"tenant": tenant}}]}},
                aggs={"sources": {"terms": {"field": "source_key", "size": 1000}}},
            )
            return [bucket["key"] for bucket in resp["aggregations"]["sources"]["buckets"]]
        return sorted(
            {
                str(rec.get("source_key"))
                for rec in self.es_manager.memory_chunks
                if self._in_scope(rec, tenant) and subject_id in (rec.get("subject_ids") or []) and rec.get("source_key")
            }
        )

    def versions_for_sources(self, source_keys: Iterable[str], *, tenant: str | None = None) -> List[IngestionReport]:
        """Every registry entry for these sources, in any state, SUPERSEDED included."""
        wanted = set(source_keys)
        return [report for report in self._tenant_reports(normalize_tenant(tenant)) if report.source_key in wanted]

    def count_chunks(
        self, *, source_keys: Sequence[str], subject_id: Optional[str] = None, tenant: str | None = None
    ) -> int:
        if not source_keys and not subject_id:
            return 0
        scope = normalize_tenant(tenant)
        if self.es_manager.is_online():
            index = self.es_manager.ensure_index(scope)
            self.es_manager.client.indices.refresh(index=index)
            return int(
                self.es_manager.client.count(index=index, query=self._lineage_query(source_keys, subject_id, scope))["count"]
            )
        return sum(1 for rec in self.es_manager.memory_chunks if self._memory_matches(rec, source_keys, subject_id, scope))

    def count_subject_mentions(
        self, subject_id: str, *, exclude_source_keys: Sequence[str] = (), tenant: str | None = None
    ) -> int:
        """Full-text scan for a subject appearing in chunks lineage did not capture."""
        needle = normalize_subject_id(subject_id)
        scope = normalize_tenant(tenant)
        if self.es_manager.is_online():
            query: Dict[str, Any] = {
                "bool": {"must": [{"match_phrase": {"content": needle}}], "filter": [{"term": {"tenant": scope}}]}
            }
            if exclude_source_keys:
                query["bool"]["must_not"] = [{"terms": {"source_key": list(exclude_source_keys)}}]
            return int(self.es_manager.client.count(index=self.es_manager.ensure_index(scope), query=query)["count"])
        return sum(
            1
            for rec in self.es_manager.memory_chunks
            if self._in_scope(rec, scope)
            and needle in str(rec.get("content", "")).lower()
            and rec.get("source_key") not in exclude_source_keys
        )

    def delete_chunks_by_lineage(
        self, *, source_keys: Sequence[str], subject_id: Optional[str] = None, tenant: str | None = None
    ) -> int:
        """Delete by lineage, not doc_id, so orphans missing from the registry go too."""
        scope = normalize_tenant(tenant)
        removed = 0
        if self.es_manager.is_online():
            resp = self.es_manager.client.delete_by_query(
                index=self.es_manager.ensure_index(scope),
                query=self._lineage_query(source_keys, subject_id, scope),
                refresh=True,
                conflicts="proceed",
            )
            removed += int(resp.get("deleted", 0))
        before = len(self.es_manager.memory_chunks)
        self.es_manager.memory_chunks = [
            rec for rec in self.es_manager.memory_chunks if not self._memory_matches(rec, source_keys, subject_id, scope)
        ]
        return removed + before - len(self.es_manager.memory_chunks)

    def expunge_deleted_documents(self, *, tenant: str | None = None) -> int:
        """Try to merge deleted documents out of Lucene segments; return how many remain.

        delete_by_query only hides documents. Their text stays on disk, and ES also holds
        it under a soft-delete retention lease until the lease syncs (about 30 seconds on
        an idle index), so a merge straight after the delete removes nothing. Callers must
        check the returned count rather than assume the merge worked.

        In the shared index the count covers every tenant's deleted documents, because
        Lucene segments are not split by tenant.
        """
        if not self.es_manager.is_online():
            return 0
        client, index = self.es_manager.client, self.es_manager.ensure_index(normalize_tenant(tenant))
        client.indices.flush(index=index)
        client.indices.forcemerge(index=index, max_num_segments=1, wait_for_completion=True)
        client.indices.refresh(index=index)
        stats = client.indices.stats(index=index, metric="docs")
        return int(stats["indices"][index]["primaries"]["docs"]["deleted"])

    def count_upload_files(self, doc_id: str) -> int:
        return sum(1 for _ in self.uploads_dir.glob(f"{doc_id}_*"))

    def delete_upload_file_count(self, doc_id: str) -> int:
        count = self.count_upload_files(doc_id)
        self.delete_upload_file(doc_id)
        return count

    def forget_registry_entries(self, source_keys: Iterable[str], *, tenant: str | None = None) -> None:
        wanted = set(source_keys)
        scope = normalize_tenant(tenant)
        self._doc_states = {
            doc_id: report
            for doc_id, report in self._doc_states.items()
            if not (report.source_key in wanted and normalize_tenant(report.tenant) == scope)
        }
        for key in wanted:
            self._current_by_source_key.pop((scope, key), None)
        self._persist_registry()

    def archive_source(self, source_key: str, *, tenant: str | None = None) -> int:
        """Source-system archive: retain every version, remove them all from retrieval."""
        scope = normalize_tenant(tenant)
        versions = self.versions_for_sources([source_key], tenant=scope)
        for report in versions:
            self._doc_states[report.doc_id] = replace(report, status=IngestionState.ARCHIVED, is_current=False)
            self._set_chunks_current(report.doc_id, is_current=False, status=IngestionState.ARCHIVED.value)
        self._current_by_source_key.pop((scope, source_key), None)
        self._persist_registry()
        return len(versions)

    def purge_tenant(self, tenant: str) -> Dict[str, Any]:
        """Offboarding: remove everything one tenant stored. Other tenants are untouched.

        Silo mode drops the tenant's index, which frees the disk at once. Pool mode can only
        delete_by_query, which leaves the text in Lucene segments until a merge
        (``expunge_deleted_documents``).
        """
        scope = require_tenant(tenant)
        doc_ids = [report.doc_id for report in self._tenant_reports(scope)]
        removed_chunks = 0
        method = "memory"
        if self.es_manager.is_online():
            if self.es_manager.per_tenant:
                method = "index_delete"
                index = self.es_manager.index_for(scope)
                if self.es_manager.client.indices.exists(index=index):
                    self.es_manager.client.indices.refresh(index=index)
                    removed_chunks = int(self.es_manager.client.count(index=index)["count"])
                self.es_manager.drop_tenant_index(scope)
            else:
                method = "delete_by_query"
                resp = self.es_manager.client.delete_by_query(
                    index=self.es_manager.ensure_index(scope),
                    query={"term": {"tenant": scope}},
                    refresh=True,
                    conflicts="proceed",
                )
                removed_chunks = int(resp.get("deleted", 0))
        before = len(self.es_manager.memory_chunks)
        self.es_manager.memory_chunks = [rec for rec in self.es_manager.memory_chunks if not self._in_scope(rec, scope)]
        removed_chunks += before - len(self.es_manager.memory_chunks)
        files = sum(self.delete_upload_file_count(doc_id) for doc_id in doc_ids)
        tables = self.tabular_store.delete_tenant(scope)
        self._doc_states = {
            doc_id: report for doc_id, report in self._doc_states.items() if normalize_tenant(report.tenant) != scope
        }
        self._current_by_source_key = {
            key: doc_id for key, doc_id in self._current_by_source_key.items() if key[0] != scope
        }
        self._persist_registry()
        return {
            "tenant": scope,
            "method": method,
            "documents": len(doc_ids),
            "chunks": removed_chunks,
            "upload_files": files,
            "tables": tables,
        }

    # --- Tombstones and processed events ---------------------------------

    def keyed_hash(self, value: str) -> str:
        if self._erasure_key is None:
            self._erasure_key = load_erasure_key(self._erasure_key_path)
        return hash_with_key(self._erasure_key, value)

    def add_tombstones(
        self,
        *,
        request_id: str,
        source_keys: Iterable[str],
        content_hashes: Iterable[str],
        subject_id: Optional[str] = None,
        tenant: str | None = None,
    ) -> None:
        """Record what was erased, per tenant.

        A tombstone only blocks the tenant that asked for the erasure. If it were global,
        one tenant erasing a file would stop every other tenant uploading the same bytes,
        and the refusal would tell them that somebody else once held that file.
        """
        for key in source_keys:
            self._tombstones[self.keyed_hash(tenant_scoped("source", tenant, key))] = request_id
        for digest in content_hashes:
            self._tombstones[self.keyed_hash(tenant_scoped("content", tenant, digest))] = request_id
        if subject_id:
            self._tombstones[self.keyed_hash(tenant_scoped("subject", tenant, subject_id))] = request_id
        self._persist_registry()

    def assert_not_erased(
        self,
        *,
        source_key: str = "",
        content_hash: str = "",
        subject_ids: Iterable[str] = (),
        tenant: str | None = None,
    ) -> None:
        """Refuse ingestion of anything a deletion request removed from this tenant."""
        if not self._tombstones:
            return
        checks = [
            ("source", tenant_scoped("source", tenant, source_key), bool(source_key)),
            ("content", tenant_scoped("content", tenant, content_hash), bool(content_hash)),
        ] + [("data subject", tenant_scoped("subject", tenant, normalize_subject_id(s)), True) for s in subject_ids]
        for label, value, enabled in checks:
            request_id = self._tombstones.get(self.keyed_hash(value)) if enabled else None
            if request_id:
                raise ErasedSourceError(
                    f"Blocked: this {label} was hard-deleted by erasure request {request_id}. "
                    "Re-ingesting it would resurrect erased data."
                )

    def lift_tombstone(self, source_key: str, *, tenant: str | None = None) -> bool:
        """Explicit, deliberate re-admission of a source, e.g. after redaction."""
        removed = self._tombstones.pop(self.keyed_hash(tenant_scoped("source", tenant, source_key)), None) is not None
        if removed:
            self._persist_registry()
        return removed

    @staticmethod
    def _event_key(event_id: str, tenant: str | None) -> str:
        # Two tenants' source systems can both emit "evt-1". Neither may swallow the other's.
        return event_id if normalize_tenant(tenant) == DEFAULT_TENANT else tenant_scoped("event", tenant, event_id)

    def has_processed_event(self, event_id: str, *, tenant: str | None = None) -> bool:
        return self._event_key(event_id, tenant) in self._processed_events

    def record_processed_event(self, event_id: str, *, tenant: str | None = None) -> None:
        self._processed_events.append(self._event_key(event_id, tenant))
        del self._processed_events[:-MAX_PROCESSED_EVENTS]
        self._persist_registry()
