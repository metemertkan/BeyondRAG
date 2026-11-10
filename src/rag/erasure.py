"""Hard deletion with lineage tracking (GDPR Art. 17 / CCPA right to delete).

Deleting "the file" is not erasure. One person's data ends up in many places:

    uploaded bytes -> every chunk of every retained version -> vectors -> SQLite rows
    -> concept triggers extracted by an LLM -> cached query translations -> the registry

Soft versioning makes this sharper, because a SUPERSEDED version is retained on purpose
and a delete by ``doc_id`` leaves it behind. Here the master lineage key is
``source_key``, stamped on every chunk, plus ``subject_ids`` naming the people involved.

Flow: plan (dry run) -> tombstone -> purge every store -> verify by re-scanning -> log
-> confirm the physical purge. Search stops returning the data at once, but Elasticsearch
keeps deleted text in its segments under a retention lease for a while. That second phase
is tracked and audited separately, so "deleted" never overstates what has happened on disk.

The tombstone goes first and survives the purge, so a nightly re-sync cannot quietly
resurrect the data. The audit log stores only keyed hashes, because a deletion record
must not itself retain the identifier that was deleted.

A request runs inside one tenant. A subject id or a source key is only unique within a
tenant, so the plan carries the tenant and every store is asked about that tenant alone.
One company's erasure never touches, counts or blocks another company's data.

Out of scope, and stated on every receipt: ES snapshots and other backups, which need
their own retention window, and copies outside this system.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.rag.governance import normalize_subject_id
from src.rag.tenancy import DEFAULT_TENANT, normalize_tenant

logger = logging.getLogger(__name__)

BACKUP_NOTICE = (
    "Covers live stores only. Elasticsearch snapshots and other backups expire under "
    "their own retention window; copies outside this system are not reached."
)


class ErasedSourceError(ValueError):
    """Raised when ingestion would resurrect data that a deletion request removed."""


@dataclass
class ErasurePlan:
    """What a deletion would remove. Review this before executing a subject request."""

    subject_id: Optional[str]
    source_keys: List[str]
    # source_key -> "declared" | "detected" | "source_request"
    match_type: Dict[str, str]
    versions: List[Dict[str, Any]]
    chunk_count: int
    table_ids: List[str]
    residual_mentions: int = 0
    tenant: str = DEFAULT_TENANT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant": self.tenant,
            "subject_id": self.subject_id,
            "source_keys": self.source_keys,
            "match_type": self.match_type,
            "versions": self.versions,
            "chunk_count": self.chunk_count,
            "table_ids": self.table_ids,
            "residual_mentions": self.residual_mentions,
        }


@dataclass
class ErasureReceipt:
    """Proof of deletion. Contains counts and keyed hashes, never the identifier."""

    request_id: str
    completed_at: str
    subject_hash: Optional[str]
    source_hashes: List[str]
    removed: Dict[str, int]
    verified: bool
    residuals: Dict[str, int] = field(default_factory=dict)
    # Deleted documents still physically in Lucene segments when this receipt was written.
    soft_deleted_pending: int = 0
    notice: str = BACKUP_NOTICE
    tenant: str = DEFAULT_TENANT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tenant": self.tenant,
            "completed_at": self.completed_at,
            "subject_hash": self.subject_hash,
            "source_hashes": self.source_hashes,
            "removed": self.removed,
            "verified": self.verified,
            "residuals": self.residuals,
            "soft_deleted_pending": self.soft_deleted_pending,
            "notice": self.notice,
        }


def load_erasure_key(key_path: Path) -> bytes:
    """Keyed hashing: a plain SHA-256 of an email can be reversed with a dictionary."""
    configured = os.getenv("RAG_ERASURE_KEY", "").strip()
    if configured:
        return configured.encode("utf-8")
    if key_path.exists():
        return key_path.read_bytes()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    key_path.touch(mode=0o600)
    key_path.write_bytes(key)
    return key


class ErasureService:
    """Coordinates lineage lookup, hard deletion, verification and the audit log."""

    def __init__(
        self,
        repository: Any,
        *,
        concept_registry: Any = None,
        audit_log_path: Optional[Path] = None,
    ) -> None:
        self._repository = repository
        self._concepts = concept_registry
        self._audit_log_path = audit_log_path

    def _hash(self, value: str) -> str:
        return self._repository.keyed_hash(value)

    def plan_subject(
        self, subject_id: str, *, include_detected: bool = True, tenant: str | None = None
    ) -> ErasurePlan:
        """Find every source linked to a person in one tenant, via declared and detected lineage."""
        subject = normalize_subject_id(subject_id)
        if not subject:
            raise ValueError("subject_id must not be empty")
        scope = normalize_tenant(tenant)
        declared, detected = self._repository.find_sources_by_subject(subject, tenant=scope)
        match_type = {key: "declared" for key in declared}
        if include_detected:
            match_type.update({key: "detected" for key in detected if key not in match_type})
        plan = self._build_plan(subject, match_type, scope)
        plan.residual_mentions = self._repository.count_subject_mentions(
            subject, exclude_source_keys=plan.source_keys, tenant=scope
        )
        return plan

    def plan_source(self, source_key: str, *, tenant: str | None = None) -> ErasurePlan:
        return self._build_plan(None, {source_key: "source_request"}, normalize_tenant(tenant))

    def _build_plan(self, subject: Optional[str], match_type: Dict[str, str], tenant: str) -> ErasurePlan:
        source_keys = sorted(match_type)
        versions = [
            {
                "doc_id": report.doc_id,
                "source_key": report.source_key,
                "filename": report.filename,
                "doc_version": report.doc_version,
                "status": report.status.value,
            }
            for report in self._repository.versions_for_sources(source_keys, tenant=tenant)
        ]
        return ErasurePlan(
            subject_id=subject,
            source_keys=source_keys,
            match_type=match_type,
            versions=versions,
            chunk_count=self._repository.count_chunks(source_keys=source_keys, subject_id=subject, tenant=tenant),
            table_ids=[
                table_id
                for key in source_keys
                for table_id in self._repository.tabular_store.table_ids_for(source_key=key, tenant=tenant)
            ],
            tenant=tenant,
        )

    def execute(self, plan: ErasurePlan, *, request_id: Optional[str] = None) -> ErasureReceipt:
        """Tombstone, purge every store, re-scan, and write the audit record.

        Every step is idempotent, so a partly failed run is safe to repeat. An exception
        leaves the tombstones in place, which keeps the source blocked until it finishes.
        """
        request_id = request_id or f"erasure-{uuid.uuid4().hex[:12]}"
        repository = self._repository
        tenant = normalize_tenant(plan.tenant)
        versions = repository.versions_for_sources(plan.source_keys, tenant=tenant)

        repository.add_tombstones(
            request_id=request_id,
            source_keys=plan.source_keys,
            content_hashes=[report.content_hash for report in versions if report.content_hash],
            subject_id=plan.subject_id,
            tenant=tenant,
        )

        removed = {
            "chunks": repository.delete_chunks_by_lineage(
                source_keys=plan.source_keys, subject_id=plan.subject_id, tenant=tenant
            ),
            "versions": len(versions),
            "upload_files": sum(repository.delete_upload_file_count(report.doc_id) for report in versions),
            "tables": sum(
                repository.tabular_store.delete_tables(source_key=key, tenant=tenant) for key in plan.source_keys
            ),
        }
        concept_ids = [concept for report in versions for concept in report.dynamic_concepts]
        removed["concepts"] = (
            self._concepts.forget(concept_ids, subject_id=plan.subject_id, tenant=tenant) if self._concepts else 0
        )
        repository.forget_registry_entries(plan.source_keys, tenant=tenant)
        soft_deleted_pending = repository.expunge_deleted_documents(tenant=tenant)

        residuals = self.verify(plan, doc_ids=[report.doc_id for report in versions])
        receipt = ErasureReceipt(
            request_id=request_id,
            completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            subject_hash=self._hash(plan.subject_id) if plan.subject_id else None,
            source_hashes=sorted(self._hash(key) for key in plan.source_keys),
            removed=removed,
            verified=not any(residuals.values()),
            residuals=residuals,
            soft_deleted_pending=soft_deleted_pending,
            tenant=tenant,
        )
        self._append_audit_log(receipt)
        return receipt

    def confirm_physical_purge(
        self,
        request_id: str,
        *,
        max_wait_seconds: float = 120.0,
        poll_seconds: float = 10.0,
        tenant: str | None = None,
    ) -> int:
        """Retry the merge until deleted text has left the segments, then audit the result.

        Returns the count still remaining. Non-zero means the deadline passed, for example
        because a replica holds a lease, and the audit record says so instead of claiming
        success. Run this after responding, since it can take a minute or more.
        """
        deadline = time.monotonic() + max_wait_seconds
        remaining = self._repository.expunge_deleted_documents(tenant=tenant)
        while remaining and time.monotonic() < deadline:
            time.sleep(poll_seconds)
            remaining = self._repository.expunge_deleted_documents(tenant=tenant)
        self._append_audit_record(
            {
                "request_id": request_id,
                "tenant": normalize_tenant(tenant),
                "event": "physical_purge_check",
                "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "soft_deleted_remaining": remaining,
            }
        )
        return remaining

    def verify(self, plan: ErasurePlan, *, doc_ids: Sequence[str]) -> Dict[str, int]:
        """Re-scan every store. Any non-zero count means the data is still somewhere."""
        repository = self._repository
        tenant = normalize_tenant(plan.tenant)
        residuals = {
            "chunks_by_lineage": repository.count_chunks(
                source_keys=plan.source_keys, subject_id=plan.subject_id, tenant=tenant
            ),
            "upload_files": sum(repository.count_upload_files(doc_id) for doc_id in doc_ids),
            "registry_entries": len(repository.versions_for_sources(plan.source_keys, tenant=tenant)),
            "tables": sum(
                len(repository.tabular_store.table_ids_for(source_key=key, tenant=tenant)) for key in plan.source_keys
            ),
        }
        if plan.subject_id:
            # Full-text scan catches mentions that lineage metadata never tagged.
            residuals["chunk_text_mentions"] = repository.count_subject_mentions(plan.subject_id, tenant=tenant)
            residuals["concept_mentions"] = (
                self._concepts.count_mentions(plan.subject_id, tenant=tenant) if self._concepts else 0
            )
            # A document kept with include_detected=False still holds the person's data in its
            # raw file, so this counts as incomplete until someone redacts or removes it.
            declared, detected = repository.find_sources_by_subject(plan.subject_id, tenant=tenant)
            residuals["sources_still_linked"] = len(set(declared) | set(detected))
        return residuals

    def _append_audit_log(self, receipt: ErasureReceipt) -> None:
        self._append_audit_record(receipt.to_dict())

    def _append_audit_record(self, record: Dict[str, Any]) -> None:
        if self._audit_log_path is None:
            return
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class SourceSyncHandler:
    """Propagates source-system events (SharePoint, Salesforce, ...) to the index.

    Webhooks retry, so events are deduplicated by ``event_id``. A ``deleted`` event is
    a hard erasure of the source; ``archived`` keeps the data but removes it from
    retrieval. Events for an unknown source are acknowledged, since the source may
    simply never have been ingested.

    An event belongs to one tenant's source system. Event ids and source keys are scoped
    to that tenant, so one tenant's webhook cannot erase or archive another's document.
    """

    SUPPORTED = ("deleted", "archived")

    def __init__(self, repository: Any, erasure: ErasureService) -> None:
        self._repository = repository
        self._erasure = erasure

    def handle(self, *, event_id: str, event_type: str, source_key: str, tenant: str | None = None) -> Dict[str, Any]:
        if event_type not in self.SUPPORTED:
            raise ValueError(f"Unsupported event_type {event_type!r}. Supported: {list(self.SUPPORTED)}")
        scope = normalize_tenant(tenant)
        if self._repository.has_processed_event(event_id, tenant=scope):
            return {"event_id": event_id, "status": "duplicate_ignored"}

        if event_type == "deleted":
            receipt = self._erasure.execute(
                self._erasure.plan_source(source_key, tenant=scope), request_id=f"sync-{event_id}"
            )
            result: Dict[str, Any] = {"status": "erased", "receipt": receipt.to_dict()}
        else:
            result = {"status": "archived", "versions": self._repository.archive_source(source_key, tenant=scope)}

        self._repository.record_processed_event(event_id, tenant=scope)
        return {"event_id": event_id, **result}


def hash_with_key(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
