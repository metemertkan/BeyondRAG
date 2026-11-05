"""Component 5: Robust, Idempotent Document Ingestion Pipeline (`PdfIngestor`).

Guarantees:
  - Explicit lifecycle: PENDING -> PROCESSING -> INDEXED / FAILED (+ soft SUPERSEDED)
  - Logical ``source_key`` (tenant-scoped) separate from per-bytes ``doc_id`` / content hash
  - Soft supersession: revised bytes keep prior versions retained but excluded from retrieval
  - Deterministic document & chunk IDs (content-hashed); retry same bytes → no duplicates
  - Embedding manifest locked at first INDEXED write; mismatch fails fast
  - Structure-preserving chunk metadata (block_type, page_number, parser_name)
  - Immediate cleanup/rollback on parser, embedding, or indexing failure
  - Lineage and governance on every chunk: source_key, subject_ids, trust tier, dates
  - Erasure tombstones refuse to re-admit data that a deletion request removed
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

from src.rag.acl import resolve_document_acl
from src.rag.embeddings import (
    EmbeddingManifest,
    EmbeddingManifestMismatchError,
    assert_manifest_compatible,
    assert_vector_matches_manifest,
)
from src.rag.extractor import ParserDegradedError
from src.rag.governance import DocumentGovernance, normalize_subject_id
from src.rag.models import IngestionReport, IngestionState, ParsedBlock, ParsedDocument
from src.rag.ports import (
    ChunkerPort,
    CrossLingualPort,
    DocumentParserPort,
    DocumentRepositoryPort,
    EmbedderPort,
    PdfTextExtractorPort,
)

logger = logging.getLogger(__name__)

# Emails are the one identifier found reliably with no NER model. Other identifiers
# (names, employee IDs) must be declared through ``DocumentGovernance.subject_ids``.
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def detect_subject_ids(text: str) -> List[str]:
    return sorted({normalize_subject_id(match) for match in _EMAIL_RE.findall(text)})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_tenant(tenant: str | None) -> str:
    """Normalize tenant/scope segment used inside source_key."""
    raw = (tenant or "default").strip().lower() or "default"
    cleaned = "".join(c for c in raw if c.isalnum() or c in ("-", "_"))[:32]
    return cleaned or "default"


def generate_source_key(filename: str, tenant: str | None = None) -> str:
    """Stable logical document identity: ``{tenant}:{normalized_filename}``.

    Tenant/scope is part of identity so the same filename in two tenants does not
    collide. Default tenant is ``default`` when callers omit it (Ep1 single-tenant
    demos keep working). Content hash is intentionally NOT part of source_key.
    """
    safe_name = Path(filename).name.lower()
    return f"{normalize_tenant(tenant)}:{safe_name}"


def content_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def generate_deterministic_doc_id(
    filename: str,
    file_bytes: bytes,
    *,
    tenant: str | None = None,
) -> str:
    """Derive version-specific document ID from filename + content hash (+ tenant when non-default).

    Ep1 compatibility: with default tenant the ID remains ``doc-{stem}-{12hex}``.
    Non-default tenants prefix the stem so identity includes scope without rewriting
    historical Ep1 content claims about the blog post itself.
    """
    content_hash = content_sha256(file_bytes)[:12]
    clean_name = "".join(
        c for c in Path(filename).stem.lower() if c.isalnum() or c in ("-", "_")
    )[:20]
    tenant_norm = normalize_tenant(tenant)
    if tenant_norm != "default":
        return f"doc-{tenant_norm}-{clean_name}-{content_hash}"
    return f"doc-{clean_name}-{content_hash}"


def _resolve_embedder_manifest(embedder: EmbedderPort) -> EmbeddingManifest:
    if hasattr(embedder, "manifest"):
        return embedder.manifest  # type: ignore[attr-defined]
    dims = getattr(embedder, "dimensions", None)
    if dims is None:
        dims = 32
    return EmbeddingManifest(
        provider="unknown",
        model_name=embedder.__class__.__name__,
        dimensions=int(dims),
    )


class PdfIngestor:
    """Domain orchestrator managing document ingestion lifecycle and transactional cleanup."""

    def __init__(
        self,
        extractor: DocumentParserPort | PdfTextExtractorPort,
        chunker: ChunkerPort,
        cross_lingual: CrossLingualPort,
        embedder: EmbedderPort,
        repository: DocumentRepositoryPort,
        expected_manifest: EmbeddingManifest | None = None,
        embedding_manifest: EmbeddingManifest | None = None,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._cross_lingual = cross_lingual
        self._embedder = embedder
        self._repository = repository
        # Explicit ingest-time contract (Ep2 fail-fast). Alias: embedding_manifest.
        self.expected_manifest = expected_manifest or embedding_manifest

    def validate_embedding_manifest(self, active: EmbeddingManifest) -> None:
        """Fail fast when active embedder disagrees with constructor or index contract."""
        if self.expected_manifest is not None:
            assert_manifest_compatible(active, self.expected_manifest.config_identifier)
        if hasattr(self._repository, "get_index_embedding_manifest"):
            assert_manifest_compatible(active, self._repository.get_index_embedding_manifest())

    def _set_state(self, report: IngestionReport) -> None:
        if hasattr(self._repository, "set_document_state"):
            self._repository.set_document_state(report)

    def _parse_document(self, filename: str, file_bytes: bytes) -> ParsedDocument:
        if hasattr(self._extractor, "extract_document"):
            try:
                return self._extractor.extract_document(filename, file_bytes)  # type: ignore[attr-defined]
            except TypeError:
                return self._extractor.extract_document(file_bytes)  # type: ignore[attr-defined]

        if hasattr(self._extractor, "extract_pages"):
            try:
                total_pages, page_texts = self._extractor.extract_pages(filename, file_bytes)  # type: ignore
            except TypeError:
                total_pages, page_texts = self._extractor.extract_pages(file_bytes)  # type: ignore
            blocks = [
                ParsedBlock(block_type="paragraph", text=text, page_number=num, metadata={})
                for num, text in page_texts
            ]
            return ParsedDocument(
                page_count=total_pages,
                pages=page_texts,
                blocks=blocks,
                parser_name=getattr(self._extractor, "name", self._extractor.__class__.__name__),
            )
        raise ValueError(f"No compatible extractor found for {filename}")

    def _build_chunks_from_blocks(
        self,
        *,
        blocks: List[ParsedBlock],
        pages: List[tuple[int, str]],
        assigned_doc_id: str,
        safe_filename: str,
        acl: Any,
        uploaded_at: str,
        source_key: str,
        content_hash: str,
        doc_version: int,
        tenant: str,
        manifest: EmbeddingManifest,
        parser_name: str,
        chunks_current: bool = True,
        governance: DocumentGovernance = DocumentGovernance(),
    ) -> List[Dict[str, Any]]:
        chunk_records: List[Dict[str, Any]] = []
        global_chunk_idx = 0
        units: List[tuple[int, str, str, Dict[str, Any]]] = []
        effective_date = governance.effective_date or uploaded_at[:10]

        def routed_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
            decision = getattr(self._chunker, "last_decision", None)
            return {**metadata, "chunking": dict(decision)} if decision else dict(metadata)

        if blocks:
            for block in blocks:
                if block.block_type in {"table", "table_summary"}:
                    # Keep tables intact as retrieval units (structure-preserving).
                    units.append((block.page_number, block.text, block.block_type, dict(block.metadata)))
                else:
                    pieces = self._chunker.split(block.text)
                    metadata = routed_metadata(block.metadata)
                    for piece in pieces:
                        units.append((block.page_number, piece, block.block_type, metadata))
        else:
            for page_num, extracted_text in pages:
                pieces = self._chunker.split(extracted_text)
                metadata = routed_metadata({})
                for piece in pieces:
                    units.append((page_num, piece, "paragraph", metadata))

        for page_num, chunk_str, block_type, block_meta in units:
            global_chunk_idx += 1
            chunk_id = f"{assigned_doc_id}_p{page_num}_c{global_chunk_idx}"
            enriched_text = f"{safe_filename} (Page {page_num}): {chunk_str}"
            vector = self._embedder.embed(enriched_text)
            assert_vector_matches_manifest(vector, manifest)
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": assigned_doc_id,
                    "filename": safe_filename,
                    "title": safe_filename,
                    "page_number": page_num,
                    "chunk_index": global_chunk_idx,
                    "content": chunk_str,
                    "is_public": acl.is_public,
                    "acl_tags": acl.acl_tags,
                    "uploaded_at": uploaded_at,
                    "embedding": vector,
                    "block_type": block_type,
                    "source_key": source_key,
                    "content_hash": content_hash,
                    "doc_version": doc_version,
                    "tenant": tenant,
                    "is_current": chunks_current,
                    "ingestion_status": IngestionState.INDEXED.value,
                    "embedding_manifest": manifest.config_identifier,
                    "parser_name": parser_name,
                    "trust_tier": governance.trust_tier,
                    "effective_date": effective_date,
                    "expires_at": governance.expires_at,
                    "business_domain": governance.business_domain,
                    "source_system": governance.source_system,
                    "subject_ids": sorted(set(governance.subject_ids) | set(detect_subject_ids(chunk_str))),
                    "metadata": {
                        "block_type": block_type,
                        **block_meta,
                    },
                }
            )
        return chunk_records

    def ingest(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: Optional[str] = None,
        tenant: str | None = None,
        source_key: str | None = None,
        governance: DocumentGovernance | None = None,
    ) -> Dict[str, Any]:
        """Ingest a document with PENDING→PROCESSING lifecycle, versioning, and rollback."""
        if not pdf_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        safe_filename = Path(filename).name
        tenant_norm = normalize_tenant(tenant)
        resolved_source_key = source_key or generate_source_key(safe_filename, tenant_norm)
        digest = content_sha256(pdf_bytes)
        assigned_doc_id = doc_id or generate_deterministic_doc_id(
            safe_filename, pdf_bytes, tenant=tenant_norm
        )
        acl = resolve_document_acl(tags)
        uploaded_at = _utc_now()
        governance = governance or DocumentGovernance()
        effective_date = governance.effective_date or uploaded_at[:10]

        # Before any state is written: erased data must not get back in, even as PENDING.
        if hasattr(self._repository, "assert_not_erased"):
            self._repository.assert_not_erased(
                source_key=resolved_source_key,
                content_hash=digest,
                subject_ids=governance.subject_ids,
            )

        previous_current_id: Optional[str] = None
        if hasattr(self._repository, "get_current_doc_id"):
            previous_current_id = self._repository.get_current_doc_id(resolved_source_key)

        # Same bytes already current → idempotent re-index path (no new version).
        same_bytes_retry = previous_current_id == assigned_doc_id
        same_bytes_prior: Optional[IngestionReport] = None
        if same_bytes_retry and hasattr(self._repository, "get_document_state"):
            same_bytes_prior = self._repository.get_document_state(assigned_doc_id)

        if hasattr(self._repository, "next_doc_version"):
            existing_state = same_bytes_prior if same_bytes_retry else None
            doc_version = (
                existing_state.doc_version
                if existing_state is not None
                else self._repository.next_doc_version(resolved_source_key)
            )
        else:
            doc_version = 1

        # 1. State: PENDING (accepted) — recorded before any parse/embed work.
        # Same-bytes retry keeps is_current=True so observers never see zero currents.
        pending_report = IngestionReport(
            doc_id=assigned_doc_id,
            filename=safe_filename,
            status=IngestionState.PENDING,
            page_count=0,
            chunk_count=0,
            is_public=acl.is_public,
            tags=acl.visible_tags,
            acl_tags=acl.acl_tags,
            uploaded_at=uploaded_at,
            source_key=resolved_source_key,
            content_hash=digest,
            doc_version=doc_version,
            tenant=tenant_norm,
            is_current=bool(same_bytes_retry),
            supersedes_doc_id=None if same_bytes_retry else previous_current_id,
            embedding_manifest="",
            parser_name=(same_bytes_prior.parser_name if same_bytes_prior else ""),
            trust_tier=governance.trust_tier,
            effective_date=effective_date,
            expires_at=governance.expires_at,
            business_domain=governance.business_domain,
            source_system=governance.source_system,
            subject_ids=list(governance.subject_ids),
        )
        self._set_state(pending_report)
        self._set_state(replace(pending_report, status=IngestionState.PROCESSING))

        try:
            manifest = _resolve_embedder_manifest(self._embedder)
            self.validate_embedding_manifest(manifest)

            self._repository.save_upload_file(assigned_doc_id, safe_filename, pdf_bytes)
            parsed = self._parse_document(safe_filename, pdf_bytes)

            if not parsed.pages and not parsed.blocks:
                raise ValueError(
                    f"No extractable text found in '{safe_filename}'. "
                    "Ensure the document contains selectable text."
                )

            # Same-bytes retry must not silently replace layout-aware INDEXED tables with flat PyPDF.
            # Use the pre-PENDING snapshot — registry state is already PROCESSING here.
            if same_bytes_retry and same_bytes_prior is not None:
                if same_bytes_prior.status == IngestionState.INDEXED:
                    prev_parser = (same_bytes_prior.parser_name or "").lower()
                    new_parser = (parsed.parser_name or "").lower()
                    degraded = bool(getattr(parsed, "parser_degraded", False))
                    if prev_parser == "pdfplumber" and (degraded or new_parser == "pypdf"):
                        reason = getattr(parsed, "parser_fallback_reason", "") or (
                            "layout-aware → flat parser on same-bytes retry"
                        )
                        raise ParserDegradedError(
                            "Same-bytes retry would replace pdfplumber INDEXED table chunks "
                            f"with degraded parser '{parsed.parser_name}' ({reason}). Refusing."
                        )

            full_doc_text = "\n\n".join(
                b.text for b in parsed.blocks if b.text
            ) or "\n\n".join(text for _, text in parsed.pages)
            extracted_concepts, concept_model = self._cross_lingual.extract_concepts_at_ingestion(
                filename=safe_filename,
                full_text=full_doc_text,
            )
            concept_ids = [c["concept_id"] for c in extracted_concepts]

            # Lineage: every email in the text links this source to that person.
            detected_subjects = [
                subject for subject in detect_subject_ids(full_doc_text) if subject not in governance.subject_ids
            ]
            if hasattr(self._repository, "assert_not_erased"):
                self._repository.assert_not_erased(subject_ids=detected_subjects)

            # Stage-then-publish: when replacing another version, stage chunks as
            # non-current first so v1 stays the sole searchable current until publish.
            revising = (
                bool(previous_current_id)
                and previous_current_id != assigned_doc_id
                and not same_bytes_retry
            )
            chunk_records = self._build_chunks_from_blocks(
                blocks=list(parsed.blocks),
                pages=list(parsed.pages),
                assigned_doc_id=assigned_doc_id,
                safe_filename=safe_filename,
                acl=acl,
                uploaded_at=uploaded_at,
                source_key=resolved_source_key,
                content_hash=digest,
                doc_version=doc_version,
                tenant=tenant_norm,
                manifest=manifest,
                parser_name=parsed.parser_name,
                chunks_current=not revising,
                governance=governance,
            )

            if hasattr(self._repository, "delete_chunks"):
                self._repository.delete_chunks(assigned_doc_id)
            else:
                self._repository.delete_document(assigned_doc_id)
            self._repository.save_chunks(chunk_records)
            parsed_tables = list(getattr(parsed, "tables", []) or [])
            if parsed_tables and hasattr(self._repository, "save_tables"):
                self._repository.save_tables(
                    parsed_tables,
                    doc_id=assigned_doc_id,
                    source_key=resolved_source_key,
                    acl_tags=acl.acl_tags,
                )

            if hasattr(self._repository, "set_index_embedding_manifest"):
                if not self._repository.get_index_embedding_manifest():
                    self._repository.set_index_embedding_manifest(manifest.config_identifier)

            success_report = replace(
                pending_report,
                status=IngestionState.INDEXED,
                page_count=parsed.page_count,
                chunk_count=len(chunk_records),
                dynamic_concepts=concept_ids,
                concept_model=concept_model,
                is_current=True,
                embedding_manifest=manifest.config_identifier,
                parser_name=parsed.parser_name,
                parser_degraded=bool(getattr(parsed, "parser_degraded", False)),
                parser_fallback_reason=str(getattr(parsed, "parser_fallback_reason", "") or ""),
                detected_subject_ids=detected_subjects,
                table_ids=[table.table_id for table in parsed_tables],
            )

            # Publish only after v2 chunks are written: INDEXED v2 then flip v1→SUPERSEDED.
            if revising and hasattr(self._repository, "publish_new_current"):
                self._repository.publish_new_current(
                    success_report,
                    previous_doc_id=previous_current_id,
                )
            else:
                self._set_state(success_report)
                if revising and hasattr(self._repository, "mark_superseded"):
                    self._repository.mark_superseded(previous_current_id)

            return {
                "doc_id": assigned_doc_id,
                "filename": safe_filename,
                "status": IngestionState.INDEXED.value,
                "page_count": parsed.page_count,
                "chunk_count": len(chunk_records),
                "is_public": acl.is_public,
                "tags": acl.visible_tags,
                "acl_tags": acl.acl_tags,
                "dynamic_concepts": concept_ids,
                "concept_model": concept_model,
                "uploaded_at": uploaded_at,
                "source_key": resolved_source_key,
                "content_hash": digest,
                "doc_version": doc_version,
                "tenant": tenant_norm,
                "is_current": True,
                "supersedes_doc_id": None if same_bytes_retry else previous_current_id,
                "embedding_manifest": manifest.config_identifier,
                "parser_name": parsed.parser_name,
                "parser_degraded": bool(getattr(parsed, "parser_degraded", False)),
                "parser_fallback_reason": str(
                    getattr(parsed, "parser_fallback_reason", "") or ""
                ),
                "trust_tier": governance.trust_tier,
                "effective_date": effective_date,
                "expires_at": governance.expires_at,
                "subject_ids": list(governance.subject_ids),
                "detected_subject_ids": detected_subjects,
                "table_ids": [table.table_id for table in parsed_tables],
            }

        except Exception as exc:
            logger.error(
                "Ingestion failed for %s (%s): %s. Rolling back.",
                assigned_doc_id,
                safe_filename,
                exc,
            )
            if same_bytes_retry and same_bytes_prior is not None:
                # Never wipe an already-INDEXED same-bytes doc (esp. degraded refuse).
                self._repository.set_document_state(same_bytes_prior)
                if hasattr(self._repository, "_set_chunks_current"):
                    self._repository._set_chunks_current(  # noqa: SLF001
                        assigned_doc_id,
                        is_current=True,
                        status=IngestionState.INDEXED.value,
                    )
                raise

            # Stage-then-publish: v1 was never superseded before v2 INDEXED, so a
            # failed revision leaves the previous current searchable without restore.
            self._repository.delete_document(assigned_doc_id)

            failed_report = replace(
                pending_report,
                status=IngestionState.FAILED,
                error_message=str(exc),
                is_current=False,
                embedding_manifest=(manifest.config_identifier if "manifest" in locals() else ""),
            )
            self._set_state(failed_report)

            if isinstance(exc, EmbeddingManifestMismatchError):
                raise
            raise
