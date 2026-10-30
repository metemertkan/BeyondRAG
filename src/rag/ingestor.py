"""Component 5: Robust, Idempotent Document Ingestion Pipeline (`PdfIngestor`).

Guarantees:
  - Explicit lifecycle states (PENDING -> PROCESSING -> INDEXED / FAILED)
  - Deterministic document & chunk IDs (content-hashed)
  - Zero duplicate chunks on ingestion retry
  - Immediate cleanup/rollback on parser, embedding, or indexing failure
  - Safe replacement of existing documents without leaving stale chunks
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.acl import resolve_document_acl
from src.rag.models import IngestionReport, IngestionState
from src.rag.ports import (
    ChunkerPort,
    CrossLingualPort,
    DocumentParserPort,
    DocumentRepositoryPort,
    EmbedderPort,
    PdfTextExtractorPort,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_deterministic_doc_id(filename: str, file_bytes: bytes) -> str:
    """Derive stable, idempotent document ID from filename and content hash."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    clean_name = "".join(c for c in Path(filename).stem.lower() if c.isalnum() or c in ("-", "_"))[:20]
    return f"doc-{clean_name}-{content_hash}"


class PdfIngestor:
    """Domain orchestrator managing document ingestion lifecycle and transactional cleanup."""

    def __init__(
        self,
        extractor: DocumentParserPort | PdfTextExtractorPort,
        chunker: ChunkerPort,
        cross_lingual: CrossLingualPort,
        embedder: EmbedderPort,
        repository: DocumentRepositoryPort,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._cross_lingual = cross_lingual
        self._embedder = embedder
        self._repository = repository

    def ingest(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a document with explicit lifecycle transitions and transactional rollback."""
        if not pdf_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        safe_filename = Path(filename).name
        assigned_doc_id = doc_id or generate_deterministic_doc_id(safe_filename, pdf_bytes)
        acl = resolve_document_acl(tags)
        uploaded_at = _utc_now()

        # 1. State: PENDING -> PROCESSING
        pending_report = IngestionReport(
            doc_id=assigned_doc_id,
            filename=safe_filename,
            status=IngestionState.PROCESSING,
            page_count=0,
            chunk_count=0,
            is_public=acl.is_public,
            tags=acl.visible_tags,
            acl_tags=acl.acl_tags,
            uploaded_at=uploaded_at,
        )
        if hasattr(self._repository, "set_document_state"):
            self._repository.set_document_state(pending_report)

        saved_path = None
        try:
            # 2. Stage upload file
            saved_path = self._repository.save_upload_file(assigned_doc_id, safe_filename, pdf_bytes)

            # 3. Extract text
            if hasattr(self._extractor, "extract_pages"):
                try:
                    total_pages, page_texts = self._extractor.extract_pages(safe_filename, pdf_bytes)  # type: ignore
                except TypeError:
                    total_pages, page_texts = self._extractor.extract_pages(pdf_bytes)  # type: ignore
            else:
                raise ValueError(f"No compatible extractor found for {safe_filename}")

            if not page_texts:
                raise ValueError(
                    f"No extractable text found in '{safe_filename}'. Ensure the document contains selectable text."
                )

            # 4. Extract cross-lingual concepts
            full_doc_text = "\n\n".join(text for _, text in page_texts)
            extracted_concepts, concept_model = self._cross_lingual.extract_concepts_at_ingestion(
                filename=safe_filename,
                full_text=full_doc_text,
            )
            concept_ids = [c["concept_id"] for c in extracted_concepts]

            # 5. Segment into deterministic chunks
            chunk_records: List[Dict[str, Any]] = []
            global_chunk_idx = 0

            for page_num, extracted_text in page_texts:
                for chunk_str in self._chunker.split(extracted_text):
                    global_chunk_idx += 1
                    # Deterministic chunk ID: {doc_id}_p{page}_c{index}
                    chunk_id = f"{assigned_doc_id}_p{page_num}_c{global_chunk_idx}"
                    enriched_text = f"{safe_filename} (Page {page_num}): {chunk_str}"

                    # 6. Embed vector
                    vector = self._embedder.embed(enriched_text)
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
                        }
                    )

            # 7. Idempotently replace existing chunks in repository without deleting uploaded file
            if hasattr(self._repository, "delete_chunks"):
                self._repository.delete_chunks(assigned_doc_id)
            else:
                self._repository.delete_document(assigned_doc_id)
            self._repository.save_chunks(chunk_records)

            # 8. State: INDEXED
            success_report = IngestionReport(
                doc_id=assigned_doc_id,
                filename=safe_filename,
                status=IngestionState.INDEXED,
                page_count=total_pages,
                chunk_count=len(chunk_records),
                is_public=acl.is_public,
                tags=acl.visible_tags,
                acl_tags=acl.acl_tags,
                uploaded_at=uploaded_at,
                dynamic_concepts=concept_ids,
                concept_model=concept_model,
            )
            if hasattr(self._repository, "set_document_state"):
                self._repository.set_document_state(success_report)

            return {
                "doc_id": assigned_doc_id,
                "filename": safe_filename,
                "status": IngestionState.INDEXED.value,
                "page_count": total_pages,
                "chunk_count": len(chunk_records),
                "is_public": acl.is_public,
                "tags": acl.visible_tags,
                "acl_tags": acl.acl_tags,
                "dynamic_concepts": concept_ids,
                "concept_model": concept_model,
                "uploaded_at": uploaded_at,
            }

        except Exception as exc:
            logger.error("Ingestion failed for %s (%s): %s. Rolling back.", assigned_doc_id, safe_filename, exc)
            # Rollback: delete any partially saved chunks & upload file
            self._repository.delete_document(assigned_doc_id)

            failed_report = IngestionReport(
                doc_id=assigned_doc_id,
                filename=safe_filename,
                status=IngestionState.FAILED,
                page_count=0,
                chunk_count=0,
                is_public=acl.is_public,
                tags=acl.visible_tags,
                acl_tags=acl.acl_tags,
                uploaded_at=uploaded_at,
                error_message=str(exc),
            )
            if hasattr(self._repository, "set_document_state"):
                self._repository.set_document_state(failed_report)

            raise exc
