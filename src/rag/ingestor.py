"""Component 5: Pure Domain PDF Ingestion Pipeline (`PdfIngestor`).

Coordinates text extraction, chunking, cross-lingual concept extraction, vector
embedding, and repository persistence through domain ports (zero direct dependency
on Elasticsearch, PyPDF, Ollama, or Gemini).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.rag.acl import resolve_document_acl
from src.rag.ports import (
    ChunkerPort,
    CrossLingualPort,
    DocumentRepositoryPort,
    EmbedderPort,
    PdfTextExtractorPort,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PdfIngestor:
    """Domain component that orchestrates ingesting a PDF document into the knowledge base."""

    def __init__(
        self,
        extractor: PdfTextExtractorPort,
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
        doc_id: str | None = None,
    ) -> Dict[str, Any]:
        """Extract text, generate dynamic cross-lingual concepts, chunk, embed, and persist."""
        acl = resolve_document_acl(tags)
        assigned_doc_id = doc_id or f"doc-{uuid.uuid4().hex[:10]}"
        safe_filename = Path(filename).name
        uploaded_at = _utc_now()

        self._repository.save_upload_file(assigned_doc_id, safe_filename, pdf_bytes)
        total_pages, page_texts = self._extractor.extract_pages(pdf_bytes)

        if not page_texts:
            raise ValueError(
                f"No extractable text found in '{safe_filename}'. Ensure the PDF contains selectable text."
            )

        full_doc_text = "\n\n".join(text for _, text in page_texts)
        extracted_concepts, concept_model = self._cross_lingual.extract_concepts_at_ingestion(
            filename=safe_filename,
            full_text=full_doc_text,
        )
        concept_ids = [c["concept_id"] for c in extracted_concepts]

        chunk_records: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for page_num, extracted_text in page_texts:
            for chunk_str in self._chunker.split(extracted_text):
                global_chunk_idx += 1
                chunk_id = f"{assigned_doc_id}_p{page_num}_c{global_chunk_idx}"
                enriched_text = f"{safe_filename} (Page {page_num}): {chunk_str}"
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
                        "embedding": self._embedder.embed(enriched_text),
                    }
                )

        self._repository.save_chunks(chunk_records)

        return {
            "doc_id": assigned_doc_id,
            "filename": safe_filename,
            "page_count": total_pages,
            "chunk_count": len(chunk_records),
            "is_public": acl.is_public,
            "tags": acl.visible_tags,
            "acl_tags": acl.acl_tags,
            "dynamic_concepts": concept_ids,
            "concept_model": concept_model,
            "uploaded_at": uploaded_at,
        }
