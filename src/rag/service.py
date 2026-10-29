"""Pure Domain Orchestration Service for BeyondRAG.

Coordinates ACL management (`acl.py`), text chunking (`ports.py`), and injected
ports (`ChunkStorePort`, `CrossLingualPort`, `AnswerSynthesizerPort`, `PdfTextExtractorPort`)
without depending directly on Elasticsearch, Gemini, Ollama, or FastAPI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Final, List

from src.rag.acl import (
    PUBLIC_SENTINEL_TAG,
    build_effective_acl_tags,
    collect_available_tags,
    extract_visible_tags,
    normalize_tags,
    resolve_document_acl,
)
from src.rag.ports import (
    AnswerSynthesizerPort,
    ChunkStorePort,
    CrossLingualPort,
    PdfTextExtractorPort,
    chunk_text,
)

KB_INDEX_NAME: Final[str] = "hk-rbac-rag-kb"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_default_adapters(
    es_url: str | None = None,
) -> tuple[
    ChunkStorePort,
    CrossLingualPort,
    AnswerSynthesizerPort,
    PdfTextExtractorPort,
    Callable[[Any], List[Dict[str, Any]]],
]:
    """Composition root helper that wires default infrastructure adapters."""
    from src.config import ELASTICSEARCH_URL
    from src.rag.cross_lingual import DefaultCrossLingualAdapter
    from src.rag.retriever import ElasticsearchHybridStore
    from src.rag.seeding import LocalPdfTextExtractor, seed_demo_pdfs
    from src.rag.synthesizer import GroundedAnswerSynthesizer

    resolved_url = es_url or ELASTICSEARCH_URL
    return (
        ElasticsearchHybridStore(es_url=resolved_url, index_name=KB_INDEX_NAME),
        DefaultCrossLingualAdapter(),
        GroundedAnswerSynthesizer(),
        LocalPdfTextExtractor(),
        seed_demo_pdfs,
    )


def _create_pdf_bytes(title: str, pages: List[List[str]]) -> bytes:
    """Backward-compatible wrapper delegating PDF byte generation to `seeding.py`."""
    from src.rag.seeding import create_pdf_bytes

    return create_pdf_bytes(title=title, pages=pages)


class RBACKnowledgeBaseService:
    """Domain application service orchestrating RBAC PDF ingestion and Cross-Lingual RAG."""

    def __init__(
        self,
        es_url: str | None = None,
        *,
        store: ChunkStorePort | None = None,
        cross_lingual: CrossLingualPort | None = None,
        synthesizer: AnswerSynthesizerPort | None = None,
        pdf_extractor: PdfTextExtractorPort | None = None,
        demo_seeder: Callable[[Any], List[Dict[str, Any]]] | None = None,
    ) -> None:
        if (
            store is None
            or cross_lingual is None
            or synthesizer is None
            or pdf_extractor is None
            or demo_seeder is None
        ):
            def_store, def_xl, def_synth, def_pdf, def_seeder = _build_default_adapters(es_url)
            self._store: ChunkStorePort = store or def_store
            self._cross_lingual: CrossLingualPort = cross_lingual or def_xl
            self._synthesizer: AnswerSynthesizerPort = synthesizer or def_synth
            self._pdf_extractor: PdfTextExtractorPort = pdf_extractor or def_pdf
            self._demo_seeder = demo_seeder or def_seeder
        else:
            self._store = store
            self._cross_lingual = cross_lingual
            self._synthesizer = synthesizer
            self._pdf_extractor = pdf_extractor
            self._demo_seeder = demo_seeder

    @property
    def es_url(self) -> str:
        return self._store.backend_url

    def _is_es_online(self) -> bool:
        return self._store.is_online()

    def ingest_pdf_bytes(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: str | None = None,
    ) -> Dict[str, Any]:
        """Extract text from a PDF, generate cross-lingual concepts, embed, and index with ACL tags."""
        acl = resolve_document_acl(tags)
        assigned_doc_id = doc_id or f"doc-{uuid.uuid4().hex[:10]}"
        safe_filename = Path(filename).name
        uploaded_at = _utc_now()

        self._pdf_extractor.save_upload(assigned_doc_id, safe_filename, pdf_bytes)
        total_pages, page_texts = self._pdf_extractor.extract_pages(pdf_bytes)

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
            for chunk_str in chunk_text(extracted_text):
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
                        "embedding": self._cross_lingual.embed(enriched_text),
                    }
                )

        self._store.index_chunks(chunk_records)

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

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a deduplicated list of all indexed PDF documents and their RBAC tags."""
        docs_by_id: Dict[str, Dict[str, Any]] = {}
        for item in self._store.list_all_chunk_metadata():
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

    def list_available_tags(self) -> List[str]:
        """Return all unique RBAC tags currently in use plus standard enterprise tags."""
        return collect_available_tags(self.list_documents())

    def update_document_tags(
        self,
        doc_id: str,
        tags: List[str] | str | None,
    ) -> Dict[str, Any]:
        """Update a document's RBAC tags in-place with zero re-embedding."""
        acl = resolve_document_acl(tags)
        self._store.update_document_acl(
            doc_id=doc_id,
            acl_tags=acl.acl_tags,
            is_public=acl.is_public,
        )
        return {
            "doc_id": doc_id,
            "is_public": acl.is_public,
            "tags": acl.visible_tags,
            "acl_tags": acl.acl_tags,
        }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document and all its indexed chunks."""
        self._store.delete_document_chunks(doc_id)
        self._pdf_extractor.delete_upload(doc_id)
        return {"deleted": True, "doc_id": doc_id}

    def seed_sample_pdfs(self) -> List[Dict[str, Any]]:
        """Seed the 4 enterprise sample PDFs via the injected seeder."""
        return self._demo_seeder(self)

    def reindex_all_sample_pdfs(self) -> List[Dict[str, Any]]:
        """Clear the chunk store and re-seed the 4 sample PDFs."""
        self._store.reset_store()
        return self.seed_sample_pdfs()

    def query_and_chat(
        self,
        question: str,
        user_roles: List[str] | str | None = None,
        top_k: int = 4,
        synthesis_mode: str = "fast",
    ) -> Dict[str, Any]:
        """Execute Pre-Filtered Cross-Lingual Hybrid Search and synthesize a grounded answer."""
        cleaned_roles, effective_acl_tags = build_effective_acl_tags(user_roles)
        is_cjk_query = self._cross_lingual.is_cross_lingual(question)

        expanded_query, matched_concepts = self._cross_lingual.prepare_query(
            question=question,
            allow_llm_fallback=(synthesis_mode == "llm"),
        )
        query_vector = self._cross_lingual.embed(expanded_query)

        search_outcome = self._store.hybrid_search(
            expanded_query=expanded_query,
            query_vector=query_vector,
            effective_acl_tags=effective_acl_tags,
            top_k=top_k,
        )

        answer = self._synthesizer.synthesize(
            question=question,
            expanded_query=expanded_query,
            matched_concepts=matched_concepts,
            is_cjk_query=is_cjk_query,
            user_roles=cleaned_roles,
            authorized_chunks=search_outcome.authorized_chunks,
            masked_restricted_docs=search_outcome.masked_restricted_docs,
            synthesis_mode=synthesis_mode,
        )

        return {
            "question": question,
            "is_cross_lingual": is_cjk_query,
            "matched_concepts": matched_concepts,
            "expanded_query": expanded_query,
            "user_roles": cleaned_roles,
            "effective_acl_filter": effective_acl_tags,
            "latency_ms": search_outcome.latency_ms,
            "authorized_chunks": search_outcome.authorized_chunks,
            "masked_restricted_docs": search_outcome.masked_restricted_docs,
            "es_query_dsl": search_outcome.query_dsl,
            "answer": answer,
        }


_KB_SERVICE_INSTANCE: RBACKnowledgeBaseService | None = None


def get_kb_service() -> RBACKnowledgeBaseService:
    """Return the singleton RBACKnowledgeBaseService instance and ensure sample PDFs are indexed."""
    global _KB_SERVICE_INSTANCE
    if _KB_SERVICE_INSTANCE is None:
        _KB_SERVICE_INSTANCE = RBACKnowledgeBaseService()
        _KB_SERVICE_INSTANCE.reindex_all_sample_pdfs()
    return _KB_SERVICE_INSTANCE
