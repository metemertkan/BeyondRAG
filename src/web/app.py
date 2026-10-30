"""FastAPI application exposing PDF upload, RBAC tagging, and Pre-Filtered Hybrid RAG chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag.service import KB_INDEX_NAME, get_kb_service, normalize_tags

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="HK-FDE RBAC RAG Knowledge Base",
    description=(
        "Enterprise Role-Based Access Control (RBAC) RAG portal backed by "
        "Elasticsearch 8.15 Pre-Filtered Hybrid Search."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    """Request payload for RBAC-filtered RAG chat."""

    question: str = Field(..., min_length=1, description="User question to ask the knowledge base.")
    user_roles: List[str] = Field(
        default_factory=list,
        description="Active roles held by the current user (e.g., ['finance'], ['operation'], or []).",
    )
    top_k: int = Field(default=4, ge=1, le=10)
    synthesis_mode: str = Field(
        default="fast",
        description="'fast' for instant citation synthesis or 'llm' for local Ollama/Gemini generation.",
    )
    retrieval_profile: str = Field(
        default="default",
        description="Retrieval profile ('default', 'precision', 'high_recall', 'executive_summary').",
    )


class UpdateTagsRequest(BaseModel):
    """Request payload for updating a document's RBAC tags without re-embedding."""

    tags: List[str] = Field(
        default_factory=list,
        description="New list of role tags (empty list makes the document PUBLIC).",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> FileResponse:
    """Serve the single-page RBAC RAG web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML not found.")
    return FileResponse(index_path)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Return Elasticsearch connection status and knowledge base statistics."""
    kb = get_kb_service()
    docs = kb.list_documents()
    return {
        "status": "ok",
        "elasticsearch_online": kb._is_es_online(),
        "elasticsearch_url": kb.es_url,
        "index_name": KB_INDEX_NAME,
        "document_count": len(docs),
        "available_tags": kb.list_available_tags(),
    }


@app.get("/api/documents")
async def get_documents() -> Dict[str, Any]:
    """List all indexed PDF documents and their RBAC tags."""
    kb = get_kb_service()
    return {
        "documents": kb.list_documents(),
        "available_tags": kb.list_available_tags(),
    }


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
) -> Dict[str, Any]:
    """Upload a PDF document, extract and chunk text, and index with RBAC tags (or public if untagged)."""
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are supported. Please upload a valid PDF document.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

    parsed_tags = normalize_tags(tags)
    kb = get_kb_service()
    try:
        doc_meta = kb.ingest_pdf_bytes(
            filename=filename,
            pdf_bytes=pdf_bytes,
            tags=parsed_tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to index PDF: {exc}"
        ) from exc

    return {
        "status": "indexed",
        "document": doc_meta,
        "available_tags": kb.list_available_tags(),
    }


@app.put("/api/documents/{doc_id}/tags")
async def update_doc_tags(doc_id: str, payload: UpdateTagsRequest) -> Dict[str, Any]:
    """Update RBAC tags on an already-indexed PDF via Elasticsearch _update_by_query."""
    kb = get_kb_service()
    updated = kb.update_document_tags(doc_id=doc_id, tags=payload.tags)
    return {
        "status": "updated",
        "document": updated,
        "available_tags": kb.list_available_tags(),
    }


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str) -> Dict[str, Any]:
    """Delete an indexed PDF document and its chunks from the knowledge base."""
    kb = get_kb_service()
    result = kb.delete_document(doc_id=doc_id)
    return {
        "status": "deleted",
        "result": result,
        "available_tags": kb.list_available_tags(),
    }


@app.post("/api/seed")
async def seed_demo_documents() -> Dict[str, Any]:
    """Ensure the 4 demo PDFs (Public, Finance, Operation, Joint Finance+Operation) are indexed."""
    kb = get_kb_service()
    seeded = kb.seed_sample_pdfs()
    return {
        "status": "seeded",
        "seeded_count": len(seeded),
        "documents": kb.list_documents(),
        "available_tags": kb.list_available_tags(),
    }


@app.post("/api/chat")
async def chat_with_kb(payload: ChatRequest) -> Dict[str, Any]:
    """Run an RBAC pre-filtered hybrid RAG query and return the grounded answer + security X-Ray."""
    kb = get_kb_service()
    return kb.query_and_chat(
        question=payload.question,
        user_roles=payload.user_roles,
        top_k=payload.top_k,
        synthesis_mode=payload.synthesis_mode,
        retrieval_profile=payload.retrieval_profile,
    )
