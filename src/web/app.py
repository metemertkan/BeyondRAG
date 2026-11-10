"""FastAPI application exposing PDF upload, RBAC tagging, and Pre-Filtered Hybrid RAG chat."""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag.erasure import ErasedSourceError
from src.rag.governance import DocumentGovernance
from src.rag.pipeline import PipelineOptions
from src.rag.seeding import seed_lab_documents
from src.rag.service import KB_INDEX_NAME, get_kb_service, normalize_tags

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
# Demo UI chrome (html lang, placeholders, quick prompts) — runtime config, not HTML hardcode.
_REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DEMO_CONFIG_PATH = Path(
    os.environ.get("RAG_UI_DEMO_CONFIG", str(_REPO_ROOT / "config" / "ui_demo.json"))
)

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
    fusion: Optional[Literal["rrf", "linear", "dense", "lexical"]] = Field(
        default=None, description="How to combine lexical and dense search. Defaults to 'rrf'."
    )
    query_mode: Optional[Literal["off", "multi", "hyde", "both"]] = Field(
        default=None, description="LLM query transformation: rewrites, HyDE, or both."
    )
    rerank: Optional[bool] = Field(default=None, description="Re-rank first-stage candidates with a cross-encoder.")
    candidates: Optional[int] = Field(default=None, ge=5, le=200, description="First-stage breadth before re-ranking.")
    multi_hop: Optional[bool] = Field(default=None, description="Decompose the question into dependent searches.")
    verify: Optional[bool] = Field(default=None, description="Check each sentence of an LLM answer against its sources.")
    trust_mode: Optional[Literal["annotate", "strip", "abstain"]] = Field(
        default=None, description="What to do with unsupported statements: show them flagged, remove them, or answer nothing."
    )


class UpdateTagsRequest(BaseModel):
    """Request payload for updating a document's RBAC tags without re-embedding."""

    tags: List[str] = Field(
        default_factory=list,
        description="New list of role tags (empty list makes the document PUBLIC).",
    )


class ErasureRequest(BaseModel):
    """Right-to-be-forgotten request. Name a person, or a source, but not both."""

    subject_id: Optional[str] = Field(default=None, description="Person to erase, e.g. an email address.")
    source_key: Optional[str] = Field(default=None, description="Logical document to erase, with all its versions.")
    dry_run: bool = Field(default=True, description="Defaults to true: show the plan and delete nothing.")
    include_detected: bool = Field(
        default=True,
        description="Also erase documents where the subject was detected in the text but not declared.",
    )


class SyncEvent(BaseModel):
    """Webhook payload from a source system (SharePoint, Salesforce, ...)."""

    event_id: str = Field(..., min_length=1)
    event_type: str = Field(..., description="'deleted' (hard erase) or 'archived' (remove from retrieval).")
    source_key: str = Field(..., min_length=1)


def _require_governance_token(token: Optional[str]) -> None:
    """Fail closed: destructive endpoints stay off until RAG_GOVERNANCE_TOKEN is set."""
    expected = os.getenv("RAG_GOVERNANCE_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Set RAG_GOVERNANCE_TOKEN to enable this endpoint.")
    if not token or not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid governance token.")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> FileResponse:
    """Serve the single-page RBAC RAG web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML not found.")
    return FileResponse(index_path)


@app.get("/api/ui-demo-config")
async def ui_demo_config() -> Dict[str, Any]:
    """Return dashboard demo chrome from config/ui_demo.json (overridable via RAG_UI_DEMO_CONFIG)."""
    path = UI_DEMO_CONFIG_PATH
    if not path.exists():
        return {
            "html_lang": "",
            "chat_placeholder": "Ask the knowledge base…",
            "demo_prompts": [],
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HTTPException(status_code=500, detail="ui_demo.json must be a JSON object")
    return {
        "html_lang": str(raw.get("html_lang") or ""),
        "chat_placeholder": str(raw.get("chat_placeholder") or ""),
        "demo_prompts": list(raw.get("demo_prompts") or []),
    }


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
async def get_documents(include_superseded: bool = True) -> Dict[str, Any]:
    """List knowledge-base documents and their RBAC tags / lifecycle status.

    SUPERSEDED versions are included by default so the dashboard can show retained
    prior revisions (they remain excluded from retrieval).
    """
    kb = get_kb_service()
    docs = kb.list_documents()
    if not include_superseded:
        docs = [d for d in docs if d.get("status") != "SUPERSEDED" and d.get("is_current", True)]
    return {
        "documents": docs,
        "available_tags": kb.list_available_tags(),
    }


_ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".xlsx"}


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    tenant: str = Form(default="default"),
    source_key: str = Form(default=""),
    trust_tier: str = Form(default="unverified"),
    effective_date: str = Form(default=""),
    expires_at: str = Form(default=""),
    business_domain: str = Form(default=""),
    subject_ids: str = Form(default="", description="Comma-separated people this document is about."),
) -> Dict[str, Any]:
    """Upload a document (PDF, Markdown, text, or Excel), chunk, and index with RBAC tags.

    Governance fields drive source precedence, expiry and GDPR lineage.
    """
    filename = file.filename or "uploaded_document.pdf"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Supported uploads: .pdf, .md, .txt, .xlsx (legacy .xls: re-save as .xlsx first).",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    parsed_tags = normalize_tags(tags)
    kb = get_kb_service()
    try:
        governance = DocumentGovernance(
            trust_tier=trust_tier.strip() or "unverified",
            effective_date=effective_date.strip() or None,
            expires_at=expires_at.strip() or None,
            business_domain=business_domain.strip(),
            subject_ids=tuple(s for s in subject_ids.split(",") if s.strip()),
        )
        doc_meta = kb.ingest_pdf_bytes(
            filename=filename,
            pdf_bytes=pdf_bytes,
            tags=parsed_tags,
            tenant=tenant or "default",
            source_key=source_key.strip() or None,
            governance=governance,
        )
    except ErasedSourceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to index document: {exc}"
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


@app.post("/api/erasure")
async def erase_data(
    payload: ErasureRequest,
    background_tasks: BackgroundTasks,
    x_governance_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Plan (default) or execute a hard deletion across every store, with verification."""
    _require_governance_token(x_governance_token)
    if bool(payload.subject_id) == bool(payload.source_key):
        raise HTTPException(status_code=422, detail="Provide exactly one of subject_id or source_key.")
    erasure = get_kb_service().erasure_service()
    plan = (
        erasure.plan_subject(payload.subject_id, include_detected=payload.include_detected)
        if payload.subject_id
        else erasure.plan_source(payload.source_key)
    )
    if payload.dry_run:
        return {"status": "dry_run", "plan": plan.to_dict()}
    receipt = erasure.execute(plan)
    if receipt.soft_deleted_pending:
        background_tasks.add_task(erasure.confirm_physical_purge, receipt.request_id)
    return {"status": "erased" if receipt.verified else "incomplete", "receipt": receipt.to_dict()}


@app.post("/api/sync/events")
async def source_sync_event(
    payload: SyncEvent,
    background_tasks: BackgroundTasks,
    x_governance_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Receive a delete or archive event from a source system. Safe to retry."""
    _require_governance_token(x_governance_token)
    kb = get_kb_service()
    try:
        result = kb.source_sync_handler().handle(
            event_id=payload.event_id,
            event_type=payload.event_type,
            source_key=payload.source_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    receipt = result.get("receipt") or {}
    if receipt.get("soft_deleted_pending"):
        background_tasks.add_task(kb.erasure_service().confirm_physical_purge, receipt["request_id"])
    return result


@app.post("/api/seed-lab")
async def seed_retrieval_lab() -> Dict[str, Any]:
    """Load the invented Episode 3 corpus used by the Retrieval Lab demo prompts."""
    kb = get_kb_service()
    seeded = await run_in_threadpool(seed_lab_documents, kb)
    return {"seeded": len(seeded), "filenames": [item["filename"] for item in seeded]}


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
    defaults = PipelineOptions.from_env()
    options = PipelineOptions(
        fusion=payload.fusion,
        query_mode=payload.query_mode or defaults.query_mode,
        rerank=defaults.rerank if payload.rerank is None else payload.rerank,
        candidates=payload.candidates or defaults.candidates,
        multi_hop=defaults.multi_hop if payload.multi_hop is None else payload.multi_hop,
        verify=defaults.verify if payload.verify is None else payload.verify,
        verify_with_llm=defaults.verify_with_llm,
        trust_mode=payload.trust_mode or defaults.trust_mode,
        min_relevance=defaults.min_relevance,
    )
    # The LLM and model calls block, so keep them off the event loop.
    return await run_in_threadpool(
        kb.query_and_chat,
        question=payload.question,
        user_roles=payload.user_roles,
        top_k=payload.top_k,
        synthesis_mode=payload.synthesis_mode,
        retrieval_profile=payload.retrieval_profile,
        pipeline_options=options,
    )
