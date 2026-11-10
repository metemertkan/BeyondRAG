"""FastAPI application exposing PDF upload, RBAC tagging, and Pre-Filtered Hybrid RAG chat.

Who is calling comes from a verified bearer token (``src/rag/auth.py``), never from the
request. Three kinds of caller exist:

  - Tenant members: every ``/api`` data endpoint. The token names one tenant and the roles.
    ``user_roles`` or ``tenant`` sent in a body, form or query string are ignored.
  - Tenant admins: the same token with a write role. Upload, retag and delete need it.
  - Operators: erasure, source sync and tenant offboarding. They use the governance token
    and must name the tenant they act on.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag.acl import build_effective_acl_tags
from src.rag.auth import (
    DEFAULT_TOKEN_TTL_SECONDS,
    AuthNotConfiguredError,
    InvalidTokenError,
    auth_mode,
    mint_token,
    verify_token,
)
from src.rag.erasure import ErasedSourceError
from src.rag.governance import DocumentGovernance
from src.rag.pipeline import PipelineOptions
from src.rag.seeding import DEMO_ROLES, DEMO_TENANTS, seed_lab_documents
from src.rag.service import get_kb_service, normalize_tags
from src.rag.tenancy import Principal, TenantNotFoundError, TenantScopeError, require_tenant

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

# The dashboard is served from this app, so no cross-origin access is needed by default.
# Other origins must be listed. Credentials stay off: the token travels in a header that
# the page sets itself, not in a cookie the browser attaches to any request.
_CORS_ORIGINS = [o.strip() for o in os.getenv("RAG_CORS_ORIGINS", "").split(",") if o.strip()]
if _CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Governance-Token"],
    )

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    """Request payload for RBAC-filtered RAG chat."""

    # No ``user_roles`` and no ``tenant``: both come from the bearer token. If a client
    # still sends them they are dropped, not trusted.
    question: str = Field(..., min_length=1, description="User question to ask the knowledge base.")
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

    tenant: str = Field(..., min_length=1, description="The one tenant this erasure applies to.")
    subject_id: Optional[str] = Field(default=None, description="Person to erase, e.g. an email address.")
    source_key: Optional[str] = Field(default=None, description="Logical document to erase, with all its versions.")
    dry_run: bool = Field(default=True, description="Defaults to true: show the plan and delete nothing.")
    include_detected: bool = Field(
        default=True,
        description="Also erase documents where the subject was detected in the text but not declared.",
    )


class SyncEvent(BaseModel):
    """Webhook payload from a source system (SharePoint, Salesforce, ...)."""

    tenant: str = Field(..., min_length=1, description="The tenant whose source system sent the event.")
    event_id: str = Field(..., min_length=1)
    event_type: str = Field(..., description="'deleted' (hard erase) or 'archived' (remove from retrieval).")
    source_key: str = Field(..., min_length=1)


class DemoTokenRequest(BaseModel):
    """Ask for a demo persona. Only served when RAG_AUTH_MODE=demo."""

    tenant: str = Field(..., min_length=1)
    roles: List[str] = Field(default_factory=list)


class OffboardRequest(BaseModel):
    """Deleting a tenant is not reversible, so the caller repeats the tenant id."""

    confirm: str = Field(..., description="Must equal the tenant id in the URL.")


def _require_governance_token(token: Optional[str]) -> None:
    """Fail closed: destructive endpoints stay off until RAG_GOVERNANCE_TOKEN is set."""
    expected = os.getenv("RAG_GOVERNANCE_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Set RAG_GOVERNANCE_TOKEN to enable this endpoint.")
    if not token or not hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid governance token.")


def _principal(authorization: Optional[str] = Header(default=None)) -> Principal:
    """Who is calling. Every data endpoint depends on this; none of them reads identity elsewhere."""
    scheme, _, token = (authorization or "").partition(" ")
    try:
        if scheme.lower() != "bearer" or not token.strip():
            raise InvalidTokenError("Missing bearer token.")
        return verify_token(token)
    except AuthNotConfiguredError as exc:
        # No secret means no way to check anyone. Refuse; never fall back to open access.
        raise HTTPException(status_code=503, detail="Authentication is not configured.") from exc
    except InvalidTokenError as exc:
        # One generic message: the reason a token failed is not the caller's business.
        raise HTTPException(
            status_code=401,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _write_roles() -> set[str]:
    return {r.strip().lower() for r in os.getenv("RAG_WRITE_ROLES", "admin").split(",") if r.strip()}


def _writer(principal: Principal = Depends(_principal)) -> Principal:
    """Changing a tenant's documents needs a write role, not just membership.

    Without this, any member could retag a finance document as public and then read it.
    """
    expanded_roles, _ = build_effective_acl_tags(list(principal.roles))
    if not _write_roles() & set(expanded_roles):
        raise HTTPException(status_code=403, detail="This action needs a write role in your tenant.")
    return principal


def _operator_tenant(tenant: str) -> str:
    try:
        return require_tenant(tenant)
    except TenantScopeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _demo_only() -> None:
    if auth_mode() != "demo":
        raise HTTPException(status_code=404, detail="Not found.")


@app.exception_handler(TenantNotFoundError)
async def _tenant_not_found(_: Request, exc: TenantNotFoundError) -> JSONResponse:
    # Another tenant's document and a document that never existed answer the same way.
    return JSONResponse(status_code=404, content={"detail": "Document not found."})


@app.exception_handler(TenantScopeError)
async def _tenant_scope_error(_: Request, exc: TenantScopeError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


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


@app.get("/api/auth/config")
async def auth_config() -> Dict[str, Any]:
    """Tell the dashboard whether it may offer the demo persona picker."""
    mode = auth_mode()
    demo = mode == "demo"
    return {
        "mode": mode,
        "tenants": list(DEMO_TENANTS) if demo else [],
        "roles": list(DEMO_ROLES) if demo else [],
    }


@app.post("/api/auth/demo-token")
async def demo_token(payload: DemoTokenRequest) -> Dict[str, Any]:
    """Mint a short-lived token for a demo persona.

    This shows how tenancy behaves. It is not authentication: anyone may ask for any
    persona, which is why the endpoint does not exist when RAG_AUTH_MODE=required.
    """
    _demo_only()
    try:
        tenant = require_tenant(payload.tenant)
    except TenantScopeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    token = mint_token(tenant, payload.roles, subject=f"demo-{tenant}")
    principal = verify_token(token)
    expanded_roles, _ = build_effective_acl_tags(list(principal.roles))
    return {
        "token": token,
        "token_type": "bearer",
        "expires_in": DEFAULT_TOKEN_TTL_SECONDS,
        "tenant": principal.tenant,
        "roles": list(principal.roles),
        "expanded_roles": expanded_roles,
        "can_write": bool(_write_roles() & set(expanded_roles)),
    }


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Liveness only. No document counts, tags, index names or URLs: this endpoint has no caller identity."""
    kb = get_kb_service()
    return {
        "status": "ok",
        "elasticsearch_online": kb._is_es_online(),
        "auth_mode": auth_mode(),
    }


@app.get("/api/documents")
async def get_documents(
    include_superseded: bool = True,
    principal: Principal = Depends(_principal),
) -> Dict[str, Any]:
    """List the caller's tenant's documents and their RBAC tags / lifecycle status.

    SUPERSEDED versions are included by default so the dashboard can show retained
    prior revisions (they remain excluded from retrieval).
    """
    kb = get_kb_service()
    docs = kb.list_documents(tenant=principal.tenant)
    if not include_superseded:
        docs = [d for d in docs if d.get("status") != "SUPERSEDED" and d.get("is_current", True)]
    return {
        "tenant": principal.tenant,
        "documents": docs,
        "available_tags": kb.list_available_tags(tenant=principal.tenant),
    }


_ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".xlsx"}


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    source_key: str = Form(default=""),
    trust_tier: str = Form(default="unverified"),
    effective_date: str = Form(default=""),
    expires_at: str = Form(default=""),
    business_domain: str = Form(default=""),
    subject_ids: str = Form(default="", description="Comma-separated people this document is about."),
    principal: Principal = Depends(_writer),
) -> Dict[str, Any]:
    """Upload a document (PDF, Markdown, text, or Excel), chunk, and index with RBAC tags.

    The document lands in the caller's tenant. A ``tenant`` form field is ignored.
    ``source_key`` only names a document inside that tenant, so it cannot replace
    another tenant's document. Governance fields drive source precedence, expiry and
    GDPR lineage.
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
            tenant=principal.tenant,
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
        "available_tags": kb.list_available_tags(tenant=principal.tenant),
    }


@app.put("/api/documents/{doc_id}/tags")
async def update_doc_tags(
    doc_id: str,
    payload: UpdateTagsRequest,
    principal: Principal = Depends(_writer),
) -> Dict[str, Any]:
    """Update RBAC tags on an already-indexed document via Elasticsearch _update_by_query."""
    kb = get_kb_service()
    updated = kb.update_document_tags(doc_id=doc_id, tags=payload.tags, tenant=principal.tenant)
    return {
        "status": "updated",
        "document": updated,
        "available_tags": kb.list_available_tags(tenant=principal.tenant),
    }


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str, principal: Principal = Depends(_writer)) -> Dict[str, Any]:
    """Delete one of the caller's tenant's documents and its chunks."""
    kb = get_kb_service()
    result = kb.delete_document(doc_id=doc_id, tenant=principal.tenant)
    return {
        "status": "deleted",
        "result": result,
        "available_tags": kb.list_available_tags(tenant=principal.tenant),
    }


@app.post("/api/erasure")
async def erase_data(
    payload: ErasureRequest,
    background_tasks: BackgroundTasks,
    x_governance_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Plan (default) or execute a hard deletion across every store of one tenant, with verification."""
    _require_governance_token(x_governance_token)
    tenant = _operator_tenant(payload.tenant)
    if bool(payload.subject_id) == bool(payload.source_key):
        raise HTTPException(status_code=422, detail="Provide exactly one of subject_id or source_key.")
    erasure = get_kb_service().erasure_service()
    plan = (
        erasure.plan_subject(payload.subject_id, include_detected=payload.include_detected, tenant=tenant)
        if payload.subject_id
        else erasure.plan_source(payload.source_key, tenant=tenant)
    )
    if payload.dry_run:
        return {"status": "dry_run", "plan": plan.to_dict()}
    receipt = erasure.execute(plan)
    if receipt.soft_deleted_pending:
        background_tasks.add_task(erasure.confirm_physical_purge, receipt.request_id, tenant=tenant)
    return {"status": "erased" if receipt.verified else "incomplete", "receipt": receipt.to_dict()}


@app.post("/api/sync/events")
async def source_sync_event(
    payload: SyncEvent,
    background_tasks: BackgroundTasks,
    x_governance_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Receive a delete or archive event from one tenant's source system. Safe to retry."""
    _require_governance_token(x_governance_token)
    tenant = _operator_tenant(payload.tenant)
    kb = get_kb_service()
    try:
        result = kb.source_sync_handler().handle(
            event_id=payload.event_id,
            event_type=payload.event_type,
            source_key=payload.source_key,
            tenant=tenant,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    receipt = result.get("receipt") or {}
    if receipt.get("soft_deleted_pending"):
        background_tasks.add_task(
            kb.erasure_service().confirm_physical_purge, receipt["request_id"], tenant=tenant
        )
    return result


@app.post("/api/tenants/{tenant}/offboard")
async def offboard_tenant(
    tenant: str,
    payload: OffboardRequest,
    x_governance_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Delete everything one tenant stored. Operator only."""
    _require_governance_token(x_governance_token)
    scope = _operator_tenant(tenant)
    if payload.confirm.strip().lower() != scope:
        raise HTTPException(status_code=422, detail="confirm must repeat the tenant id.")
    receipt = await run_in_threadpool(get_kb_service().offboard_tenant, scope)
    return {"status": "offboarded", "receipt": receipt}


@app.post("/api/seed-lab")
async def seed_retrieval_lab(principal: Principal = Depends(_writer)) -> Dict[str, Any]:
    """Load the invented Episode 3 corpus into the caller's tenant. Demo mode only."""
    _demo_only()
    kb = get_kb_service()
    seeded = await run_in_threadpool(seed_lab_documents, kb, tenant=principal.tenant)
    return {"tenant": principal.tenant, "seeded": len(seeded), "filenames": [item["filename"] for item in seeded]}


@app.post("/api/seed")
async def seed_demo_documents(principal: Principal = Depends(_principal)) -> Dict[str, Any]:
    """Make sure each demo tenant has its sample documents. Demo mode only."""
    _demo_only()
    kb = get_kb_service()
    seeded = kb.seed_sample_pdfs()
    return {
        "status": "seeded",
        "seeded_count": len(seeded),
        "tenant": principal.tenant,
        "documents": kb.list_documents(tenant=principal.tenant),
        "available_tags": kb.list_available_tags(tenant=principal.tenant),
    }


@app.post("/api/chat")
async def chat_with_kb(payload: ChatRequest, principal: Principal = Depends(_principal)) -> Dict[str, Any]:
    """Run a tenant- and RBAC-pre-filtered hybrid RAG query and return the grounded answer + security X-Ray."""
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
        user_roles=list(principal.roles),
        top_k=payload.top_k,
        synthesis_mode=payload.synthesis_mode,
        retrieval_profile=payload.retrieval_profile,
        pipeline_options=options,
        tenant=principal.tenant,
    )
