"""Elasticsearch 8.15 Pre-Filtered RBAC & Cross-Lingual (English <-> Chinese) RAG Service."""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List

from elasticsearch import Elasticsearch
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.config import ELASTICSEARCH_URL, WORKSPACE_DIR, get_llm
from src.rag.embeddings import (
    EMBEDDING_DIMS,
    STOPWORDS,
    embed_text,
    expand_cross_lingual_query,
    has_cjk,
)

KB_INDEX_NAME: Final[str] = "hk-rbac-rag-kb"
PUBLIC_SENTINEL_TAG: Final[str] = "__public__"
UPLOADS_DIR: Final[Path] = WORKSPACE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_tags(raw_tags: List[str] | str | None) -> List[str]:
    """Normalize a list or comma-separated string of RBAC tags.

    Returns an empty list if no restricted tags are present (meaning the document is PUBLIC).
    """
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        parts = [p.strip().lower() for p in raw_tags.replace(";", ",").split(",")]
    else:
        parts = [str(p).strip().lower() for p in raw_tags]

    cleaned: list[str] = []
    for tag in parts:
        if not tag or tag in {"public", "__public__", "none", "untagged"}:
            continue
        sanitized = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
        if sanitized and sanitized not in cleaned:
            cleaned.append(sanitized)
    return cleaned


def _chunk_text(text: str, chunk_size: int = 650, overlap: int = 100) -> List[str]:
    """Split page text into overlapping semantic paragraphs/chunks."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_pos = normalized.rfind(". ", start, end)
            if split_pos == -1 or split_pos <= start + chunk_size // 2:
                split_pos = normalized.rfind(" ", start, end)
            if split_pos != -1 and split_pos > start:
                end = split_pos + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _create_pdf_bytes(title: str, pages: List[List[str]]) -> bytes:
    """Generate a valid multi-page English PDF document in memory using ReportLab."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    for page_idx, paragraphs in enumerate(pages, start=1):
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, height - 55, f"{title} (Page {page_idx})")
        pdf.setFont("Helvetica", 10.5)
        y_pos = height - 90
        for para in paragraphs:
            words = para.split()
            line = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if len(candidate) > 82:
                    pdf.drawString(50, y_pos, line)
                    y_pos -= 16
                    line = word
                else:
                    line = candidate
            if line:
                pdf.drawString(50, y_pos, line)
                y_pos -= 22
        pdf.showPage()

    pdf.save()
    return buffer.getvalue()


class RBACKnowledgeBaseService:
    """Manages PDF ingestion, RBAC tag pre-filtering, and Cross-Lingual Hybrid Search in ES 8.15."""

    def __init__(self, es_url: str = ELASTICSEARCH_URL) -> None:
        self.es_url = es_url
        self.es = Elasticsearch(self.es_url, request_timeout=5, max_retries=1)
        self._memory_chunks: List[Dict[str, Any]] = []
        self._ensure_index()

    def _is_es_online(self) -> bool:
        try:
            return bool(self.es.ping())
        except Exception:
            return False

    def _ensure_index(self) -> None:
        if not self._is_es_online():
            return
        if not self.es.indices.exists(index=KB_INDEX_NAME):
            self.es.indices.create(
                index=KB_INDEX_NAME,
                settings={
                    "index.search.slowlog.threshold.query.warn": "0ms",
                    "index.search.slowlog.threshold.fetch.warn": "0ms",
                },
                mappings={
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "title": {"type": "text"},
                        "page_number": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "content": {"type": "text"},
                        "is_public": {"type": "boolean"},
                        "acl_tags": {"type": "keyword"},
                        "uploaded_at": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIMS,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "int8_hnsw",
                                "m": 16,
                                "ef_construction": 100,
                            },
                        },
                    }
                },
            )

    def ingest_pdf_bytes(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: str | None = None,
    ) -> Dict[str, Any]:
        """Extract text from a PDF, embed into the cross-lingual space, and index with RBAC tags."""
        normalized_tags = normalize_tags(tags)
        is_public = len(normalized_tags) == 0
        acl_tags = [PUBLIC_SENTINEL_TAG] if is_public else normalized_tags

        assigned_doc_id = doc_id or f"doc-{uuid.uuid4().hex[:10]}"
        safe_filename = Path(filename).name
        saved_path = UPLOADS_DIR / f"{assigned_doc_id}_{safe_filename}"
        saved_path.write_bytes(pdf_bytes)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        uploaded_at = _utc_now()

        chunk_records: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for page_num, page in enumerate(reader.pages, start=1):
            extracted_text = (page.extract_text() or "").strip()
            if not extracted_text:
                continue
            page_chunks = _chunk_text(extracted_text)
            for chunk_str in page_chunks:
                global_chunk_idx += 1
                chunk_id = f"{assigned_doc_id}_p{page_num}_c{global_chunk_idx}"
                enriched_text = f"{safe_filename} (Page {page_num}): {chunk_str}"
                vector = embed_text(enriched_text)
                record = {
                    "chunk_id": chunk_id,
                    "doc_id": assigned_doc_id,
                    "filename": safe_filename,
                    "title": safe_filename,
                    "page_number": page_num,
                    "chunk_index": global_chunk_idx,
                    "content": chunk_str,
                    "is_public": is_public,
                    "acl_tags": acl_tags,
                    "uploaded_at": uploaded_at,
                    "embedding": vector,
                }
                chunk_records.append(record)

        if not chunk_records:
            raise ValueError(
                f"No extractable text found in '{safe_filename}'. Ensure the PDF contains selectable text."
            )

        if self._is_es_online():
            self._ensure_index()
            for rec in chunk_records:
                self.es.index(
                    index=KB_INDEX_NAME,
                    id=rec["chunk_id"],
                    document=rec,
                )
            self.es.indices.refresh(index=KB_INDEX_NAME)
        else:
            self._memory_chunks.extend(chunk_records)

        return {
            "doc_id": assigned_doc_id,
            "filename": safe_filename,
            "page_count": total_pages,
            "chunk_count": len(chunk_records),
            "is_public": is_public,
            "tags": [] if is_public else normalized_tags,
            "acl_tags": acl_tags,
            "uploaded_at": uploaded_at,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a deduplicated list of all uploaded PDF documents and their RBAC tags."""
        docs_by_id: Dict[str, Dict[str, Any]] = {}

        if self._is_es_online():
            self._ensure_index()
            resp = self.es.search(
                index=KB_INDEX_NAME,
                size=1000,
                _source=[
                    "doc_id",
                    "filename",
                    "page_number",
                    "is_public",
                    "acl_tags",
                    "uploaded_at",
                ],
                query={"match_all": {}},
            )
            hits = [h["_source"] for h in resp["hits"]["hits"]]
        else:
            hits = self._memory_chunks

        for item in hits:
            d_id = item["doc_id"]
            raw_acl = item.get("acl_tags", [PUBLIC_SENTINEL_TAG])
            visible_tags = [t for t in raw_acl if t != PUBLIC_SENTINEL_TAG]
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
        baseline = {"finance", "operation"}
        for doc in self.list_documents():
            for tag in doc.get("tags", []):
                baseline.add(tag)
        return sorted(baseline)

    def update_document_tags(
        self,
        doc_id: str,
        tags: List[str] | str | None,
    ) -> Dict[str, Any]:
        """Update a document's RBAC tags in-place via Elasticsearch _update_by_query (zero re-embedding)."""
        normalized_tags = normalize_tags(tags)
        is_public = len(normalized_tags) == 0
        acl_tags = [PUBLIC_SENTINEL_TAG] if is_public else normalized_tags

        if self._is_es_online():
            self._ensure_index()
            self.es.update_by_query(
                index=KB_INDEX_NAME,
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
            for rec in self._memory_chunks:
                if rec["doc_id"] == doc_id:
                    rec["acl_tags"] = acl_tags
                    rec["is_public"] = is_public

        return {
            "doc_id": doc_id,
            "is_public": is_public,
            "tags": [] if is_public else normalized_tags,
            "acl_tags": acl_tags,
        }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document and all its indexed chunks from Elasticsearch."""
        if self._is_es_online():
            self._ensure_index()
            self.es.delete_by_query(
                index=KB_INDEX_NAME,
                query={"term": {"doc_id": doc_id}},
                refresh=True,
            )
        self._memory_chunks = [
            c for c in self._memory_chunks if c.get("doc_id") != doc_id
        ]
        for file_path in UPLOADS_DIR.glob(f"{doc_id}_*"):
            file_path.unlink(missing_ok=True)
        return {"deleted": True, "doc_id": doc_id}

    def reindex_all_sample_pdfs(self) -> List[Dict[str, Any]]:
        """Clear and re-seed the 4 English PDF documents with updated cross-lingual vectors."""
        if self._is_es_online() and self.es.indices.exists(index=KB_INDEX_NAME):
            self.es.indices.delete(index=KB_INDEX_NAME)
        self._memory_chunks.clear()
        self._ensure_index()
        return self.seed_sample_pdfs()

    def seed_sample_pdfs(self) -> List[Dict[str, Any]]:
        """Seed 4 English enterprise PDFs (Public, Finance, Operation, and Joint Finance+Operation)."""
        existing_filenames = {d["filename"] for d in self.list_documents()}
        seeded: List[Dict[str, Any]] = []

        samples = [
            {
                "filename": "HK_Enterprise_Public_Handbook_2026.pdf",
                "tags": [],  # Untagged -> PUBLIC (accessible to everyone)
                "pages": [
                    [
                        "1. General Workplace & Hybrid Attendance Policy (Public Knowledge Base): "
                        "All Hong Kong headquarters employees at Two International Finance Centre (2 IFC) "
                        "operate on a core hours schedule from 09:30 HKT to 18:00 HKT Monday through Friday.",
                        "2. Severe Weather & HK Observatory Typhoon Signal No. 8 / Black Rainstorm Protocol: "
                        "When Typhoon Signal No. 8 or a Black Rainstorm Warning is hoisted by the Hong Kong "
                        "Observatory, physical office attendance is suspended and remote work mode activates "
                        "automatically in accordance with Labour Department guidelines.",
                        "3. IT Service Desk & Public Wi-Fi Guest Access: Employees and visitors may connect "
                        "to the 'HK-FDE-Guest' network. For hardware support, contact helpdesk@hk-fde.dev "
                        "or dial extension 8800.",
                    ]
                ],
            },
            {
                "filename": "HK_Treasury_FX_Liquidity_Policy_Q4.pdf",
                "tags": ["finance"],
                "pages": [
                    [
                        "RESTRICTED — FINANCE & TREASURY ROLE ONLY. "
                        "1. HKMA Liquidity Coverage Ratio (LCR) & Minimum Cash Buffer: "
                        "The Group Treasury division must maintain an intraday minimum unencumbered cash "
                        "reserve of HKD 450,000,000 (450M HKD) across HSBC (0005.HK) and Standard Chartered "
                        "clearing accounts at all times.",
                        "2. Foreign Exchange (FX) Hedging & USD/HKD Peg Band Tolerance: "
                        "Any FX forward contract exceeding USD 10,000,000 requires sign-off from the Chief "
                        "Financial Officer (CFO) and Head of Treasury. Unhedged CNY exposure is capped at "
                        "HKD 85,000,000.",
                        "3. Quarterly Net Interest Margin (NIM) & Covenant Reporting: "
                        "Finance analysts must submit the Q4 audited balance sheet and covenant compliance "
                        "certificate by the 5th business day of each quarter.",
                    ]
                ],
            },
            {
                "filename": "Global_Supply_Chain_Terminal_SOP.pdf",
                "tags": ["operation"],
                "pages": [
                    [
                        "RESTRICTED — OPERATIONS & LOGISTICS ROLE ONLY. "
                        "1. Kwai Tsing Container Terminal 9 Dispatch SLA: "
                        "Operations supervisors must maintain a 99.4% on-time vessel loading and gate-out "
                        "dispatch rate. Any container dwell time exceeding 36 hours triggers an automatic "
                        "Tier-2 escalation to the Director of Operations.",
                        "2. Cold-Chain Pharmaceutical & High-Value Air Freight Protocol: "
                        "Reefer containers transiting Hong Kong International Airport (HKIA) Cargo Terminal "
                        "must maintain continuous IoT telemetry between +2.0 C and +8.0 C. Excursion alarms "
                        "require physical inspection within 15 minutes.",
                        "3. Cross-Boundary Fleet & Shenzhen Bay Port Customs Manifests: "
                        "Dual-plate logistics trucks must complete electronic Road Cargo System (ROCARS) "
                        "customs clearance 4 hours prior to border crossing.",
                    ]
                ],
            },
            {
                "filename": "Cross_Border_CapEx_Procurement_Matrix.pdf",
                "tags": ["finance", "operation"],
                "pages": [
                    [
                        "SHARED RBAC — ACCESSIBLE TO EITHER 'FINANCE' OR 'OPERATION' ROLES. "
                        "1. Greater Bay Area (GBA) Warehouse Automation CapEx Thresholds: "
                        "Capital expenditure requisitions for autonomous mobile robots (AMRs) and warehouse "
                        "conveyors between HKD 500,000 and HKD 2,500,000 can be initiated by either the "
                        "Operations Lead or Finance Business Partner.",
                        "2. Emergency Vendor Expedite Fee Authorization: "
                        "Expedited air-charter surcharges up to HKD 350,000 are pre-authorized under the "
                        "joint Finance-Operations contingency budget code CAPEX-GBA-2026.",
                    ]
                ],
            },
        ]

        for sample in samples:
            if sample["filename"] in existing_filenames:
                continue
            pdf_bytes = _create_pdf_bytes(sample["filename"], sample["pages"])
            meta = self.ingest_pdf_bytes(
                filename=sample["filename"],
                pdf_bytes=pdf_bytes,
                tags=sample["tags"],
            )
            seeded.append(meta)

        return seeded

    def _translate_unseen_chinese_query(self, question: str) -> str:
        """Use the local LLM router to translate unseen Chinese queries into English keywords when needed."""
        try:
            llm = get_llm(role="manager", temperature=0.0)
            resp = llm.invoke(
                f"Translate this Chinese query into concise English search keywords only (no explanation): {question}"
            )
            translated = str(getattr(resp, "content", resp)).strip()
            return f"{question} {translated}" if translated else question
        except Exception:
            return question

    def query_and_chat(
        self,
        question: str,
        user_roles: List[str] | str | None = None,
        top_k: int = 4,
        synthesis_mode: str = "fast",
    ) -> Dict[str, Any]:
        """Execute Pre-Filtered Cross-Lingual Hybrid Search in ES 8.15 and synthesize an answer."""
        cleaned_roles = normalize_tags(user_roles)
        effective_acl_tags = [PUBLIC_SENTINEL_TAG] + cleaned_roles
        is_cjk_query = has_cjk(question)

        # Cross-lingual query alignment (bridges Chinese queries -> English PDF documents)
        expanded_query, matched_concepts = expand_cross_lingual_query(question)
        if is_cjk_query and not matched_concepts and synthesis_mode == "llm":
            expanded_query = self._translate_unseen_chinese_query(question)

        query_vector = embed_text(expanded_query)
        acl_filter_clause = {"terms": {"acl_tags": effective_acl_tags}}

        es_query_dsl = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": expanded_query, "boost": 2.0}}},
                        {"match": {"title": {"query": expanded_query, "boost": 1.2}}},
                    ],
                    "filter": [acl_filter_clause],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": "<256-dim Cross-Lingual EN<->ZH L2-normalized vector>",
                "k": top_k,
                "num_candidates": max(20, top_k * 4),
                "filter": acl_filter_clause,
            },
        }

        t0 = time.perf_counter()
        authorized_chunks: List[Dict[str, Any]] = []
        masked_restricted_docs: List[Dict[str, Any]] = []

        if self._is_es_online():
            self._ensure_index()
            resp = self.es.search(
                index=KB_INDEX_NAME,
                size=top_k,
                query={
                    "bool": {
                        "should": [
                            {"match": {"content": {"query": expanded_query, "boost": 2.0}}},
                            {"match": {"title": {"query": expanded_query, "boost": 1.2}}},
                        ],
                        "filter": [acl_filter_clause],
                    }
                },
                knn={
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": max(20, top_k * 4),
                    "filter": acl_filter_clause,
                },
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                raw_acl = src.get("acl_tags", [PUBLIC_SENTINEL_TAG])
                visible_tags = [t for t in raw_acl if t != PUBLIC_SENTINEL_TAG]
                authorized_chunks.append(
                    {
                        "chunk_id": src.get("chunk_id"),
                        "doc_id": src.get("doc_id"),
                        "filename": src.get("filename"),
                        "page_number": src.get("page_number", 1),
                        "content": src.get("content", ""),
                        "is_public": bool(src.get("is_public", len(visible_tags) == 0)),
                        "tags": visible_tags,
                        "score": round(float(hit.get("_score") or 0.0), 4),
                    }
                )

            # 2. Audit restricted documents masked by the RBAC Pre-Filter (using expanded bilingual keywords)
            domain_keywords = " ".join(
                w.strip("?.,!\"'()？，。！")
                for w in expanded_query.split()
                if len(w.strip("?.,!\"'()？，。！")) >= 3
                and w.lower().strip("?.,!\"'()？，。！") not in STOPWORDS
                and not has_cjk(w)
            )
            unfiltered_resp = self.es.search(
                index=KB_INDEX_NAME,
                size=10,
                _source=["doc_id", "filename", "page_number", "acl_tags", "is_public"],
                query={
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": domain_keywords or expanded_query,
                                        "minimum_should_match": "2<60%",
                                    }
                                }
                            }
                        ],
                        "must_not": [acl_filter_clause],
                    }
                },
            )
            seen_masked = set()
            for hit in unfiltered_resp["hits"]["hits"]:
                src = hit["_source"]
                key = (src.get("filename"), src.get("page_number"))
                if key in seen_masked:
                    continue
                seen_masked.add(key)
                req_tags = [
                    t for t in src.get("acl_tags", []) if t != PUBLIC_SENTINEL_TAG
                ]
                masked_restricted_docs.append(
                    {
                        "filename": src.get("filename"),
                        "page_number": src.get("page_number", 1),
                        "required_any_of_tags": req_tags,
                    }
                )
        else:
            for rec in self._memory_chunks:
                rec_tags = set(rec.get("acl_tags", [PUBLIC_SENTINEL_TAG]))
                if rec_tags.intersection(effective_acl_tags):
                    score = sum(
                        a * b for a, b in zip(query_vector, rec["embedding"])
                    )
                    visible_tags = [t for t in rec_tags if t != PUBLIC_SENTINEL_TAG]
                    authorized_chunks.append(
                        {
                            "chunk_id": rec["chunk_id"],
                            "doc_id": rec["doc_id"],
                            "filename": rec["filename"],
                            "page_number": rec["page_number"],
                            "content": rec["content"],
                            "is_public": bool(rec.get("is_public", len(visible_tags) == 0)),
                            "tags": visible_tags,
                            "score": round(float(score), 4),
                        }
                    )
            authorized_chunks.sort(key=lambda c: c["score"], reverse=True)
            authorized_chunks = authorized_chunks[:top_k]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        answer = self._synthesize_answer(
            question=question,
            expanded_query=expanded_query,
            matched_concepts=matched_concepts,
            is_cjk_query=is_cjk_query,
            user_roles=cleaned_roles,
            authorized_chunks=authorized_chunks,
            masked_restricted_docs=masked_restricted_docs,
            synthesis_mode=synthesis_mode,
        )

        return {
            "question": question,
            "is_cross_lingual": is_cjk_query,
            "matched_concepts": matched_concepts,
            "expanded_query": expanded_query,
            "user_roles": cleaned_roles,
            "effective_acl_filter": effective_acl_tags,
            "latency_ms": elapsed_ms,
            "authorized_chunks": authorized_chunks,
            "masked_restricted_docs": masked_restricted_docs,
            "es_query_dsl": es_query_dsl,
            "answer": answer,
        }

    def _synthesize_answer(
        self,
        question: str,
        expanded_query: str,
        matched_concepts: List[str],
        is_cjk_query: bool,
        user_roles: List[str],
        authorized_chunks: List[Dict[str, Any]],
        masked_restricted_docs: List[Dict[str, Any]],
        synthesis_mode: str = "fast",
    ) -> str:
        """Synthesize an RBAC-aware and Cross-Lingual-aware response grounded in authorized chunks."""
        role_display = (
            ", ".join(f"`{r}`" for r in user_roles)
            if user_roles
            else "`None (Public Knowledge Base Only)`"
        )

        extra_ignored = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "how",
            "why",
            "policy",
            "document",
            "limit",
            "rate",
            "minimum",
            "maximum",
        }
        q_tokens = {
            w.lower().strip("?.,!\"'()？，。！")
            for w in expanded_query.split()
            if len(w.strip("?.,!\"'()？，。！")) >= 3
            and w.lower().strip("?.,!\"'()？，。！") not in STOPWORDS
            and w.lower().strip("?.,!\"'()？，。！") not in extra_ignored
            and not has_cjk(w)
        }

        relevant_chunks = [
            c
            for c in authorized_chunks
            if not q_tokens
            or sum(
                1
                for tok in q_tokens
                if tok in c["content"].lower() or tok in c["filename"].lower()
            )
            >= max(1, min(2, len(q_tokens)))
        ]

        if not relevant_chunks:
            if masked_restricted_docs:
                locked_doc = masked_restricted_docs[0]["filename"]
                req_role_str = ", ".join(masked_restricted_docs[0]["required_any_of_tags"])
                if is_cjk_query:
                    return (
                        f"🔒 **RBAC 權限攔截 (Access Blocked)**\n"
                        f"📄 `{locked_doc}` — Required Role: **`{req_role_str}`**"
                    )
                return (
                    f"🔒 **Access Blocked by RBAC Pre-Filter**\n"
                    f"📄 `{locked_doc}` — Required Role: **`{req_role_str}`**"
                )
            return f"No matching documents for role {role_display}."

        target_chunks = relevant_chunks

        if synthesis_mode == "llm":
            try:
                context_str = "\n\n".join(
                    f"Source: {c['filename']} (Page {c['page_number']}, Tags: {c['tags'] or ['PUBLIC']})\n"
                    f"Content: {c['content']}"
                    for c in target_chunks
                )
                lang_instruction = (
                    "Respond in Traditional Chinese in 2 concise bullet points citing [filename.pdf, p. 1]."
                    if is_cjk_query
                    else "Respond in English in 2 concise bullet points citing [filename.pdf, p. 1]."
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are the BeyondRAG enterprise assistant. Answer concisely using ONLY "
                            f"the authorized chunks. {lang_instruction}",
                        ),
                        (
                            "human",
                            "Active User Roles: {roles}\n\n"
                            "Authorized Context:\n{context}\n\n"
                            "Question: {question}",
                        ),
                    ]
                )
                llm = get_llm(role="architect", temperature=0.1)
                chain = prompt | llm
                resp = chain.invoke(
                    {
                        "roles": ", ".join(user_roles) if user_roles else "Public Only",
                        "context": context_str,
                        "question": question,
                    }
                )
                content = str(getattr(resp, "content", resp)).strip()
                if content:
                    return content
            except Exception:
                synthesis_mode = "fast"

        top = target_chunks[0]
        tag_label = "PUBLIC" if top["is_public"] else ", ".join(top["tags"])
        raw_text = top["content"]
        if "(Page 1)" in raw_text:
            raw_text = raw_text.split("(Page 1)", 1)[-1].strip()
        for prefix in (
            "RESTRICTED — FINANCE & TREASURY ROLE ONLY.",
            "RESTRICTED — OPERATIONS & SUPPLY CHAIN ROLE ONLY.",
            "RESTRICTED — JOINT FINANCE & OPERATIONS ROLES.",
            "RESTRICTED FINANCE DOCUMENT:",
        ):
            raw_text = raw_text.replace(prefix, "").strip()

        if is_cjk_query:
            return (
                f"✅ **跨語言檢索成功 (ZH → EN PDF Unlocked)**\n"
                f"📄 `{top['filename']}` *(Role: `{tag_label}`)*\n"
                f"💡 {raw_text[:155]}..."
            )

        return (
            f"✅ **Authorized Match Retrieved**\n"
            f"📄 `{top['filename']}` *(Role: `{tag_label}`)*\n"
            f"💡 {raw_text[:155]}..."
        )


_KB_SERVICE_INSTANCE: RBACKnowledgeBaseService | None = None


def get_kb_service() -> RBACKnowledgeBaseService:
    """Return the singleton RBACKnowledgeBaseService instance and ensure sample PDFs are indexed."""
    global _KB_SERVICE_INSTANCE
    if _KB_SERVICE_INSTANCE is None:
        _KB_SERVICE_INSTANCE = RBACKnowledgeBaseService()
        _KB_SERVICE_INSTANCE.reindex_all_sample_pdfs()
    return _KB_SERVICE_INSTANCE
