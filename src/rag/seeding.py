"""Component 7: Demo Document Seeding (`DemoDocumentSeeder`).

Generates the 4 enterprise sample PDFs in memory using ReportLab and seeds them
into the RAG knowledge base without coupling demo fixtures to the domain service.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Final, List, Protocol

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


class _SeedTarget(Protocol):
    def list_documents(self) -> List[Dict[str, Any]]:
        ...

    def ingest_pdf_bytes(
        self,
        filename: str,
        pdf_bytes: bytes,
        tags: List[str] | str | None = None,
        doc_id: str | None = None,
    ) -> Dict[str, Any]:
        ...


def create_pdf_bytes(title: str, pages: List[List[str]]) -> bytes:
    """Generate a valid multi-page English PDF document in memory using ReportLab."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4

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


DEMO_ENTERPRISE_SAMPLES: Final[List[Dict[str, Any]]] = [
    {
        "filename": "HK_Enterprise_Public_Handbook_2026.pdf",
        "tags": [],
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


class DemoDocumentSeeder:
    """Seeds the 4 sample enterprise PDFs into any target implementing `_SeedTarget`."""

    def seed(self, service: _SeedTarget) -> List[Dict[str, Any]]:
        existing_filenames = {d["filename"] for d in service.list_documents()}
        seeded: List[Dict[str, Any]] = []

        for sample in DEMO_ENTERPRISE_SAMPLES:
            if sample["filename"] in existing_filenames:
                continue
            pdf_bytes = create_pdf_bytes(sample["filename"], sample["pages"])
            meta = service.ingest_pdf_bytes(
                filename=sample["filename"],
                pdf_bytes=pdf_bytes,
                tags=sample["tags"],
            )
            seeded.append(meta)

        return seeded
