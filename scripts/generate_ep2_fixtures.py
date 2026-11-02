#!/usr/bin/env python3
"""Regenerate Episode 2 structured PDF fixtures under tests/fixtures/pdfs/."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "pdfs"


def make_policy_v1(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title = Paragraph("HK Treasury FX & Liquidity Policy — Q4 (v1)", styles["Heading1"])
    body = Paragraph(
        "RESTRICTED — FINANCE ROLE ONLY. Group Treasury maintains clearing accounts at HSBC and "
        "Standard Chartered. The following limits apply until superseded by a later revision.",
        styles["Normal"],
    )
    data = [
        ["Limit type", "Metric", "Threshold", "Approver"],
        ["Intraday cash buffer", "Unencumbered HKD", "HKD 450,000,000", "Head of Treasury"],
        ["FX forward > USD 10M", "Notional USD", "USD 10,000,000", "CFO + Head of Treasury"],
        ["Unhedged CNY exposure", "CNY notional HKD-eq", "HKD 85,000,000", "Head of Treasury"],
        ["LCR floor", "Liquidity Coverage Ratio", "100%", "Group Risk"],
    ]
    table = Table(data, colWidths=[120, 110, 120, 130])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f7fa")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    note = Paragraph(
        "Cell values used in Episode 2 structure-fidelity tests: HKD 450,000,000; "
        "USD 10,000,000; HKD 85,000,000.",
        styles["Italic"],
    )
    doc.build([title, Spacer(1, 12), body, Spacer(1, 16), table, Spacer(1, 16), note])


def make_policy_v2(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title = Paragraph("HK Treasury FX & Liquidity Policy — Q4 (v2 REVISED)", styles["Heading1"])
    body = Paragraph(
        "RESTRICTED — FINANCE ROLE ONLY. Revision raises the intraday cash buffer after stress "
        "testing. This document supersedes v1 for the same source_key.",
        styles["Normal"],
    )
    data = [
        ["Limit type", "Metric", "Threshold", "Approver"],
        ["Intraday cash buffer", "Unencumbered HKD", "HKD 520,000,000", "Head of Treasury"],
        ["FX forward > USD 10M", "Notional USD", "USD 10,000,000", "CFO + Head of Treasury"],
        ["Unhedged CNY exposure", "CNY notional HKD-eq", "HKD 85,000,000", "Head of Treasury"],
        ["LCR floor", "Liquidity Coverage Ratio", "110%", "Group Risk"],
    ]
    table = Table(data, colWidths=[120, 110, 120, 130])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f5132")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#edf7f1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    note = Paragraph(
        "v2 markers for supersession tests: HKD 520,000,000; LCR 110%. "
        "Prior revision thresholds must not remain searchable after soft supersession.",
        styles["Italic"],
    )
    doc.build([title, Spacer(1, 12), body, Spacer(1, 16), table, Spacer(1, 16), note])


def make_multicolumn(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "Operations & Compliance Digest (Multi-column)")
    left_x, right_x = 40, width / 2 + 10
    y = height - 70
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_x, y, "Operations column")
    c.setFont("Helvetica", 9)
    left_lines = [
        "Shift handover at 07:00 HKT.",
        "Primary datacenter: TKO-1.",
        "Failover RTO target: 15 minutes.",
        "Marker TOKEN_LEFT_COLUMN_ALPHA.",
        "Pager: ops-oncall@hk-fde.dev.",
    ]
    y_l = y - 18
    for line in left_lines:
        c.drawString(left_x, y_l, line)
        y_l -= 14
    c.setFont("Helvetica-Bold", 11)
    c.drawString(right_x, y, "Compliance column")
    c.setFont("Helvetica", 9)
    right_lines = [
        "Policy owner: Chief Compliance Officer.",
        "Retention: 7 years for audit logs.",
        "Marker TOKEN_RIGHT_COLUMN_BETA.",
        "Escalation path: CCO -> Board Risk.",
        "No PII in shared ops channels.",
    ]
    y_r = y - 18
    for line in right_lines:
        c.drawString(right_x, y_r, line)
        y_r -= 14
    c.setStrokeColor(colors.grey)
    c.line(width / 2, 80, width / 2, height - 60)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        40,
        50,
        "Structure test: LEFT marker must not interleave mid-token with RIGHT marker after layout-aware parse.",
    )
    c.save()


def make_mispaired_limits_table(path: Path) -> None:
    """Finance limits table painted column-major so plain PyPDF mis-pairs label→value.

    Content stream writes all Limit-type cells first, then all Threshold cells.
    Observed PyPDF reading order clusters labels together, then values — the line
    after ``Intraday cash buffer`` is another label, not ``HKD 450,000,000``.
    pdfplumber still recovers correct row pairs via table extraction.
    """
    c = canvas.Canvas(str(path), pagesize=A4)
    _width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 50, "HK Treasury Limit Register — Q4 (synthetic)")
    c.setFont("Helvetica", 9)
    c.drawString(
        40,
        height - 68,
        "RESTRICTED — FINANCE. Synthetic limits for Episode 2 flat vs layout-aware demos.",
    )
    headers = ["Limit type", "Threshold"]
    labels = [
        "Intraday cash buffer",
        "FX forward approval floor",
        "Unhedged CNY exposure cap",
    ]
    values = [
        "HKD 450,000,000",
        "USD 10,000,000",
        "HKD 85,000,000",
    ]
    x0, x1 = 40, 310
    y0 = height - 100
    row_h = 26
    # Column-major paint: entire left column, then entire right column.
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0 + 6, y0, headers[0])
    c.setFont("Helvetica", 10)
    for i, lab in enumerate(labels):
        c.drawString(x0 + 6, y0 - (i + 1) * row_h, lab)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x1 + 6, y0, headers[1])
    c.setFont("Helvetica", 10)
    for i, val in enumerate(values):
        c.drawString(x1 + 6, y0 - (i + 1) * row_h, val)
    for i in range(4):
        y = y0 - i * row_h + 14
        c.setStrokeColor(colors.grey)
        c.rect(x0, y - row_h, x1 - x0, row_h)
        c.rect(x1, y - row_h, 220, row_h)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        40,
        60,
        "Plain-text extractors often emit labels then values; layout-aware table parse keeps row pairs.",
    )
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_policy_v1(OUT / "policy_v1_liquidity_table.pdf")
    make_policy_v2(OUT / "policy_v2_liquidity_table.pdf")
    make_multicolumn(OUT / "ops_compliance_multicolumn.pdf")
    make_mispaired_limits_table(OUT / "policy_mispaired_limits_table.pdf")
    for p in sorted(OUT.glob("*.pdf")):
        print(f"{p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
