#!/usr/bin/env python3
"""Generate the Excel and Arabic fixtures for the Episode 2 ingestion challenges.

The output is committed, so tests and CI need none of these generator dependencies:

    pip install .[fixtures] && python scripts/generate_ep2_challenge_fixtures.py

All figures, names and email addresses are invented.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import arabic_reshaper
import pypdfium2
import xlsxwriter
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "tests" / "fixtures" / "pdfs"
XLSX_DIR = ROOT / "tests" / "fixtures" / "xlsx"
# Covers Arabic presentation forms and Latin, so "HKD 520,000,000" actually renders.
ARABIC_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Logical-order source lines: the ground truth tests compare extraction against.
ARABIC_LINES = [
    "سياسة السيولة النقدية لعام 2026",
    "الحد الأدنى للسيولة النقدية هو HKD 520,000,000 يوميا.",
    "يجب على قسم الخزينة مراجعة هذا الحد كل 90 يوما.",
    "تتم الموافقة على العقود الآجلة من قبل المدير المالي.",
]


def _visual(line: str) -> str:
    """Shape and reorder like many legacy PDF producers: glyphs stored in visual order."""
    return get_display(arabic_reshaper.reshape(line))


def _arabic_pdf_bytes() -> bytes:
    pdfmetrics.registerFont(TTFont("DejaVuSans", ARABIC_FONT))
    buffer = io.BytesIO()
    page = canvas.Canvas(buffer, pagesize=A4, invariant=1)
    page.setFont("DejaVuSans", 16)
    y = 760
    for line in ARABIC_LINES:
        page.drawRightString(540, y, _visual(line))
        y -= 34
    page.showPage()
    page.save()
    return buffer.getvalue()


def write_arabic_pdfs() -> None:
    raw = _arabic_pdf_bytes()
    (PDF_DIR / "arabic_mixed_policy.pdf").write_bytes(raw)

    # A scanned copy has images only and no text layer, so it needs OCR.
    rendered = pypdfium2.PdfDocument(raw)[0].render(scale=2.5).to_pil().convert("RGB")
    buffer = io.BytesIO()
    rendered.save(buffer, format="PDF", resolution=180)
    (PDF_DIR / "arabic_scanned_policy.pdf").write_bytes(buffer.getvalue())


def write_workbook() -> None:
    path = XLSX_DIR / "regional_budget.xlsx"
    workbook = xlsxwriter.Workbook(str(path), {"default_date_format": "yyyy-mm-dd"})
    bold = workbook.add_format({"bold": True})
    center = workbook.add_format({"bold": True, "align": "center"})

    rates = workbook.add_worksheet("Rates")
    rates.write_row(0, 0, ["Currency", "To HKD"], bold)
    rates.write_row(1, 0, ["USD", 7.8])
    rates.write_row(2, 0, ["EUR", 8.4])

    budget = workbook.add_worksheet("Budget")
    budget.merge_range("A1:E1", "FY2026 Regional Budget", center)
    budget.merge_range("A2:A3", "Region", center)
    budget.merge_range("B2:B3", "Country", center)
    budget.merge_range("C2:E2", "Q1", center)
    budget.write_row(2, 2, ["USD Budget", "HKD Budget", "Owner"], bold)
    budget.merge_range("A4:A5", "Asia", bold)
    budget.write_row(3, 1, ["Hong Kong", 100000])
    budget.write_formula("D4", "=C4*Rates!B2", None, 780000)
    budget.write("E4", "maya.chen@example.com")
    budget.write_row(4, 1, ["Singapore", 50000])
    budget.write_formula("D5", "=C5*Rates!B2", None, 390000)
    budget.write("E5", "lee.tan@example.com")
    budget.merge_range("A6:B6", "Total", bold)
    budget.write_formula("C6", "=SUM(C4:C5)", None, 150000)
    budget.write_formula("D6", "=SUM(D4:D5)", None, 1170000)

    hidden = workbook.add_worksheet("Salaries_Confidential")
    hidden.write_row(0, 0, ["Employee", "Salary"], bold)
    hidden.write_row(1, 0, ["maya.chen@example.com", 999999])
    hidden.hide()

    rng = random.Random(42)
    transactions = workbook.add_worksheet("Transactions")
    transactions.write_row(0, 0, ["Txn ID", "Region", "Category", "Amount USD"], bold)
    for index in range(1, 2001):
        transactions.write_row(
            index,
            0,
            [
                f"TXN-{index:05d}",
                rng.choice(["Asia", "Europe", "Americas"]),
                rng.choice(["Travel", "Software", "Hardware", "Consulting"]),
                rng.randint(100, 9000),
            ],
        )
    workbook.close()


if __name__ == "__main__":
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    XLSX_DIR.mkdir(parents=True, exist_ok=True)
    write_arabic_pdfs()
    write_workbook()
    print("Fixtures written to", PDF_DIR, "and", XLSX_DIR)
