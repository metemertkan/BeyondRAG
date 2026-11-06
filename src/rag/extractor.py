"""Component 2: Pluggable Document Parsers.

Production Episode 2 stack:
  - PdfPlumberDocumentParser: primary layout-aware PDF adapter (tables, columns)
  - PyPdfDocumentParser: lightweight offline Docker fallback
  - PlainTextDocumentParser: .txt / .md / .markdown
  - XlsxDocumentParser: merged cells, formula results, cross-sheet lineage, large-sheet routing
  - DocumentParserRegistry: extension-based dispatch

Trade-offs (also in README):
  Docling — strong structure fidelity, heavier image/ML deps, larger Docker image,
  slower cold start; deferred for sandbox size.
  pdfplumber — MIT, pure-ish PDF layout/tables via pdfminer, modest image size,
  works offline once installed; chosen as primary production-grade adapter.
  PyMuPDF — fast/capable but AGPL licensing friction for some deployments.
  PyPDF — tiny fallback when pdfplumber is unavailable; plain extract_text only.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from pypdf import PdfReader

from src.rag.models import ParsedBlock, ParsedDocument, ParsedTable
from src.rag.multilingual import dominant_direction, has_rtl, repair_rtl_text
from src.rag.ocr import OcrEngine, create_ocr_engine_from_env, order_ocr_lines

logger = logging.getLogger(__name__)


class ParserDegradedError(RuntimeError):
    """Raised when layout-aware parse fails and degraded fallback is not allowed."""

try:
    import pdfplumber

    _PDFPLUMBER_AVAILABLE = True
except Exception:  # pragma: no cover - optional at import time in slim images
    pdfplumber = None  # type: ignore
    _PDFPLUMBER_AVAILABLE = False


class DocumentParser(Protocol):
    """Protocol for pluggable document parsers."""

    def can_parse(self, filename: str) -> bool:
        ...

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        """Return (total_pages, list of (page_number, page_text))."""
        ...


class StructuredDocumentParser(DocumentParser, Protocol):
    """Parsers that emit structure-preserving blocks."""

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        ...


_MIN_GUTTER_POINTS = 10.0


def _table_to_markdown(table: Sequence[Optional[Sequence[Optional[str]]]]) -> str:
    """Render a pdfplumber table as markdown-like rows (cell text preserved)."""
    rows: List[List[str]] = []
    for raw_row in table:
        if raw_row is None:
            continue
        cells = [("" if c is None else repair_rtl_text(str(c).strip()).replace("\n", " ")) for c in raw_row]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized = [r + [""] * (width - len(r)) for r in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _has_column_gutter(left: List[Dict[str, object]], right: List[Dict[str, object]]) -> bool:
    """True when rows crossing the midpoint show a real gap, not an ordinary word space.

    Centroids alone misread right-aligned RTL text as two columns: words land on both
    sides of the midpoint with nothing separating them. On rows that have words on both
    sides, the typical gap must be several times wider than a word space.
    """
    rows: Dict[float, List[float]] = {}
    for word in left:
        key = round(float(word.get("top", 0) or 0), 0)
        rows.setdefault(key, [0.0, float("inf")])
        rows[key][0] = max(rows[key][0], float(word.get("x1", 0) or 0))
    gaps: List[float] = []
    for word in right:
        key = round(float(word.get("top", 0) or 0), 0)
        if key in rows:
            rows[key][1] = min(rows[key][1], float(word.get("x0", 0) or 0))
    for left_edge, right_edge in rows.values():
        if right_edge != float("inf"):
            gaps.append(right_edge - left_edge)
    if not gaps:
        # No row has words on both sides, so nothing contradicts the two-column reading.
        return True
    gaps.sort()
    return gaps[len(gaps) // 2] >= _MIN_GUTTER_POINTS


def _cluster_columns(
    words: List[Dict[str, object]],
    page_width: float,
) -> List[ParsedBlock]:
    """Split mid-page word streams into left-then-right column reading order when clear.

    Uses word ``x0`` vs page midpoint (not ``x1``), so long lines that visually
    cross the gutter still assign to the column they start in. Requires clearly
    separated column centroids so dense single-column pages stay intact.
    """
    if len(words) < 8 or page_width <= 0:
        return []
    page_is_rtl = dominant_direction(" ".join(str(w.get("text", "")) for w in words)) == "rtl"
    mid = page_width / 2.0
    left = [w for w in words if float(w.get("x0", 0) or 0) < mid]
    right = [w for w in words if float(w.get("x0", 0) or 0) >= mid]
    if len(left) < 3 or len(right) < 3:
        return []
    left_cx = sum(float(w.get("x0", 0) or 0) for w in left) / len(left)
    right_cx = sum(float(w.get("x0", 0) or 0) for w in right) / len(right)
    # Centroids must sit on opposite sides of mid with meaningful separation.
    if not (left_cx < mid - 20 and right_cx > mid + 20):
        return []
    if (right_cx - left_cx) < page_width * 0.18:
        return []
    if not _has_column_gutter(left, right):
        return []

    def _words_to_text(group: List[Dict[str, object]]) -> str:
        ordered = sorted(
            group,
            key=lambda w: (round(float(w.get("top", 0) or 0), 1), float(w.get("x0", 0) or 0)),
        )
        parts: List[str] = []
        last_top: Optional[float] = None
        for w in ordered:
            top = float(w.get("top", 0) or 0)
            text = str(w.get("text", "") or "")
            if not text:
                continue
            if last_top is not None and abs(top - last_top) > 3.0:
                parts.append("\n")
            elif parts and not parts[-1].endswith("\n"):
                parts.append(" ")
            parts.append(text)
            last_top = top
        return "".join(parts).strip()

    blocks: List[ParsedBlock] = []
    # RTL pages read the right-hand column first.
    columns = [("right", right), ("left", left)] if page_is_rtl else [("left", left), ("right", right)]
    for order, (side, group) in enumerate(columns):
        column_text = repair_rtl_text(_words_to_text(group))
        if column_text:
            blocks.append(
                ParsedBlock(
                    block_type="column",
                    text=column_text,
                    page_number=0,  # filled by caller
                    metadata={"column": side, "order": order},
                )
            )
    return blocks


class PdfPlumberDocumentParser:
    """Layout-aware PDF parser (primary). Emits table + column blocks when detected."""

    name = "pdfplumber"

    def __init__(self, ocr_engine: Optional[OcrEngine] = None) -> None:
        self._ocr_engine = ocr_engine

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".pdf"

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        doc = self.extract_document(file_bytes)
        return doc.page_count, doc.pages

    def extract_blocks(self, file_bytes: bytes) -> List[Dict[str, object]]:
        """Acceptance/helper API: structured blocks as plain dicts."""
        doc = self.extract_document(file_bytes)
        return [
            {
                "block_type": b.block_type,
                "type": b.block_type,
                "text": b.text,
                "page_number": b.page_number,
                "metadata": dict(b.metadata),
            }
            for b in doc.blocks
        ]

    def extract_structured_blocks(self, file_bytes: bytes) -> List[Dict[str, object]]:
        return self.extract_blocks(file_bytes)

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        if not _PDFPLUMBER_AVAILABLE:
            raise RuntimeError(
                "pdfplumber is not installed. Install pdfplumber or use PyPdfDocumentParser fallback."
            )
        pages: List[Tuple[int, str]] = []
        blocks: List[ParsedBlock] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_blocks: List[ParsedBlock] = []
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables):
                    md = _table_to_markdown(table)
                    if md:
                        page_blocks.append(
                            ParsedBlock(
                                block_type="table",
                                text=md,
                                page_number=page_num,
                                metadata={"table_index": t_idx},
                            )
                        )

                words = page.extract_words() or []
                if page_blocks:
                    # Prefer structured tables; keep only prose outside table bboxes
                    # so flattened cell streams do not re-pollute retrieval.
                    table_bboxes = []
                    try:
                        for t in page.find_tables() or []:
                            if getattr(t, "bbox", None):
                                table_bboxes.append(tuple(t.bbox))
                    except Exception:
                        table_bboxes = []

                    def _in_table(word: Dict[str, object]) -> bool:
                        x0 = float(word.get("x0", 0) or 0)
                        x1 = float(word.get("x1", 0) or 0)
                        top = float(word.get("top", 0) or 0)
                        bottom = float(word.get("bottom", 0) or 0)
                        for bx0, btop, bx1, bbottom in table_bboxes:
                            if x0 >= bx0 - 1 and x1 <= bx1 + 1 and top >= btop - 1 and bottom <= bbottom + 1:
                                return True
                        return False

                    prose_words = [w for w in words if not _in_table(w)]
                    if prose_words:
                        ordered = sorted(
                            prose_words,
                            key=lambda w: (
                                round(float(w.get("top", 0) or 0), 1),
                                float(w.get("x0", 0) or 0),
                            ),
                        )
                        parts: List[str] = []
                        last_top: Optional[float] = None
                        for w in ordered:
                            top = float(w.get("top", 0) or 0)
                            text = str(w.get("text", "") or "")
                            if not text:
                                continue
                            if last_top is not None and abs(top - last_top) > 3.0:
                                parts.append("\n")
                            elif parts and not parts[-1].endswith("\n"):
                                parts.append(" ")
                            parts.append(text)
                            last_top = top
                        prose = repair_rtl_text("".join(parts).strip())
                        if prose:
                            page_blocks.insert(
                                0,
                                ParsedBlock(
                                    block_type="paragraph",
                                    text=prose,
                                    page_number=page_num,
                                    metadata={"outside_tables": True},
                                ),
                            )
                else:
                    column_blocks = _cluster_columns(words, float(page.width or 0))
                    if column_blocks:
                        for cb in column_blocks:
                            page_blocks.append(
                                ParsedBlock(
                                    block_type=cb.block_type,
                                    text=cb.text,
                                    page_number=page_num,
                                    metadata=dict(cb.metadata),
                                )
                            )
                    else:
                        plain = repair_rtl_text((page.extract_text() or "").strip())
                        if plain:
                            metadata = {"direction": "rtl"} if has_rtl(plain) else {}
                            page_blocks.append(
                                ParsedBlock(
                                    block_type="paragraph",
                                    text=plain,
                                    page_number=page_num,
                                    metadata=metadata,
                                )
                            )
                        else:
                            ocr_block = self._ocr_page(page, page_num)
                            if ocr_block is not None:
                                page_blocks.append(ocr_block)

                blocks.extend(page_blocks)
                page_text = "\n\n".join(b.text for b in page_blocks if b.text).strip()
                if page_text:
                    pages.append((page_num, page_text))

        return ParsedDocument(
            page_count=total_pages,
            pages=pages,
            blocks=blocks,
            parser_name=self.name,
        )


    def _ocr_page(self, page: Any, page_num: int) -> Optional[ParsedBlock]:
        """OCR a page with no text layer. Without an engine, say so rather than skip silently."""
        if not page.images:
            return None
        if self._ocr_engine is None:
            logger.warning(
                "Page %s has images but no text layer and was NOT indexed. "
                "Set RAG_OCR_LANGS and install the [ocr] extra to OCR scanned pages.",
                page_num,
            )
            return None
        image = page.to_image(resolution=200).original
        lines = self._ocr_engine.recognize(image)
        text = order_ocr_lines(lines)
        if not text:
            return None
        return ParsedBlock(
            block_type="ocr_text",
            text=text,
            page_number=page_num,
            metadata={
                "ocr_engine": self._ocr_engine.name,
                "ocr_mean_confidence": round(sum(l.confidence for l in lines) / len(lines), 3),
                "direction": dominant_direction(text),
            },
        )


class PyPdfDocumentParser:
    """Lightweight offline PDF fallback using PyPDF plain extract_text()."""

    name = "pypdf"

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".pdf"

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        doc = self.extract_document(file_bytes)
        return doc.page_count, doc.pages

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        page_texts: List[Tuple[int, str]] = []
        blocks: List[ParsedBlock] = []
        for page_num, page in enumerate(reader.pages, start=1):
            extracted_text = (page.extract_text() or "").strip()
            if extracted_text:
                page_texts.append((page_num, extracted_text))
                blocks.append(
                    ParsedBlock(
                        block_type="paragraph",
                        text=extracted_text,
                        page_number=page_num,
                        metadata={"parser": self.name},
                    )
                )
        return ParsedDocument(
            page_count=total_pages,
            pages=page_texts,
            blocks=blocks,
            parser_name=self.name,
        )


class PlainTextDocumentParser:
    """Plain text and Markdown parser."""

    name = "plaintext"

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in {".txt", ".md", ".markdown"}

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        doc = self.extract_document(file_bytes)
        return doc.page_count, doc.pages

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        text = file_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            return ParsedDocument(page_count=0, pages=[], blocks=[], parser_name=self.name)
        block = ParsedBlock(block_type="paragraph", text=text, page_number=1, metadata={})
        return ParsedDocument(
            page_count=1,
            pages=[(1, text)],
            blocks=[block],
            parser_name=self.name,
        )


_CROSS_SHEET_RE = re.compile(r"(?:'([^']+)'|([A-Za-z_][\w.]*))!\$?[A-Z]{1,3}\$?\d+")
_MAX_FORMULA_NOTES = 12


def _format_cell(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time(0) else value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, str):
        return value.strip()
    return value


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}" if abs(value) < 1 else f"{value:,.2f}".replace(",", "")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(_cell_text(c) for c in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(_cell_text(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _column_profile(name: str, values: Sequence[Any]) -> str:
    present = [v for v in values if v not in (None, "")]
    if not present:
        return f"{name} (empty)"
    numbers = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numbers) == len(present):
        return f"{name} (number, min {_cell_text(min(numbers))}, max {_cell_text(max(numbers))}, sum {_cell_text(sum(numbers))})"
    distinct = sorted({str(v) for v in present})
    examples = ", ".join(distinct[:5])
    return f"{name} (text, {len(distinct)} distinct, e.g. {examples})"


class XlsxDocumentParser:
    """Structure-aware .xlsx parser. Fixes what flat ETL destroys in spreadsheets:

      - Merged cells: the value is copied to every cell of the merged range, so a row
        under a vertically merged "Asia" label still says "Asia".
      - Merged headers: a spanning header above sub-headers becomes "Q1 Revenue".
      - Formulas: the cached result is indexed, not "=SUM(...)". A formula that was
        never calculated is shown as its text and flagged, not as an empty cell.
      - Cross-sheet references: ``=Rates!B2*C5`` adds a lineage note naming the sheet
        the number depends on, because that link is gone once a sheet becomes a chunk.
      - Blank-row separated regions become separate tables.
      - Hidden sheets are skipped, because their content is not meant for every reader.

    Regions over ``max_inline_rows`` are not embedded row by row. They yield one
    ``table_summary`` block plus a :class:`ParsedTable` for the relational store.
    """

    name = "openpyxl"
    MAX_CELLS = 2_000_000

    def __init__(
        self,
        *,
        max_inline_rows: Optional[int] = None,
        rows_per_block: int = 25,
        summarizer: Optional[Callable[[str, Sequence[str], Sequence[Sequence[Any]]], str]] = None,
    ) -> None:
        self.max_inline_rows = max_inline_rows or int(os.getenv("RAG_XLSX_INLINE_ROWS", "40"))
        self.rows_per_block = rows_per_block
        self._summarizer = summarizer

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".xlsx"

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        doc = self.extract_document(file_bytes)
        return doc.page_count, doc.pages

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        from openpyxl import load_workbook

        values_wb = load_workbook(io.BytesIO(file_bytes), data_only=True)
        formula_wb = load_workbook(io.BytesIO(file_bytes), data_only=False)
        file_digest = hashlib.sha256(file_bytes).hexdigest()[:12]

        blocks: List[ParsedBlock] = []
        tables: List[ParsedTable] = []
        visible = [ws for ws in values_wb.worksheets if ws.sheet_state == "visible"]
        hidden = [ws.title for ws in values_wb.worksheets if ws.sheet_state != "visible"]
        if hidden:
            logger.warning("Skipping %d hidden sheet(s): %s", len(hidden), hidden)

        for sheet_number, ws in enumerate(visible, start=1):
            if (ws.max_row or 0) * (ws.max_column or 0) > self.MAX_CELLS:
                raise ValueError(f"Sheet {ws.title!r} exceeds {self.MAX_CELLS:,} cells; refusing to load it.")
            grid, merged_cols, notes, uncached = self._read_sheet(ws, formula_wb[ws.title])
            for region_index, (start, region) in enumerate(self._regions(grid)):
                title, columns, rows = self._split_header(region, merged_cols, start)
                if not columns or not rows:
                    continue
                context = f"Sheet: {ws.title}" + (f" — {title}" if title else "")
                base_meta = {"sheet": ws.title, "sheet_index": sheet_number, "region": region_index}
                region_notes = self._notes_text(notes, uncached)
                if len(rows) > self.max_inline_rows:
                    table_id = f"t_{file_digest}_{sheet_number}_{region_index}"
                    tables.append(ParsedTable(table_id=table_id, name=f"{ws.title}/{region_index}", columns=columns, rows=rows))
                    blocks.append(
                        ParsedBlock(
                            block_type="table_summary",
                            text=self._summary(context, table_id, columns, rows, region_notes),
                            page_number=sheet_number,
                            metadata={**base_meta, "table_id": table_id, "row_count": len(rows)},
                        )
                    )
                    continue
                # Every block repeats the header so no row is ever separated from its labels.
                for offset in range(0, len(rows), self.rows_per_block):
                    batch = rows[offset : offset + self.rows_per_block]
                    text = f"{context}\n{_markdown_table(columns, batch)}"
                    if region_notes and offset == 0:
                        text += f"\n{region_notes}"
                    blocks.append(
                        ParsedBlock(
                            block_type="table",
                            text=text,
                            page_number=sheet_number,
                            metadata={**base_meta, "first_row": start + offset},
                        )
                    )

        pages_by_sheet: Dict[int, List[str]] = {}
        for block in blocks:
            pages_by_sheet.setdefault(block.page_number, []).append(block.text)
        return ParsedDocument(
            page_count=len(visible),
            pages=[(number, "\n\n".join(texts)) for number, texts in sorted(pages_by_sheet.items())],
            blocks=blocks,
            parser_name=self.name,
            tables=tables,
        )

    def _read_sheet(self, ws: Any, formula_ws: Any) -> Tuple[List[List[Any]], Dict[int, List[Tuple[int, int]]], List[str], List[str]]:
        """Return the value grid with merges expanded, plus formula lineage notes."""
        max_row, max_col = ws.max_row or 0, ws.max_column or 0
        grid: List[List[Any]] = [[None] * max_col for _ in range(max_row)]
        notes: List[str] = []
        uncached: List[str] = []
        for row in formula_ws.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
            for cell in row:
                raw = cell.value
                value = ws.cell(row=cell.row, column=cell.column).value
                if isinstance(raw, str) and raw.startswith("="):
                    referenced = sorted({a or b for a, b in _CROSS_SHEET_RE.findall(raw)})
                    if value is None:
                        uncached.append(cell.coordinate)
                        value = raw
                    if referenced and len(notes) < _MAX_FORMULA_NOTES:
                        notes.append(f"{cell.coordinate} `{raw}` depends on {', '.join(referenced)}")
                grid[cell.row - 1][cell.column - 1] = _format_cell(value)

        merged_cols: Dict[int, List[Tuple[int, int]]] = {}
        for merged in ws.merged_cells.ranges:
            source = grid[merged.min_row - 1][merged.min_col - 1]
            for row_index in range(merged.min_row - 1, merged.max_row):
                for col_index in range(merged.min_col - 1, merged.max_col):
                    grid[row_index][col_index] = source
            if merged.max_col > merged.min_col:
                merged_cols.setdefault(merged.min_row - 1, []).append((merged.min_col - 1, merged.max_col - 1))
        return grid, merged_cols, notes, uncached

    @staticmethod
    def _regions(grid: List[List[Any]]) -> List[Tuple[int, List[List[Any]]]]:
        """Split on fully blank rows and trim columns that are empty in a whole region."""
        regions: List[Tuple[int, List[List[Any]]]] = []
        current: List[List[Any]] = []
        start = 0
        for index, row in enumerate(grid + [[None]]):
            if any(cell not in (None, "") for cell in row):
                if not current:
                    start = index
                current.append(row)
            elif current:
                regions.append((start, current))
                current = []
        return regions

    @staticmethod
    def _split_header(
        region: List[List[Any]],
        merged_cols: Dict[int, List[Tuple[int, int]]],
        start: int,
    ) -> Tuple[str, List[str], List[List[Any]]]:
        """Separate an optional title row and a one- or two-level header from the data."""
        keep = [c for c in range(len(region[0])) if any(row[c] not in (None, "") for row in region)]
        rows = [[row[c] for c in keep] for row in region]
        title = ""
        offset = 0
        # A title is a single value, possibly merged across the whole width.
        if len(rows) > 1 and len({str(v) for v in rows[0] if v not in (None, "")}) == 1 and len(keep) > 1:
            title = str(next(v for v in rows[0] if v not in (None, "")))
            offset = 1
        if offset >= len(rows):
            return title, [], []
        header = [str(v) if v not in (None, "") else "" for v in rows[offset]]
        body_start = offset + 1
        spans = merged_cols.get(start + offset, [])
        if spans and body_start < len(rows) and all(isinstance(v, str) or v is None for v in rows[body_start]):
            # Two-level header: "Q1" spanning Revenue/Cost becomes "Q1 Revenue", "Q1 Cost".
            lower = rows[body_start]
            header = [
                f"{top} {str(low)}".strip() if low not in (None, "") and str(low) != top else top
                for top, low in zip(header, lower)
            ]
            body_start += 1
        header = [name or f"column_{i + 1}" for i, name in enumerate(header)]
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for name in header:
            seen[name] = seen.get(name, 0) + 1
            unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
        return title, unique, rows[body_start:]

    @staticmethod
    def _notes_text(notes: Sequence[str], uncached: Sequence[str]) -> str:
        parts: List[str] = []
        if notes:
            parts.append("Cross-sheet lineage: " + "; ".join(notes))
        if uncached:
            parts.append(f"Formulas never calculated (showing formula text): {', '.join(uncached[:10])}")
        return "\n".join(parts)

    def _summary(
        self,
        context: str,
        table_id: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        notes: str,
    ) -> str:
        profile = "; ".join(_column_profile(name, [row[i] for row in rows]) for i, name in enumerate(columns))
        lines = [
            context,
            f"Large table with {len(rows):,} rows, stored for structured queries as table_id={table_id}.",
            f"Columns: {profile}.",
        ]
        if self._summarizer is not None:
            try:
                purpose = self._summarizer(context, columns, rows[:8]).strip()
                if purpose:
                    lines.insert(1, f"Purpose: {purpose}")
            except Exception as exc:  # The deterministic profile is still a usable summary.
                logger.warning("LLM table summary failed for %s: %s", table_id, exc)
        lines.append(f"Sample rows:\n{_markdown_table(columns, rows[:3])}")
        if notes:
            lines.append(notes)
        return "\n".join(lines)


def llm_table_summarizer(context: str, columns: Sequence[str], sample: Sequence[Sequence[Any]]) -> str:
    """One-sentence semantic summary of a table's purpose (enabled by ``RAG_XLSX_LLM_SUMMARY``)."""
    from src.config import get_llm

    response = get_llm(role="manager", temperature=0.0).invoke(
        "In one sentence, state what business question this table answers. "
        "Do not list the columns and do not quote values.\n"
        f"{context}\n{_markdown_table(columns, sample)}"
    )
    return str(getattr(response, "content", response))


def _env_allow_degraded(default: bool = True) -> bool:
    raw = os.getenv("RAG_PARSER_ALLOW_DEGRADED", "true" if default else "false")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class DocumentParserRegistry:
    """Registry coordinating format-specific document parsers.

    When the layout-aware PDF parser fails at runtime, behaviour is controlled by
    ``allow_degraded`` / ``RAG_PARSER_ALLOW_DEGRADED``:
      - True (default): WARNING + reason, fall back to PyPDF, mark ``parser_degraded``.
      - False: WARNING + reason, raise ``ParserDegradedError`` (fail-loud).
    """

    def __init__(
        self,
        *,
        prefer_layout_aware: bool = True,
        allow_degraded: Optional[bool] = None,
    ) -> None:
        self._parsers: List[DocumentParser] = []
        self._prefer_layout_aware = prefer_layout_aware
        self._allow_degraded = (
            _env_allow_degraded(default=True) if allow_degraded is None else bool(allow_degraded)
        )
        # Text formats first for extension clarity; PDF adapters next.
        self.register(PlainTextDocumentParser())
        self.register(
            XlsxDocumentParser(
                summarizer=llm_table_summarizer
                if os.getenv("RAG_XLSX_LLM_SUMMARY", "false").strip().lower() == "true"
                else None
            )
        )
        if prefer_layout_aware and _PDFPLUMBER_AVAILABLE:
            self.register(PdfPlumberDocumentParser(ocr_engine=create_ocr_engine_from_env()))
        self.register(PyPdfDocumentParser())

    @property
    def allow_degraded(self) -> bool:
        return self._allow_degraded

    def register(self, parser: DocumentParser) -> None:
        self._parsers.insert(0, parser)

    def get_parser(self, filename: str) -> DocumentParser:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._select_pdf_parser()
        for parser in self._parsers:
            if parser.can_parse(filename):
                return parser
        return self._select_pdf_parser()

    def _select_pdf_parser(self) -> DocumentParser:
        if self._prefer_layout_aware:
            for parser in self._parsers:
                if isinstance(parser, PdfPlumberDocumentParser):
                    return parser
        for parser in self._parsers:
            if isinstance(parser, PyPdfDocumentParser):
                return parser
        for parser in self._parsers:
            if parser.can_parse("doc.pdf"):
                return parser
        raise RuntimeError("No PDF parser registered in DocumentParserRegistry.")

    def _pypdf_parser(self) -> Optional[PyPdfDocumentParser]:
        for parser in self._parsers:
            if isinstance(parser, PyPdfDocumentParser):
                return parser
        return None

    def extract_pages(self, filename: str, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        parser = self.get_parser(filename)
        return parser.extract_pages(file_bytes)

    def extract_document(self, filename: str, file_bytes: bytes) -> ParsedDocument:
        parser = self.get_parser(filename)
        if hasattr(parser, "extract_document"):
            try:
                return parser.extract_document(file_bytes)  # type: ignore[attr-defined]
            except RuntimeError as exc:
                # Layout-aware primary failed — explicit degraded policy, never silent.
                primary_name = str(getattr(parser, "name", "") or type(parser).__name__).lower()
                is_layout_primary = (
                    self._prefer_layout_aware
                    and (
                        isinstance(parser, PdfPlumberDocumentParser)
                        or primary_name == "pdfplumber"
                        or "plumber" in primary_name
                        or "layout" in primary_name
                    )
                )
                if is_layout_primary:
                    reason = (
                        f"Layout-aware parse failed ({exc}); "
                        f"primary={getattr(parser, 'name', type(parser).__name__)}"
                    )
                    fallback = self._pypdf_parser()
                    if not self._allow_degraded:
                        logger.warning(
                            "%s; degraded fallback disabled (RAG_PARSER_ALLOW_DEGRADED=false).",
                            reason,
                        )
                        raise ParserDegradedError(
                            f"{reason}. Set RAG_PARSER_ALLOW_DEGRADED=true to permit PyPDF fallback."
                        ) from exc
                    if fallback is None:
                        logger.warning("%s; no PyPDF fallback registered.", reason)
                        raise
                    logger.warning(
                        "%s; falling back to PyPDF (degraded, allow_degraded=true).",
                        reason,
                    )
                    degraded = fallback.extract_document(file_bytes)
                    return ParsedDocument(
                        page_count=degraded.page_count,
                        pages=list(degraded.pages),
                        blocks=list(degraded.blocks),
                        parser_name=getattr(fallback, "name", "pypdf"),
                        parser_degraded=True,
                        parser_fallback_reason=reason,
                    )
                raise
        total, pages = parser.extract_pages(file_bytes)
        blocks = [
            ParsedBlock(block_type="paragraph", text=text, page_number=num, metadata={})
            for num, text in pages
        ]
        return ParsedDocument(
            page_count=total,
            pages=pages,
            blocks=blocks,
            parser_name=getattr(parser, "name", parser.__class__.__name__),
        )

    @property
    def pdfplumber_available(self) -> bool:
        return _PDFPLUMBER_AVAILABLE


# Alias for backward compatibility
PdfTextExtractor = PyPdfDocumentParser
