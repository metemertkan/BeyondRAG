"""Component 2: Pluggable Document Parsers.

Production Episode 2 stack:
  - PdfPlumberDocumentParser: primary layout-aware PDF adapter (tables, columns)
  - PyPdfDocumentParser: lightweight offline Docker fallback
  - PlainTextDocumentParser: .txt / .md / .markdown
  - ExcelDocumentParser: registry-ready stub (.xlsx/.xls) — adapter deferred
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

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from pypdf import PdfReader

from src.rag.models import ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

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


def _table_to_markdown(table: Sequence[Optional[Sequence[Optional[str]]]]) -> str:
    """Render a pdfplumber table as markdown-like rows (cell text preserved)."""
    rows: List[List[str]] = []
    for raw_row in table:
        if raw_row is None:
            continue
        cells = [("" if c is None else str(c).strip().replace("\n", " ")) for c in raw_row]
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
    left_text = _words_to_text(left)
    right_text = _words_to_text(right)
    if left_text:
        blocks.append(
            ParsedBlock(
                block_type="column",
                text=left_text,
                page_number=0,  # filled by caller
                metadata={"column": "left", "order": 0},
            )
        )
    if right_text:
        blocks.append(
            ParsedBlock(
                block_type="column",
                text=right_text,
                page_number=0,
                metadata={"column": "right", "order": 1},
            )
        )
    return blocks


class PdfPlumberDocumentParser:
    """Layout-aware PDF parser (primary). Emits table + column blocks when detected."""

    name = "pdfplumber"

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
                        prose = "".join(parts).strip()
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
                        plain = (page.extract_text() or "").strip()
                        if plain:
                            page_blocks.append(
                                ParsedBlock(
                                    block_type="paragraph",
                                    text=plain,
                                    page_number=page_num,
                                    metadata={},
                                )
                            )

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


class ExcelDocumentParser:
    """Registry-ready Excel stub — Episode 2 does not ship a production adapter.

    Interfaces accept registration so a future .xlsx adapter does not require
    port changes. ``extract_*`` raises until an adapter is implemented.
    """

    name = "excel_stub"

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in {".xlsx", ".xls"}

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        raise NotImplementedError(
            "Excel (.xlsx/.xls) adapter is deferred past Episode 2. "
            "Register a real ExcelDocumentParser implementation when ready."
        )

    def extract_document(self, file_bytes: bytes) -> ParsedDocument:
        raise NotImplementedError(
            "Excel (.xlsx/.xls) adapter is deferred past Episode 2."
        )


class DocumentParserRegistry:
    """Registry coordinating format-specific document parsers."""

    def __init__(self, *, prefer_layout_aware: bool = True) -> None:
        self._parsers: List[DocumentParser] = []
        self._prefer_layout_aware = prefer_layout_aware
        # Text formats first for extension clarity; PDF adapters next.
        self.register(PlainTextDocumentParser())
        if prefer_layout_aware and _PDFPLUMBER_AVAILABLE:
            self.register(PdfPlumberDocumentParser())
        self.register(PyPdfDocumentParser())
        # Excel stub is intentionally NOT auto-registered for upload safety;
        # callers may ``register(ExcelDocumentParser())`` without port changes.

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

    def extract_pages(self, filename: str, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        parser = self.get_parser(filename)
        return parser.extract_pages(file_bytes)

    def extract_document(self, filename: str, file_bytes: bytes) -> ParsedDocument:
        parser = self.get_parser(filename)
        if hasattr(parser, "extract_document"):
            try:
                return parser.extract_document(file_bytes)  # type: ignore[attr-defined]
            except RuntimeError as exc:
                # Fall back to PyPDF when pdfplumber fails at runtime.
                if self._prefer_layout_aware:
                    logger.warning("Layout-aware parse failed (%s); falling back to PyPDF.", exc)
                    for parser in self._parsers:
                        if isinstance(parser, PyPdfDocumentParser):
                            return parser.extract_document(file_bytes)
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
