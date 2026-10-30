"""Component 2: Pluggable Document Parsers.

Supports multiple document formats via pluggable parser implementations:
  - PyPdfDocumentParser: Handles .pdf documents using PyPDF
  - PlainTextDocumentParser: Handles .txt and .md documents
  - DocumentParserRegistry: Dispatches parsing dynamically based on file extension
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

from pypdf import PdfReader


class DocumentParser(Protocol):
    """Protocol for pluggable document parsers."""

    def can_parse(self, filename: str) -> bool:
        ...

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        """Return (total_pages, list of (page_number, page_text))."""
        ...


class PyPdfDocumentParser:
    """PDF parser using PyPDF."""

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".pdf"

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        page_texts: List[Tuple[int, str]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            extracted_text = (page.extract_text() or "").strip()
            if extracted_text:
                page_texts.append((page_num, extracted_text))
        return total_pages, page_texts


class PlainTextDocumentParser:
    """Plain text and Markdown parser."""

    def can_parse(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in {".txt", ".md", ".markdown"}

    def extract_pages(self, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        text = file_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            return 0, []
        return 1, [(1, text)]


class DocumentParserRegistry:
    """Registry coordinating format-specific document parsers."""

    def __init__(self) -> None:
        self._parsers: List[DocumentParser] = [
            PyPdfDocumentParser(),
            PlainTextDocumentParser(),
        ]

    def register(self, parser: DocumentParser) -> None:
        self._parsers.insert(0, parser)

    def get_parser(self, filename: str) -> DocumentParser:
        for parser in self._parsers:
            if parser.can_parse(filename):
                return parser
        # Default to PDF parser for backward compatibility
        return self._parsers[0]

    def extract_pages(self, filename: str, file_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        parser = self.get_parser(filename)
        return parser.extract_pages(file_bytes)


# Alias for backward compatibility
PdfTextExtractor = PyPdfDocumentParser
