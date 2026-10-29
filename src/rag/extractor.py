"""Component 2: PDF Text Extraction Adapter.

Handles exclusively extracting selectable text per page from raw PDF byte streams.
"""

from __future__ import annotations

import io
from typing import List, Tuple

from pypdf import PdfReader


class PdfTextExtractor:
    """Extracts page count and non-empty per-page text strings from raw PDF bytes."""

    def extract_pages(self, pdf_bytes: bytes) -> Tuple[int, List[Tuple[int, str]]]:
        """Return `(total_pages, [(page_number, page_text), ...])`."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        page_texts: List[Tuple[int, str]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            extracted_text = (page.extract_text() or "").strip()
            if extracted_text:
                page_texts.append((page_num, extracted_text))
        return total_pages, page_texts
