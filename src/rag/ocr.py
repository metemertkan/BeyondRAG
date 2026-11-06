"""OCR port for scanned pages, with a PaddleOCR 3.x adapter.

Pages with no text layer would otherwise vanish silently from the index. PaddleOCR is
an optional extra (``pip install .[ocr]``) because it adds about 1 GB of dependencies,
and it was chosen because it ships Arabic and Chinese recognizers. Choose the language
per corpus with ``RAG_OCR_LANGS`` (for example ``ar,ch,en``); with several languages
each page runs through every recognizer and the most confident result wins, so the
cost scales with the list.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

from src.rag.multilingual import dominant_direction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    # Left, top, right, bottom in pixels.
    box: tuple[float, float, float, float]


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: Any) -> List[OcrLine]:
        """Return recognized lines for a PIL image, in any order."""
        ...


def order_ocr_lines(lines: Sequence[OcrLine]) -> str:
    """Join lines in reading order: top to bottom, then by the page's direction.

    Fragments on the same visual row are read right-to-left on RTL pages.
    """
    if not lines:
        return ""
    joined = " ".join(line.text for line in lines)
    rtl_page = dominant_direction(joined) == "rtl"
    heights = sorted(line.box[3] - line.box[1] for line in lines)
    row_tolerance = max(heights[len(heights) // 2] * 0.6, 1.0)

    rows: List[List[OcrLine]] = []
    for line in sorted(lines, key=lambda item: item.box[1]):
        if rows and abs(line.box[1] - rows[-1][0].box[1]) <= row_tolerance:
            rows[-1].append(line)
        else:
            rows.append([line])
    return "\n".join(
        " ".join(item.text for item in sorted(row, key=lambda item: item.box[0], reverse=rtl_page))
        for row in rows
    )


class PaddleOcrEngine:
    """PaddleOCR 3.x adapter. Models load lazily and download on first use."""

    name = "paddleocr"

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._engine: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("PaddleOCR is not installed. Install with: pip install .[ocr]") from exc
            self._engine = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # PaddlePaddle 3.x's oneDNN backend raises NotImplementedError on CPU.
                enable_mkldnn=False,
            )
        return self._engine

    def recognize(self, image: Any) -> List[OcrLine]:
        import numpy as np

        array = np.array(image.convert("RGB"))[:, :, ::-1]  # PaddleOCR expects BGR.
        lines: List[OcrLine] = []
        for result in self._load().predict(array):
            data: Dict[str, Any] = result.json["res"] if hasattr(result, "json") else result
            boxes = data.get("rec_boxes")
            for index, text in enumerate(data.get("rec_texts", [])):
                if not str(text).strip():
                    continue
                box = tuple(float(v) for v in boxes[index]) if boxes is not None else (0.0, float(index), 0.0, float(index + 1))
                lines.append(
                    OcrLine(
                        # PP-OCRv5 already returns Arabic in logical order: no RTL repair.
                        text=str(text),
                        confidence=float(data.get("rec_scores", [1.0] * (index + 1))[index]),
                        box=box,  # type: ignore[arg-type]
                    )
                )
        return lines


class MultiLanguageOcrEngine:
    """Runs one recognizer per configured language and keeps the most confident page."""

    name = "paddleocr-multi"

    def __init__(self, engines: Sequence[OcrEngine]) -> None:
        self._engines = list(engines)

    def recognize(self, image: Any) -> List[OcrLine]:
        best: List[OcrLine] = []
        best_score = -1.0
        for engine in self._engines:
            lines = engine.recognize(image)
            if not lines:
                continue
            total_chars = sum(len(line.text) for line in lines) or 1
            score = sum(line.confidence * len(line.text) for line in lines) / total_chars
            if score > best_score:
                best, best_score = lines, score
        return best


def create_ocr_engine_from_env() -> OcrEngine | None:
    """Build the configured engine, or None when ``RAG_OCR_LANGS`` is unset."""
    languages = [code.strip() for code in os.getenv("RAG_OCR_LANGS", "").split(",") if code.strip()]
    if not languages:
        return None
    engines = [PaddleOcrEngine(lang=code) for code in languages]
    return engines[0] if len(engines) == 1 else MultiLanguageOcrEngine(engines)
