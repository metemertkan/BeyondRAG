"""Script-aware text primitives for mixed English / Chinese / Arabic ingestion.

Everything here is about *writing systems*, not product languages, so it stays
compatible with the data-driven language packs in ``language_packs.py``:

  - ``repair_rtl_line``: PDF extractors sort glyphs by x-coordinate, which yields
    visual order. For RTL text that means reversed words; this restores logical order.
  - ``normalize_for_matching``: folds presentation forms, tashkeel, tatweel and
    Arabic-Indic digits so the same word always produces the same tokens.
  - ``MultilingualTokenizer``: segments Han with jieba and never fragments Arabic
    words at their diacritics.
  - ``split_sentences``: sentence stops that do not need a following space (。) and
    the Arabic stops (؟ ۔) a Latin-only regex misses.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Final, Iterable, List

logger = logging.getLogger(__name__)

_RTL_CLASS: Final[str] = "֐-ࣿיִ-﷿ﹰ-ﻼ"
_RTL_RE: Final[re.Pattern[str]] = re.compile(f"[{_RTL_CLASS}]")
_LATIN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]")
_MARK_RE: Final[re.Pattern[str]] = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL: Final[str] = "ـ"
_HAN_RANGES: Final[str] = "㐀-䶿一-鿿豈-﫿"
# Scripts written without spaces between words. Deliberately excludes U+FE70+ (Arabic
# presentation forms), which the older 一-￿ range misclassified as CJK.
_NO_SPACE_CLASS: Final[str] = f"぀-ヿ{_HAN_RANGES}"
NO_SPACE_RE: Final[re.Pattern[str]] = re.compile(f"[{_NO_SPACE_CLASS}]")
_ARABIC_INDIC_DIGITS: Final[dict[int, str]] = {
    **{0x0660 + i: str(i) for i in range(10)},
    **{0x06f0 + i: str(i) for i in range(10)},
}
_MIRRORED: Final[dict[int, str]] = str.maketrans("()[]{}<>", ")(][}{><")

# RTL run / whitespace / LTR-or-neutral run. Spaces between two RTL words stay in
# the whitespace group so word order can be reversed; a multi-word LTR run such as
# "HKD 450,000,000" is kept together because bidi preserves its internal order.
_VISUAL_RUN_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?P<rtl>[{_RTL_CLASS}]+)|(?P<space>\s+)|(?P<other>[^\s{_RTL_CLASS}]+(?:\s+[^\s{_RTL_CLASS}]+)*)"
)
_EDGE_NEUTRAL_RE: Final[re.Pattern[str]] = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.DOTALL)

_SENTENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[。！？؟۔])\s*"  # CJK / Arabic stops: whitespace optional
    r"|(?<=[.!?;；])\s+"  # Latin stops: whitespace required ("3.5" must survive)
    r"|\n\s*\n"  # paragraph break
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    rf"[{_HAN_RANGES}]+"
    r"|[぀-ヿ]+"
    # Arabic-script letters and marks, without Arabic punctuation (، ؛ ؟ ۔).
    r"|[\u0590-\u05ff\u0610-\u061a\u0620-\u065f\u066e-\u06d3\u06d5-\u08ff\ufb1d-\ufdff\ufe70-\ufefc]+"
    r"|\d[\d,]*(?:\.\d+)?"
    r"|[^\W\d_]+(?:[-'][^\W\d_]+)*"
    r"|\S",
    re.UNICODE,
)
_HAN_RUN_RE: Final[re.Pattern[str]] = re.compile(f"[{_HAN_RANGES}]+")
_WORD_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"\w", re.UNICODE)


def dominant_direction(text: str) -> str:
    """Return ``rtl`` when RTL letters outnumber Latin letters, else ``ltr``."""
    rtl = len(_RTL_RE.findall(text))
    return "rtl" if rtl and rtl >= len(_LATIN_RE.findall(text)) else "ltr"


def has_rtl(text: str) -> bool:
    return bool(_RTL_RE.search(text))


def repair_rtl_line(line: str) -> str:
    """Convert one x-sorted (visual order) line back to logical order.

    A line dominated by RTL text reverses its run order: RTL runs flip, embedded
    LTR runs keep their internal order, and edge punctuation moves to the far side
    of the run it visually sits on. In a dominantly LTR line only the RTL runs flip,
    in place. Reversal must happen before NFKC because a lam-alef ligature is one
    codepoint in visual order and expands to two in logical order.
    """
    if not has_rtl(line):
        return line

    if dominant_direction(line) == "ltr":
        pattern = rf"[{_RTL_CLASS}]+(?:\s+[{_RTL_CLASS}]+)*"
        return re.sub(pattern, lambda m: m.group(0)[::-1], line)

    pieces: List[str] = []
    for match in _VISUAL_RUN_RE.finditer(line):
        kind = match.lastgroup
        text = match.group(0)
        if kind == "rtl":
            pieces.append(text[::-1])
        elif kind == "space":
            pieces.append(text)
        else:
            lead, core, trail = _EDGE_NEUTRAL_RE.match(text).groups()  # type: ignore[union-attr]
            # Reversing the run list moves this run as a unit, so its edge
            # punctuation has to swap sides and mirror to stay attached correctly.
            pieces.append(trail.translate(_MIRRORED)[::-1] + core + lead.translate(_MIRRORED)[::-1])
    return "".join(reversed(pieces))


def repair_rtl_text(text: str) -> str:
    """Repair a multi-line extraction block and fold presentation forms to letters."""
    if not has_rtl(text):
        return text
    return "\n".join(
        unicodedata.normalize("NFKC", repair_rtl_line(line)) if has_rtl(line) else line
        for line in text.split("\n")
    )


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics and the elongation mark; base letters are untouched."""
    return _MARK_RE.sub("", text).replace(_TATWEEL, "")


def normalize_for_matching(text: str) -> str:
    """Canonical form for embedding and matching (stored chunk text is not rewritten).

    NFKC folds presentation forms and full-width Latin; tashkeel removal means
    ``الْحَدّ`` and ``الحد`` embed identically; Arabic-Indic digits become ASCII so
    ``٤٥٠`` matches ``450``.
    """
    return strip_tashkeel(unicodedata.normalize("NFKC", text)).translate(_ARABIC_INDIC_DIGITS)


def split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries across Latin, CJK and Arabic punctuation."""
    return [part.strip() for part in _SENTENCE_RE.split(text) if part and part.strip()]


def join_units(parts: Iterable[str]) -> str:
    """Join sentences with a space only where the script actually uses spaces."""
    out = ""
    for part in parts:
        if not part:
            continue
        needs_space = bool(out) and not (NO_SPACE_RE.match(part[0]) and (NO_SPACE_RE.match(out[-1]) or out[-1] in "。！？"))
        out += (" " if needs_space else "") + part
    return out


_jieba_module = None
_jieba_checked = False


def _jieba():
    """Import jieba on first use (it loads a dictionary) and quiet its startup log."""
    global _jieba_module, _jieba_checked
    if not _jieba_checked:
        _jieba_checked = True
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            _jieba_module = jieba
        except ImportError:
            logger.warning("jieba not installed; Han text falls back to per-character tokens.")
    return _jieba_module


class MultilingualTokenizer:
    """Script-routed tokenizer implementing ``TokenizerProtocol``.

    Han runs go through jieba, Kana and unsegmented Han fall back to single
    characters, and Arabic runs (diacritics included) count as one word per run.
    """

    def split_into_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for token in _TOKEN_RE.findall(text):
            if _HAN_RUN_RE.fullmatch(token):
                segmenter = _jieba()
                tokens.extend(segmenter.lcut(token) if segmenter else list(token))
            elif NO_SPACE_RE.search(token):
                tokens.extend(list(token))
            else:
                tokens.append(token)
        return tokens

    def count_tokens(self, text: str) -> int:
        return len(self.split_into_tokens(text))


def content_terms(text: str) -> set[str]:
    """Normalized word-like tokens, used for lexical overlap across scripts."""
    normalized = normalize_for_matching(text).lower()
    return {
        token
        for token in MultilingualTokenizer().split_into_tokens(normalized)
        if _WORD_CHAR_RE.search(token) and (len(token) > 2 or bool(NO_SPACE_RE.search(token)))
    }
