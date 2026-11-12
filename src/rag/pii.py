"""Find and mask personal data with rules: a pattern, then a checksum where one exists.

Used in two places. Guardrails mask what goes to the model and what comes back. Telemetry
masks everything it stores, always, so a trace or a log line is never the place a card
number survives.

What this cannot find: names, addresses, and phone numbers written without a country code.
A pattern has nothing to hold on to there. ``scripts/ep6_benchmark.py`` measures that gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Final, List, Optional, Tuple


@dataclass(frozen=True)
class PiiMatch:
    kind: str
    start: int
    end: int
    text: str


def luhn_valid(digits: str) -> bool:
    """The check digit on payment cards. One random number in ten still passes."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def hkid_valid(candidate: str) -> bool:
    """Hong Kong identity card: one or two letters, six digits, a check character."""
    cleaned = re.sub(r"[()\s]", "", candidate.upper())
    match = re.fullmatch(r"([A-Z]{1,2})(\d{6})([0-9A])", cleaned)
    if not match:
        return False
    letters, digits, check = match.groups()
    # A single-letter prefix counts as a space (36) followed by the letter.
    values = [36] if len(letters) == 1 else []
    values += [ord(letter) - 55 for letter in letters]
    values += [int(digit) for digit in digits]
    total = sum(value * weight for value, weight in zip(values, range(9, 1, -1)))
    total += 10 if check == "A" else int(check)
    return total % 11 == 0


def iban_valid(candidate: str) -> bool:
    cleaned = re.sub(r"\s", "", candidate.upper())
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", cleaned):
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def _card_valid(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 13 <= len(digits) <= 19 and luhn_valid(digits)


def _phone_valid(candidate: str) -> bool:
    return 8 <= len(re.sub(r"\D", "", candidate)) <= 15


# Order is priority: when two patterns overlap, the earlier one keeps the text.
_DETECTORS: Final[Tuple[Tuple[str, re.Pattern[str], Optional[Callable[[str], bool]]], ...]] = (
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), None),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b"), iban_valid),
    ("CARD", re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"), _card_valid),
    ("HKID", re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,2}\d{6}\s?\(?[0-9A]\)?(?![A-Za-z0-9])"), hkid_valid),
    # Only the international form. An eight-digit local number looks like any other number.
    ("PHONE", re.compile(r"(?<![\w+])\+\d[\d ()-]{6,18}\d(?!\d)"), _phone_valid),
)

PII_KINDS: Final[Tuple[str, ...]] = tuple(kind for kind, _, _ in _DETECTORS)


def find_pii(text: str, *, checksums: bool = True) -> List[PiiMatch]:
    """Matches in reading order. ``checksums=False`` keeps every pattern hit, valid or not."""
    found: List[PiiMatch] = []
    for kind, pattern, validator in _DETECTORS:
        for match in pattern.finditer(text or ""):
            if checksums and validator is not None and not validator(match.group()):
                continue
            if any(match.start() < other.end and other.start < match.end() for other in found):
                continue
            found.append(PiiMatch(kind, match.start(), match.end(), match.group()))
    return sorted(found, key=lambda item: item.start)


def redact_pii(text: str, *, checksums: bool = True) -> Tuple[str, List[PiiMatch]]:
    """Replace each match with its kind in brackets, for example ``[EMAIL]``."""
    matches = find_pii(text, checksums=checksums)
    if not matches:
        return text, []
    parts: List[str] = []
    cursor = 0
    for match in matches:
        parts.append(text[cursor:match.start])
        parts.append(f"[{match.kind}]")
        cursor = match.end
    parts.append(text[cursor:])
    return "".join(parts), matches
