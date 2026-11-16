"""Scan text for words that date it after the cutoff.

Two users: the corpus build (every document) and the probe build (every prompt and letter
that goes to a historical model).

    uv run python -m corpus.scan some_file.txt
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import yaml

TERMS_FILE = Path(__file__).resolve().parent / "banned_terms.yaml"


def _compile(entries: list[dict]) -> list[tuple[str, re.Pattern]]:
    return [(e["name"], re.compile(e["re"], re.IGNORECASE if e.get("i") else 0)) for e in entries]


@lru_cache(maxsize=1)
def terms() -> dict:
    raw = yaml.safe_load(TERMS_FILE.read_text())
    return {
        "reject": _compile(raw["reject"]),
        "flag": _compile(raw["flag"]),
        "prompt_only": _compile(raw["prompt_only"]),
        "late_year": re.compile(raw["late_year"]["re"]),
        "late_year_min": raw["late_year"]["min_matches"],
    }


def _hits(patterns: list[tuple[str, re.Pattern]], text: str) -> dict[str, int]:
    out = {}
    for name, pattern in patterns:
        n = sum(1 for _ in pattern.finditer(text))
        if n:
            out[name] = n
    return out


def scan_text(text: str, prompt: bool = False) -> dict:
    """Return what was found. `prompt=True` adds the hindsight words banned from my own prompts.

    A corpus document is rejected on any `reject` hit and quarantined on a `flag` hit or on
    too many late years. A prompt fails on any hit at all, and on a single late year.
    """
    t = terms()
    late_years = t["late_year"].findall(text)
    result = {
        "reject": _hits(t["reject"], text),
        "flag": _hits(t["flag"], text),
        "late_years": len(late_years),
    }
    if prompt:
        result["prompt_only"] = _hits(t["prompt_only"], text)
        result["clean"] = not (result["reject"] or result["flag"] or result["prompt_only"] or late_years)
    else:
        result["rejected"] = bool(result["reject"])
        result["flagged"] = bool(result["flag"]) or len(late_years) >= t["late_year_min"]
        result["clean"] = not (result["rejected"] or result["flagged"])
    return result


def context(text: str, name: str, width: int = 80) -> list[str]:
    """Snippets around each hit of one term, for the review of flagged documents."""
    t = terms()
    for group in ("reject", "flag", "prompt_only"):
        for term_name, pattern in t[group]:
            if term_name == name:
                return [text[max(0, m.start() - width): m.end() + width].replace("\n", " ") for m in pattern.finditer(text)]
    raise KeyError(name)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path, scan_text(Path(path).read_text(errors="replace")))
