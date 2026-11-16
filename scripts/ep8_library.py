#!/usr/bin/env python3
"""Build the Episode 8 library: physics of moving bodies from English Wikisource, each work dated.

  python scripts/ep8_library.py fetch      # list the works, fetch their text, write the manifest
  python scripts/ep8_library.py report     # what the manifest holds, per library date

Texts go to $E1905_HOME/library/texts/ (outside the repo). The manifest goes to
config/ep8_library.jsonl: one line per work with its date and where the date came from.
Dates read from the page header are a first guess; `checked` is set by hand after reading.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("E1905_HOME", "/home/server/Projects/AIContentCreator/einstein1905-data"))
TEXTS = HOME / "library" / "texts"
MANIFEST = ROOT / "config" / "ep8_library.jsonl"
API = "https://en.wikisource.org/w/api.php"
UA = {"User-Agent": "BeyondRAG-ep8/0.1 (research script; library of public-domain physics texts)"}

# The paper under test. Never in any library, or it would support itself.
SUBJECT = "On the Electrodynamics of Moving Bodies (1920 edition)"
PORTAL = "Portal:Relativity"
# Works outside the portal that belong in the library: Poincaré on time, relativity and hypotheses.
EXTRA = [
    ("Science and Hypothesis", None),
    ("The Measure of Time", None),
    ("The Foundations of Science/The Value of Science", ["Chapter 7", "Chapter 8", "Chapter 9"]),
]
# Portal entries that aren't texts to check claims against.
SKIP = {"The Einstein Theory of Relativity (film)"}
MIN_WORDS = 150


def api(**params: Any) -> Dict[str, Any]:
    params["format"] = "json"
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503):
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("Wikisource keeps refusing; try later")


def polite() -> None:
    time.sleep(1.5)


def portal_titles() -> List[str]:
    d = api(action="parse", page=PORTAL, prop="links")
    return [l["*"] for l in d["parse"]["links"] if l["ns"] in (0, 114) and "exists" in l]


def wikitext(title: str) -> str:
    d = api(action="query", prop="revisions", rvprop="content", rvslots="main", titles=title, redirects=1)
    page = next(iter(d["query"]["pages"].values()))
    if "revisions" not in page:
        return ""
    return page["revisions"][0]["slots"]["main"]["*"]


def header_fields(text: str) -> Dict[str, str]:
    """year, author, translator and override_author from {{header}} or {{translation header}}."""
    fields = {}
    for name in ("year", "author", "translator", "override_author", "notes", "override_year"):
        m = re.search(r"\|\s*" + name + r"\s*=\s*([^|\n}]*)", text)
        if m and m.group(1).strip():
            fields[name] = m.group(1).strip()
    # Notes run over several lines and often hold the exact issue or session date.
    m = re.search(r"\|\s*notes\s*=\s*(.*?)\n\s*(?:\|\s*\w+\s*=|\}\})", text, re.S)
    if m and m.group(1).strip():
        fields["notes"] = re.sub(r"\s+", " ", m.group(1)).strip()[:400]
    return fields


def subpages(title: str) -> List[str]:
    ns = 114 if title.startswith("Translation:") else 0
    base = title.split(":", 1)[1] if ns else title
    out: List[str] = []
    cont: Dict[str, str] = {}
    while True:
        d = api(action="query", list="allpages", apprefix=base + "/", apnamespace=ns, aplimit=500, **cont)
        out += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


_DROP_BLOCKS = re.compile(
    r'<(div|table|span|ol|ul)[^>]*class="[^"]*(?:ws-noexport|licenseContainer|authortemplate|noprint|ws-header|wst-header|'
    r'header_notes|references|reflist|mw-references|pagenum|ws-pagenum|navbox|wst-translation-license)[^"]*"[^>]*>.*?</\1>',
    re.S,
)


def page_text(title: str) -> Optional[str]:
    return page(title)[0]


def page(title: str) -> tuple:
    """(rendered text with formulas kept as TeX between $ signs, wikitext) in one request."""
    d = api(action="parse", page=title, prop="text|wikitext", redirects=1, disabletoc=1, disableeditsection=1)
    if "error" in d:
        return None, ""
    source = d["parse"].get("wikitext", {}).get("*", "")
    h = d["parse"]["text"]["*"]
    h = re.sub(
        r'<span class="mwe-math-element[^"]*".*?<annotation encoding="application/x-tex">(.*?)</annotation>.*?</span>\s*(<img[^>]*>)?(</span>)*',
        lambda m: " $" + html.unescape(m.group(1)).replace("{\\displaystyle", "").strip().rstrip("}").strip() + "$ ",
        h,
        flags=re.S,
    )
    h = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", h, flags=re.S)
    for _ in range(3):  # nested blocks
        h = _DROP_BLOCKS.sub(" ", h)
    h = re.sub(r"<sup[^>]*class=\"reference\"[^>]*>.*?</sup>", " ", h, flags=re.S)
    h = re.sub(r"</(p|div|h\d|li|tr|dd|dt)>|<br\s*/?>", "\n", h)
    t = html.unescape(re.sub(r"<[^>]+>", "", h))
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip(), source


_LICENSE_LINE = re.compile(
    r"public domain|Public domainPublic domain|copyright term|longest-living author|Creative Commons|CC-BY|"
    r"is licensed under|GNU Free Documentation|^(Original|Translation):\s*$",
    re.I,
)


def clean_text(text: str, year: Optional[int]) -> str:
    """Wikisource licence lines out, and any notes the author added to a later reprint.

    Voigt's 1887 Doppler paper, for example, ends with "Notes by the author (1915)"."""
    lines = []
    for line in text.splitlines():
        later = re.match(r"\s*Notes by the author \((\d{4})\)", line)
        if later and year and int(later.group(1)) > year:
            break
        if _LICENSE_LINE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def read_text(record: Dict[str, Any]) -> str:
    return clean_text((TEXTS / f"{record['id']}.txt").read_text(), record.get("year"))


def slug(title: str) -> str:
    s = re.sub(r"^Translation:", "tr-", title)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:90]


def fetch() -> None:
    TEXTS.mkdir(parents=True, exist_ok=True)
    existing = {}
    if MANIFEST.exists():
        existing = {json.loads(l)["title"]: json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()}
    titles = [(t, None) for t in portal_titles() if t != SUBJECT and t not in SKIP]
    polite()
    titles += EXTRA
    print(f"{len(titles)} works to consider", flush=True)
    records = []
    for n, (title, only) in enumerate(titles, 1):
        if title in existing and (TEXTS / (existing[title]["id"] + ".txt")).exists():
            records.append(existing[title])
            continue
        main, source = page(title)
        polite()
        fields = header_fields(source)
        parts = []
        if main:
            parts.append(main)
        # A contents page: the text is in its subpages.
        if not main or len(main.split()) < 1500 or only:
            subs = subpages(title)
            polite()
            if only:
                subs = [s for s in subs if any(s.endswith("/" + k) for k in only)]
            if subs:
                parts = [] if only else parts
                for sub in subs:
                    text = page_text(sub)
                    polite()
                    if text:
                        parts.append(f"## {sub}\n\n{text}")
        text = "\n\n".join(parts)
        words = len(text.split())
        rid = slug(title)
        (TEXTS / f"{rid}.txt").write_text(text)
        year = fields.get("override_year") or fields.get("year") or ""
        m = re.search(r"\d{4}", year)
        record = {
            "id": rid,
            "title": title,
            "author": re.sub(r"\[\[|\]\]|Author:", "", fields.get("override_author") or fields.get("author") or ""),
            "translator": fields.get("translator", ""),
            "year": int(m.group(0)) if m else None,
            "month": None,
            "date_source": "wikisource header" if m else "missing",
            "words": words,
            "mentions_einstein": len(re.findall(r"Einstein", text)),
            "header_notes": fields.get("notes", ""),
            "checked": False,
            "include": words >= MIN_WORDS,
            "notes": "",
        }
        records.append(record)
        print(f"{n:3}/{len(titles)} {record['year'] or '????'} {words:7} words  {title[:70]}", flush=True)
        MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


LIBRARY_DATES = [(1905, 6, 30), (1908, 12, 31), (1911, 12, 31), (1914, 12, 31), (1921, 12, 31)]


def in_library(record: Dict[str, Any], date: tuple) -> bool:
    """Published on or before the library date. A work dated by year only counts from 31 December,
    and one dated by month from the last day of that month, so an unknown day never lets a work in early."""
    if not record.get("include") or record.get("year") is None:
        return False
    published = (record["year"], record.get("month") or 12, record.get("day") or 31)
    return published <= tuple(date)


def report() -> None:
    records = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    print(f"{len(records)} works, {sum(r['include'] for r in records)} included, {sum(r['checked'] for r in records)} checked")
    for date in LIBRARY_DATES:
        works = [r for r in records if in_library(r, date)]
        print(f"library {date[0]}-{date[1]:02}: {len(works):3} works, {sum(r['words'] for r in works):8} words")
    missing = [r["title"] for r in records if r["year"] is None]
    if missing:
        print("no date:", *missing, sep="\n  ")
    early = [r for r in records if r["year"] and r["year"] <= 1905 and r["mentions_einstein"]]
    if early:
        print("dated 1905 or earlier but names Einstein (check by hand):")
        for r in early:
            print(f"  {r['year']} {r['title']} ({r['mentions_einstein']}x)")


if __name__ == "__main__":
    {"fetch": fetch, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
