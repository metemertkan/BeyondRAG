#!/usr/bin/env python3
"""Read the Episode 8 verdicts, one claim at a time, and record my reading of each.

  python scripts/ep8_read.py show E04            # every verdict for a claim, each passage printed once
  python scripts/ep8_read.py grep "local time"   # search the June 1905 library (to spot retrieval misses)
  python scripts/ep8_read.py status              # how many verdicts have a reading

Readings live in <run>/reading.jsonl, one line per verdict:
  {"library", "claim", "phrasing", "reading": fair | judge error | retrieval miss,
   "should_be": supported | contradicted | not stated, "note"}
`fair` means the label follows from the four sources the checker saw. `judge error` means the
sources say otherwise. `retrieval miss` means the label is fair for those four, but the library
holds a passage that states the claim.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ep8_claims as claims  # noqa: E402
import ep8_library as library  # noqa: E402

RUN = Path(os.environ.get("EP8_RUN", str(library.HOME / "runs" / "ep8")))
ABBR = {"supported": "SUP", "contradicted": "CON", "not stated": "n/s"}


def rows():
    return [json.loads(l) for l in (RUN / "verdicts.jsonl").read_text().splitlines() if l.strip()]


def readings():
    path = RUN / "reading.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["library"], r["claim"], r["phrasing"])] = r
    return out


def show(claim_id: str, width: int = 650) -> None:
    claim = next(c for c in claims.CLAIMS if c["id"] == claim_id)
    print(f"### {claim_id} ({claim['group']}{', ' + claim['kind'] if claim.get('kind') else ''})")
    for n, text in enumerate(claim["phrasings"], 1):
        print(f"  {n}: {text}")
    if claim.get("formula"):
        print(f"  f: {claim['formula']}")
    if claim.get("quote"):
        print(f"  paper: {claim['quote'][:300]}")
    mine = [r for r in rows() if r["claim"] == claim_id]
    keys: dict = {}
    for r in mine:
        for s in r["sources"]:
            keys.setdefault(s["text"], (f"P{len(keys) + 1}", s))
    print("\n-- passages")
    for text, (key, s) in keys.items():
        print(f"[{key}] {s['title'][:60]} ({s['author'][:20]}, {s['year']})\n    {text[:width]}")
    print("\n-- verdicts")
    done = readings()
    for r in mine:
        srcs = " ".join(keys[s["text"]][0] for s in r["sources"])
        mark = done.get((r["library"], r["claim"], r["phrasing"]), {}).get("reading", "")
        reason = " / ".join(x[:260] for x in r["reasons"])
        print(f"{r['library'][:4]} .{r['phrasing']} {ABBR.get(r['verdict'], r['verdict'])} [{srcs}] {mark}  {reason}")


def grep(pattern: str, date: str = "1905") -> None:
    cut = next(d for d in library.LIBRARY_DATES if str(d[0]) == date)
    rx = re.compile(pattern, re.I)
    for line in library.MANIFEST.read_text().splitlines():
        r = json.loads(line)
        if not library.in_library(r, cut):
            continue
        text = re.sub(r"\s+", " ", library.read_text(r))
        for m in list(rx.finditer(text))[:3]:
            print(f"{r['title'][:50]} ({r['year']}): ...{text[max(0, m.start() - 220):m.end() + 220]}...")


def record(claim_id: str, lines: str) -> None:
    """Lines of `YEAR PHRASING | reading | should_be | note`. YEAR may be `*` (every library) or
    a comma list; PHRASING may be `*`. Later lines override earlier ones."""
    mine = [r for r in rows() if r["claim"] == claim_id]
    done = readings()
    for line in lines.strip().splitlines():
        if not line.strip():
            continue
        head, reading, should, *note = [p.strip() for p in line.split("|")]
        years, phr = head.split()
        assert reading in ("fair", "judge error", "retrieval miss"), reading
        assert should in ("supported", "contradicted", "not stated"), should
        for r in mine:
            if (years == "*" or r["library"][:4] in years.split(",")) and (phr == "*" or r["phrasing"] in phr.split(",")):
                done[(r["library"], r["claim"], r["phrasing"])] = {
                    "library": r["library"], "claim": claim_id, "phrasing": r["phrasing"], "verdict": r["verdict"],
                    "reading": reading, "should_be": should, "note": note[0] if note else ""}
    (RUN / "reading.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in done.values()))
    left = [r for r in mine if (r["library"], r["claim"], r["phrasing"]) not in done]
    for v in done.values():
        if v["claim"] == claim_id and v["reading"] == "fair" and v["should_be"] != v["verdict"]:
            print("inconsistent:", v)
    print(f"{claim_id}: {len(mine) - len(left)} of {len(mine)} read" + (f"; missing {[(r['library'][:4], r['phrasing']) for r in left]}" if left else ""))


def status() -> None:
    all_rows, done = rows(), readings()
    print(f"{sum((r['library'], r['claim'], r['phrasing']) in done for r in all_rows)} of {len(all_rows)} read")
    from collections import Counter
    print(Counter(r["reading"] for r in done.values()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "show":
        show(sys.argv[2])
    elif cmd == "record":
        record(sys.argv[2], sys.stdin.read())
    elif cmd == "grep":
        grep(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "1905")
    else:
        status()
