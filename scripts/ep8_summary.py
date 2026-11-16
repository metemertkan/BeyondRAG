#!/usr/bin/env python3
"""Summarise the Episode 8 run and my reading of it, and copy both to the episode's bench folder.

  python scripts/ep8_summary.py [--run DIR] [--out DIR]

summary.json holds, per claim group and library date, the counts of supported / contradicted /
not stated, as the checker labelled them and as the sources support them after reading; how often
the two agree; the kinds of judge error; and, per claim, the label at each date.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import ep8_claims as claims
import ep8_library as library

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "ContentEngine" / "episodes" / "ep08-beyondrag" / "bench"
LABELS = ("supported", "contradicted", "not stated")


def group_of(claim: dict) -> str:
    if claim["group"] != "W":
        return claim["group"]
    return "W invented" if claim["kind"] == "invented" else "W of the day"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(library.HOME / "runs" / "ep8"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    rows = [json.loads(l) for l in (run / "verdicts.jsonl").read_text().splitlines() if l.strip()]
    reading = {(r["library"], r["claim"], r["phrasing"]): r
               for r in map(json.loads, (run / "reading.jsonl").read_text().splitlines()) if r}
    missing = [(r["library"], r["claim"], r["phrasing"]) for r in rows if (r["library"], r["claim"], r["phrasing"]) not in reading]
    assert not missing, f"{len(missing)} verdicts have no reading"
    by_id = {c["id"]: c for c in claims.CLAIMS}
    dates = sorted({r["library"] for r in rows})
    groups = ["E", "K", "W invented", "W of the day"]

    def counts(select, key):
        c = Counter(key(r) for r in rows if select(r))
        return {label: c[label] for label in LABELS}

    table = {}
    for g in groups:
        table[g] = {}
        for d in dates:
            pick = lambda r, g=g, d=d: r["library"] == d and r["phrasing"] != "f" and group_of(by_id[r["claim"]]) == g
            table[g][d] = {
                "checker": counts(pick, lambda r: r["verdict"]),
                "after_reading": counts(pick, lambda r: reading[(r["library"], r["claim"], r["phrasing"])]["should_be"]),
            }
    per_claim = {}
    for c in claims.CLAIMS:
        per_claim[c["id"]] = {
            "group": group_of(c), "phrasings": c["phrasings"], "formula": c.get("formula"),
            "section": c.get("section"), "quote": c.get("quote"), "source": c.get("source"),
            "by_date": {d: {r["phrasing"]: {"checker": r["verdict"],
                                            "after_reading": reading[(d, r["claim"], r["phrasing"])]["should_be"],
                                            "reading": reading[(d, r["claim"], r["phrasing"])]["reading"],
                                            "note": reading[(d, r["claim"], r["phrasing"])]["note"],
                                            "top_source": f"{r['sources'][0]['title']} ({r['sources'][0]['year']})" if r["sources"] else None}
                            for r in rows if r["claim"] == c["id"] and r["library"] == d}
                        for d in dates},
        }
    agree = sum(v["verdict"] == v["should_be"] for v in reading.values())
    errors = Counter(f"{v['verdict']} -> {v['should_be']}" for v in reading.values() if v["reading"] == "judge error")
    manifest = [json.loads(l) for l in library.MANIFEST.read_text().splitlines() if l.strip()]
    summary = {
        "checker": "src/rag/trust.py AnswerVerifier(LlmClaimVerifier), local qwen3:8b; app retrieval, rerank, top 4 of 20",
        "verdicts": len(rows),
        "library": {d: sum(library.in_library(m, tuple(int(x) for x in d.split("-"))) for m in manifest) for d in dates},
        "readings": dict(Counter(v["reading"] for v in reading.values())),
        "agreement_checker_vs_reading": round(agree / len(reading), 3),
        "judge_errors": dict(errors),
        "table": table,
        "claims": per_claim,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    for name in ("verdicts.jsonl", "reading.jsonl"):
        shutil.copy(run / name, out / name)
    shutil.copy(library.MANIFEST, out / "library_manifest.jsonl")
    print(f"{len(rows)} verdicts, agreement {summary['agreement_checker_vs_reading']}, written to {out}")


if __name__ == "__main__":
    main()
