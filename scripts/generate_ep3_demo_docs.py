#!/usr/bin/env python3
"""Write config/ep3_demo_docs.json: the small Retrieval Lab corpus, from the benchmark fixtures.

Everything in it is invented. Revenue reports are tagged finance, so a public user's multi-hop
question visibly stops at the ACL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep3_fixtures as fixtures  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "config" / "ep3_demo_docs.json"


def main() -> None:
    documents = []

    def add(name: str, text: str, tags: list[str] | None = None) -> None:
        documents.append({"filename": f"{name}.md", "text": text, "tags": tags or []})

    for doc in fixtures.mismatch_corpus()["docs"]:
        add(f"runbook-{doc['id']}", doc["text"])
    hard = fixtures.hard_exact_match_corpus()["docs"]
    errors = [doc for doc in hard if doc["id"].startswith("err-")][:10]
    skus = [doc for doc in hard if doc["id"].startswith("sku-acme") or doc["id"].startswith("sku-zenta")][:12]
    for doc in errors + skus:
        add(doc["id"], doc["text"])
    for doc in fixtures.parent_child_corpus()["docs"][:4]:
        add(f"policy-{doc['id']}", doc["text"])
    for doc in fixtures.multi_hop_corpus()["docs"]:
        add(doc["id"].lower(), doc["text"], ["finance"] if doc["id"].startswith("revenue") else [])
    # Similar is not relevant: every page is about this router, and one states its price.
    for name, text in {
        "aurora-history": "The history of the Aurora router began in 2014, when a small team in Eindhoven built the first prototype. The Aurora router line grew across Europe, and the original Aurora design won an industry award.",
        "aurora-launch": "The Aurora router launch event drew 400 partners. Press coverage of the Aurora router focused on its design, its founders and the story behind the Aurora name.",
        "aurora-reviews": "Reviews of the Aurora router praise its build quality. Analysts compare the Aurora router favorably with rivals and expect the Aurora brand to keep growing.",
        "aurora-support": "Aurora router support is available in eleven languages. Owners of an Aurora router can register the device online and join the Aurora community forum.",
        "catalog-sheet-7": "Catalog sheet 7. The list price is 289 dollars per unit for the AR-7 model, sold under the Aurora name, with volume discounts from 50 units.",
    }.items():
        add(name, text)
    OUT.write_text(json.dumps(documents, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(documents)} documents -> {OUT}")


if __name__ == "__main__":
    main()
