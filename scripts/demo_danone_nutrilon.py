#!/usr/bin/env python3
"""BeyondRAG Live Demo: Danone Nutrilon Recall Products.

Showcases:
1. On-the-fly PDF upload and indexing with RBAC tagging.
2. Role-Based Access Control (RBAC) pre-filtering & Security X-Ray auditing.
3. Cross-lingual retrieval (Chinese & English queries over Dutch recall tables).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.parse

API_BASE = "http://localhost:8090"
PDF_PATH = Path("workspace/Danone_Nutrilon_Recall_Products.pdf")


def upload_pdf_on_the_fly(pdf_path: Path, tag: str = "operation") -> dict:
    print(f"\n📤 [Step 1] Uploading '{pdf_path.name}' on the fly with RBAC tag: '{tag}'...")
    boundary = "----WebKitFormBoundaryBeyondRAGDemo"
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + pdf_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="tags"\r\n\r\n'
        f"{tag}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        doc = data.get("document", {})
        print(f"   ✅ Indexed: Doc ID: {doc.get('doc_id')}")
        print(f"   📄 Pages: {doc.get('page_count')} | Chunks: {doc.get('chunk_count')}")
        print(f"   🏷️ ACL Tags: {doc.get('acl_tags')}")
        print(f"   🧠 Dynamic Concepts: {doc.get('dynamic_concepts')}")
        return data


def ask_question(question: str, user_roles: list[str], synthesis_mode: str = "fast") -> dict:
    payload = {
        "question": question,
        "user_roles": user_roles,
        "top_k": 3,
        "synthesis_mode": synthesis_mode,
    }
    req = urllib.request.Request(
        f"{API_BASE}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        elapsed = time.time() - t0
        res = json.loads(resp.read().decode("utf-8"))
        res["_roundtrip_seconds"] = round(elapsed, 2)
        return res


def main():
    if not PDF_PATH.exists():
        print(f"Error: {PDF_PATH} does not exist. Please place the PDF in workspace/.")
        sys.exit(1)

    print("=" * 70)
    print("🚀 BeyondRAG — Live Dynamic Ingestion & Retrieval Showcase")
    print("=" * 70)

    # 1. Upload on the fly
    upload_pdf_on_the_fly(PDF_PATH, tag="operation")

    # 2. Scenario A: RBAC Pre-Filtering Block (Unauthorized Role: Finance / Public)
    print("\n" + "=" * 70)
    print("🔒 SCENARIO A: Role-Based Access Control Block")
    print("User Role: ['finance'] (Unauthorized for Operations recall document)")
    print("Query (ZH): '召回的 Nutrilon Duobalans 1 批號和保質期是多少？'")
    print("-" * 70)
    res_a = ask_question("召回的 Nutrilon Duobalans 1 批號和保質期是多少？", user_roles=["finance"])
    print(f"Status: Cross-Lingual={res_a['is_cross_lingual']} | Filter={res_a['effective_acl_filter']}")
    print(f"Answer:\n{res_a['answer']}")
    if res_a.get("masked_restricted_docs"):
        print("\n🔍 Security X-Ray (Masked Restricted Documents):")
        for m in res_a["masked_restricted_docs"]:
            print(f"   • {m['filename']} (p. {m['page_number']}) — Requires: {m['required_any_of_tags']}")

    # 3. Scenario B: RBAC Unlocked & Cross-Lingual Concept Mapping (Traditional Chinese -> Dutch PDF)
    print("\n" + "=" * 70)
    print("🌐 SCENARIO B: Cross-Lingual Search (Traditional Chinese Query -> Dutch PDF)")
    print("User Role: ['operation'] (Authorized)")
    print("Query (ZH): '召回的 Nutrilon Duobalans 1 批號和保質期是多少？'")
    print("-" * 70)
    res_b = ask_question("召回的 Nutrilon Duobalans 1 批號和保質期是多少？", user_roles=["operation"])
    print(f"Status: Cross-Lingual={res_b['is_cross_lingual']}")
    print(f"Matched Concepts: {res_b['matched_concepts']}")
    print(f"Expanded Query: '{res_b['expanded_query']}'")
    print(f"Latency: {res_b['latency_ms']} ms")
    print(f"Answer:\n{res_b['answer']}")

    # 4. Scenario C: Grounded LLM Synthesis with Exact Dutch Batch Table
    print("\n" + "=" * 70)
    print("🤖 SCENARIO C: Grounded Synthesis with Local Ollama")
    print("User Role: ['operation'] (Authorized)")
    print("Query: 'Nutrilon AR 2 batch nummer'")
    print("-" * 70)
    res_c = ask_question("Nutrilon AR 2 batch nummer", user_roles=["operation"], synthesis_mode="llm")
    print(f"Answer:\n{res_c['answer']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
