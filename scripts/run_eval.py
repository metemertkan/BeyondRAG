#!/usr/bin/env python3
"""Run the LLM-as-a-judge gate on local Ollama. Exit codes: 0 passed, 1 quality gate failed, 2 inconclusive.

Show the failure mode on purpose with:  python scripts/run_eval.py --simulate-drift stale_index
Choose another judge with:  RAG_EVAL_OLLAMA_MODEL=<installed model> python scripts/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set before src.config loads .env, which does not override existing variables.
EVAL_MODEL = os.getenv("RAG_EVAL_OLLAMA_MODEL", "qwen3:8b")
os.environ["USE_LOCAL_ONLY"] = "true"
os.environ["OLLAMA_MODEL"] = EVAL_MODEL

from src.config import ELASTICSEARCH_URL, OLLAMA_BASE_URL, _detect_host_ollama_models  # noqa: E402
from src.eval.harness import DEFAULT_DATASET, DRIFT_MODES, FAILED, PASSED, render_markdown, run_evaluation  # noqa: E402
from src.eval.judge import LlmJudge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--simulate-drift", choices=DRIFT_MODES, default="none")
    parser.add_argument("--report-json", type=Path, help="Write the full report here.")
    parser.add_argument("--require-elasticsearch", action="store_true", help="CI mode: never fall back to memory retrieval.")
    args = parser.parse_args()

    # The router would otherwise swap in another installed model without saying so.
    if EVAL_MODEL not in _detect_host_ollama_models(OLLAMA_BASE_URL):
        print(f"INCONCLUSIVE: Ollama model {EVAL_MODEL} is not available at {OLLAMA_BASE_URL}.", file=sys.stderr)
        return 2
    print(f"Judge: Ollama {EVAL_MODEL}", file=sys.stderr)

    report = run_evaluation(
        LlmJudge(),
        es_url=os.getenv("ELASTICSEARCH_URL", ELASTICSEARCH_URL),
        dataset_path=args.dataset,
        drift=args.simulate_drift,
    )
    if args.require_elasticsearch and report.retrieval_backend != "elasticsearch":
        print("INCONCLUSIVE: Elasticsearch is not reachable.", file=sys.stderr)
        return 2

    markdown = render_markdown(report)
    print(markdown)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report.status == PASSED else 1 if report.status == FAILED else 2


if __name__ == "__main__":
    sys.exit(main())
