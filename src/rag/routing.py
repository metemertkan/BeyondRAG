"""Model routing: send each model call to the cheapest model that does the job.

``RAG_ROUTING``:
- ``off`` (default): every call uses ``OLLAMA_MODEL``, as before.
- ``static``: each task has a tier (``small``, ``medium``, ``large``), from ``ROUTES`` or a
  ``RAG_ROUTE_<TASK>`` override. The defaults come from ``scripts/ep7_benchmark.py``
  (section ``routing_tasks``): a task only moves to a smaller model where that model
  measured as well as the large one.
- ``cascade``: ``static``, and answers are first written by ``RAG_CASCADE_FIRST`` (default
  ``small``). They go to the large model when the cheap Episode 4 checks fail or the small
  model says it can't answer.

``RAG_ROUTE_EXTRACTIVE=on`` adds a tier below all of them: no model. When the re-ranker is
confident and the question asks for one fact, the extractive answer is returned.

Every routed call is labelled with its task, tier and model, so a cheaper model never
answers without anyone knowing (Episode 6 found the model factory doing exactly that).
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

from src.rag import telemetry

TIERS = ("small", "medium", "large")
TASKS = ("query_rewrite", "hyde", "decompose", "step_answer", "translate", "concepts", "judge", "verify", "table", "generate")

# Measured defaults. "large" means no change from before routing.
ROUTES: Dict[str, str] = {task: "large" for task in TASKS}


def routing_mode() -> str:
    mode = os.getenv("RAG_ROUTING", "off").strip().lower()
    if mode not in {"off", "static", "cascade"}:
        raise ValueError(f"RAG_ROUTING must be off, static or cascade; got {mode!r}")
    return mode


def tier_models() -> Dict[str, str]:
    large = os.getenv("RAG_MODEL_LARGE", "").strip() or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    return {
        "small": os.getenv("RAG_MODEL_SMALL", "qwen3:1.7b").strip(),
        "medium": os.getenv("RAG_MODEL_MEDIUM", "qwen3:4b").strip(),
        "large": large.strip(),
    }


def tier_for(task: str) -> str:
    tier = os.getenv(f"RAG_ROUTE_{task.upper()}", "").strip().lower() or ROUTES.get(task, "large")
    if tier not in TIERS:
        raise ValueError(f"RAG_ROUTE_{task.upper()} must be one of {', '.join(TIERS)}; got {tier!r}")
    return tier


def model_for(task: Optional[str]) -> Optional[str]:
    """The model a task is routed to, or None when routing is off or the task is unknown."""
    if not task or task not in TASKS or routing_mode() == "off":
        return None
    tier = tier_for(task)
    model = tier_models()[tier]
    telemetry.route_decision(task, tier, model)
    return model


_SINGLE_FACT = re.compile(
    r"^\s*(what|which|who|when|where|how (much|many|long|often|soon))\b", re.IGNORECASE
)


def is_single_fact(question: str) -> bool:
    """A short what/which/who/when/how-much question. Comparisons and explanations are not."""
    words = question.split()
    lowered = question.casefold()
    if len(words) > 16 or any(w in lowered for w in (" and ", " compare", " differ", " why", "explain", "summar", " versus", " vs")):
        return False
    return bool(_SINGLE_FACT.match(question))


def cascade_first_tier() -> str:
    tier = os.getenv("RAG_CASCADE_FIRST", "small").strip().lower()
    if tier not in TIERS:
        raise ValueError(f"RAG_CASCADE_FIRST must be one of {', '.join(TIERS)}; got {tier!r}")
    return tier


def cascade_first_model() -> Optional[str]:
    """The model that answers first under ``RAG_ROUTING=cascade``, or None."""
    if routing_mode() != "cascade":
        return None
    return tier_models()[cascade_first_tier()]


def extractive_route_enabled() -> bool:
    return os.getenv("RAG_ROUTE_EXTRACTIVE", "off").strip().lower() == "on"


def extractive_min_score() -> float:
    """Re-ranker score (0 to 1) above which a single-fact question gets the extractive answer."""
    return float(os.getenv("RAG_EXTRACTIVE_MIN_SCORE", "0.9"))
