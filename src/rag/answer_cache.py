"""Answer cache: skip retrieval and the model for a question already answered.

Two layers, both off by default (``RAG_ANSWER_CACHE=off|exact|semantic``):

- **exact**: the same normalised question.
- **semantic**: a question whose embedding is close enough (``RAG_SEMANTIC_CACHE_THRESHOLD``)
  and, with the guard on, mentions the same numbers, years, quarters, currencies, names,
  negations and comparison words. ``scripts/ep7_benchmark.py`` measures why the guard exists.

What makes an answer reusable is in the scope, which every key includes:

- the tenant and the caller's effective ACL tags (Episodes 1 and 5): an answer built from
  finance documents is never served to someone without the finance role;
- the synthesis mode, the pipeline options, the model and the prompt version;
- the tenant's **corpus generation**. Every write to the tenant (upload, retag, supersede,
  delete, erasure, sync) bumps it, so an answer from before the change can't be found again.

A bumped generation only makes old entries unreachable. When something is *removed*, the
tenant's entries are deleted outright, because an answer that quotes erased text is a copy
of erased text (Episodes 2 and 3).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.rag import telemetry
from src.rag.cache_store import CacheStore, cache_prefix, get_cache_store

logger = logging.getLogger(__name__)

# Bumped when the generation prompt changes, so answers written by an older prompt expire.
PROMPT_VERSION = "ep7-1"


@dataclass(frozen=True)
class CachePolicy:
    exact: bool = False
    semantic: bool = False
    threshold: float = 0.92
    guard: bool = True
    ttl_seconds: int = 86400

    @classmethod
    def from_env(cls) -> "CachePolicy":
        mode = os.getenv("RAG_ANSWER_CACHE", "off").strip().lower()
        if mode not in {"off", "exact", "semantic"}:
            raise ValueError(f"RAG_ANSWER_CACHE must be off, exact or semantic; got {mode!r}")
        return cls(
            exact=mode in {"exact", "semantic"},
            semantic=mode == "semantic",
            threshold=float(os.getenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.92")),
            guard=os.getenv("RAG_SEMANTIC_CACHE_GUARD", "on").strip().lower() != "off",
            ttl_seconds=int(os.getenv("RAG_ANSWER_CACHE_TTL", "86400")),
        )

    @property
    def enabled(self) -> bool:
        return self.exact or self.semantic


def normalize_question(question: str) -> str:
    text = " ".join(str(question).casefold().split())
    return text.rstrip(" ?.!？。！")


# --- The guard ---------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_QUARTER = re.compile(r"\bq[1-4]\b|\b(first|second|third|fourth) quarter\b", re.IGNORECASE)
_CURRENCIES = {"hkd", "usd", "eur", "rmb", "cny", "gbp", "jpy", "sgd", "$", "€", "£", "¥"}
_NEGATIONS = {"not", "no", "never", "without", "except", "excluding", "none", "nobody", "cannot", "can't", "isn't", "aren't", "doesn't", "don't"}
_COMPARISONS = {
    "min": "min", "minimum": "min", "lowest": "min", "least": "min", "smallest": "min", "floor": "min",
    "max": "max", "maximum": "max", "highest": "max", "most": "max", "largest": "max", "cap": "max", "ceiling": "max",
    "before": "before", "earlier": "before", "after": "after", "later": "after",
    "increase": "up", "raise": "up", "rise": "up", "decrease": "down", "reduce": "down", "cut": "down",
    "first": "first", "last": "last", "old": "old", "previous": "old", "former": "old", "new": "new", "current": "new", "latest": "new",
}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}


def slots(question: str) -> Dict[str, Set[str]]:
    """The parts of a question that change its answer while barely moving its embedding."""
    words = re.findall(r"[\w$€£¥']+", question)
    lowered = [w.casefold() for w in words]
    names = {w.casefold() for i, w in enumerate(words) if i > 0 and w[:1].isupper() and w.casefold() not in _MONTHS}
    return {
        "numbers": {n.replace(",", "") for n in _NUMBER.findall(question)},
        "quarters": {q.group(0).casefold() for q in _QUARTER.finditer(question)},
        "currencies": {w for w in lowered if w in _CURRENCIES},
        "months": {w for w in lowered if w in _MONTHS},
        "negations": {w for w in lowered if w in _NEGATIONS},
        "comparisons": {_COMPARISONS[w] for w in lowered if w in _COMPARISONS},
        "names": names,
    }


def guard_allows(question: str, cached_question: str) -> Tuple[bool, str]:
    mine, theirs = slots(question), slots(cached_question)
    for name in mine:
        if mine[name] != theirs[name]:
            return False, f"{name} differ: {sorted(mine[name])} vs {sorted(theirs[name])}"
    return True, ""


# --- The cache ---------------------------------------------------------------------------


@dataclass
class CacheContext:
    """Everything a lookup and the store that may follow need to agree on."""

    question: str
    tenant: str
    scope: str
    exact_key: str
    vector: Optional[List[float]] = None


class AnswerCache:
    def __init__(self, policy: Optional[CachePolicy] = None, *, store: Optional[CacheStore] = None, embedder: Any = None) -> None:
        self.policy = policy or CachePolicy.from_env()
        self._store = store
        self._embedder = embedder

    @property
    def store(self) -> CacheStore:
        return self._store or get_cache_store()

    def _key(self, *parts: str) -> str:
        return ":".join([cache_prefix(), *parts])

    def generation(self, tenant: str) -> int:
        raw = self.store.get(self._key("gen", tenant))
        return int(raw) if raw else 0

    def context(
        self, question: str, tenant: str, acl_tags: Iterable[str], *, mode: str, options: Dict[str, Any],
        top_k: int, profile: Optional[str], model: str,
    ) -> CacheContext:
        scope_source = json.dumps(
            {
                "tenant": tenant, "acl": sorted(set(acl_tags)), "mode": mode, "options": options, "top_k": top_k,
                "profile": profile or "default", "model": model, "prompt": PROMPT_VERSION, "generation": self.generation(tenant),
            },
            sort_keys=True, default=str,
        )
        scope = hashlib.sha1(scope_source.encode()).hexdigest()
        digest = hashlib.sha256(normalize_question(question).encode()).hexdigest()
        return CacheContext(question, tenant, scope, self._key("ans", tenant, scope, digest))

    def lookup(self, ctx: CacheContext) -> Optional[Dict[str, Any]]:
        if not self.policy.enabled:
            return None
        raw = self.store.get(ctx.exact_key)
        if raw:
            return self._hit("answer_exact", json.loads(raw), 1.0)
        telemetry.cache_lookup("answer_exact", False)
        if not (self.policy.semantic and self._embedder is not None):
            return None
        ctx.vector = self._embedder.embed(ctx.question)
        for _, similarity, payload in self.store.vector_search(ctx.vector, ctx.scope, k=3):
            if similarity < self.policy.threshold:
                break
            entry = json.loads(payload)
            if self.policy.guard:
                allowed, reason = guard_allows(ctx.question, entry["question"])
                if not allowed:
                    telemetry.note("semantic_cache_guard", f"similarity {similarity:.3f}; {reason}")
                    continue
            return self._hit("answer_semantic", entry, similarity)
        telemetry.cache_lookup("answer_semantic", False)
        return None

    def _hit(self, layer: str, entry: Dict[str, Any], similarity: float) -> Dict[str, Any]:
        cost = entry.get("cost", {})
        age = max(0.0, time.time() - float(entry.get("created", time.time())))
        telemetry.cache_lookup(
            layer, True, input_tokens=cost.get("input_tokens", 0), output_tokens=cost.get("output_tokens", 0),
            model_seconds=cost.get("model_seconds", 0.0), detail=f"similarity {similarity:.3f}, age {age:.0f} s",
        )
        telemetry.note("answer_cache_hit", f"{layer}, similarity {similarity:.3f}")
        response = dict(entry["response"])
        response["cache"] = {
            "layer": layer, "similarity": round(similarity, 4), "age_seconds": round(age, 1),
            "cached_question": entry["question"],
        }
        return response

    def store_answer(self, ctx: CacheContext, response: Dict[str, Any], cost: Dict[str, float]) -> None:
        if not self.policy.enabled:
            return
        ttl = self._ttl(response)
        if ttl <= 0:
            return
        entry = {
            "question": ctx.question,
            "response": {k: v for k, v in response.items() if k not in {"telemetry", "guardrails", "cache"}},
            "cost": cost,
            "created": time.time(),
        }
        payload = json.dumps(entry, ensure_ascii=False, default=str).encode()
        self.store.set(ctx.exact_key, payload, ttl=ttl)
        members = [ctx.exact_key]
        if self.policy.semantic and self._embedder is not None:
            vector = ctx.vector or self._embedder.embed(ctx.question)
            self.store.vector_add(ctx.exact_key, vector, ctx.scope, payload, ttl)
        self.store.sadd(self._key("ans-index", ctx.tenant), members)

    def _ttl(self, response: Dict[str, Any]) -> int:
        """Never outlive the earliest expiry date of a source the answer used (Episode 2)."""
        ttl = self.policy.ttl_seconds
        now = datetime.now(timezone.utc)
        for chunk in response.get("authorized_chunks") or []:
            expires = chunk.get("expires_at")
            if not expires:
                continue
            try:
                moment = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            ttl = min(ttl, int((moment - now).total_seconds()))
        return ttl

    def on_change(self, tenant: str, kind: str, doc_ids: List[str]) -> None:
        """Repository listener. Any change: a new generation. A removal: delete the entries too."""
        self.store.incr(self._key("gen", tenant))
        if kind in {"remove", "reset"}:
            self.purge_tenant(tenant)

    def purge_tenant(self, tenant: str) -> int:
        index = self._key("ans-index", tenant)
        removed = self.store.delete(self.store.smembers(index))
        self.store.delete([index])
        return removed
