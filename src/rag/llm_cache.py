"""Cache for model calls whose answer is fully decided by the prompt (temperature 0).

It plugs into LangChain's own cache interface, so every chain and structured-output call
built on a model from ``get_llm`` uses it: query rewrites, HyDE, decomposition, translation,
the guardrail judge, the verifier, table planning. ``RAG_LLM_CACHE=on`` turns it on.

The key is LangChain's: the full prompt and the model's parameters. The same input gives the
same output, so an entry can't carry one tenant's documents into another's answer. A hit is
faster, though, which tells the caller that someone sent that exact prompt before.

On a hit LangChain replays the stored usage as if the tokens were spent again. This cache
marks the hit, and telemetry counts it as saved instead (Episode 6 would have charged it).
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from typing import Any, Optional, Sequence

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads

from src.rag import telemetry
from src.rag.cache_store import CacheStore, cache_prefix, get_cache_store

logger = logging.getLogger(__name__)


def llm_cache_enabled() -> bool:
    return os.getenv("RAG_LLM_CACHE", "off").strip().lower() == "on"


class StoreLLMCache(BaseCache):
    def __init__(self, store: Optional[CacheStore] = None, ttl_seconds: Optional[int] = None) -> None:
        self._store = store
        self._ttl = ttl_seconds if ttl_seconds is not None else int(os.getenv("RAG_LLM_CACHE_TTL", str(7 * 86400)))

    @property
    def store(self) -> CacheStore:
        return self._store or get_cache_store()

    def _key(self, prompt: str, llm_string: str) -> str:
        digest = hashlib.sha256(f"{llm_string}\x00{prompt}".encode("utf-8")).hexdigest()
        return f"{cache_prefix()}:llm:{digest}"

    def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Any]]:
        raw = self.store.get(self._key(prompt, llm_string))
        if raw is None:
            telemetry.cache_lookup("llm_call", False)
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # loads() is marked beta in langchain-core 1.x.
                generations = loads(raw.decode("utf-8"), allowed_objects="core")
        except Exception as exc:  # A bad entry is a miss, not a failed answer.
            logger.warning("Unreadable LLM cache entry; treating it as a miss: %s", exc)
            return None
        telemetry.mark_llm_cache_hit()
        return generations

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Any]) -> None:
        self.store.set(self._key(prompt, llm_string), dumps(list(return_val)).encode("utf-8"), ttl=self._ttl)

    def clear(self, **kwargs: Any) -> None:
        store = self.store
        store.delete(store.keys(f"{cache_prefix()}:llm:*"))


_CACHE: Optional[StoreLLMCache] = None


def get_llm_call_cache() -> Optional[StoreLLMCache]:
    global _CACHE
    if not llm_cache_enabled():
        return None
    if _CACHE is None:
        _CACHE = StoreLLMCache()
    return _CACHE
