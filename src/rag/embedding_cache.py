"""Content-addressed embedding cache.

The key is the embedding manifest (provider, model, dimensions: Episode 2) plus a hash of
the tenant and the normalised text. A new model means new keys, so nothing needs invalidating. A vector of
personal data is personal data, though, so erasing a document also deletes the vectors it
put here (``purge_owner``).

Each entry stores how long the model took to make it, so a hit can report what it saved.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.rag import telemetry
from src.rag.cache_store import CacheStore, cache_prefix, get_cache_store
from src.rag.multilingual import normalize_for_matching

_HEADER = struct.Struct("<f")  # model seconds spent on this vector


def embedding_cache_enabled() -> bool:
    return os.getenv("RAG_EMBEDDING_CACHE", "on").strip().lower() != "off"


class CachingEmbedder:
    """Wraps an embedder (``embed``, ``embed_batch``) and serves repeated texts from the store."""

    def __init__(self, inner: Any, store: Optional[CacheStore] = None, *, ttl_seconds: Optional[int] = None) -> None:
        self._inner = inner
        self._store = store
        self._ttl = ttl_seconds if ttl_seconds is not None else int(os.getenv("RAG_EMBEDDING_CACHE_TTL", str(30 * 86400)))
        self._namespace = hashlib.sha1(inner.manifest.config_identifier.encode()).hexdigest()[:12]
        self._shared = os.getenv("RAG_EMBEDDING_CACHE_SCOPE", "tenant").strip().lower() == "shared"

    @property
    def store(self) -> CacheStore:
        return self._store or get_cache_store()

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def manifest(self) -> Any:
        return self._inner.manifest

    def key(self, text: str, tenant: Optional[str] = None) -> str:
        # Per tenant by default. Shared, a cache hit is faster, and the speed tells one tenant
        # that another has already embedded the same text or asked the same question.
        scope = "shared" if self._shared else (tenant or telemetry.current_tenant() or "none")
        digest = hashlib.sha256(f"{scope}\x00{normalize_for_matching(text)}".encode("utf-8")).hexdigest()
        return f"{cache_prefix()}:emb:{self._namespace}:{digest}"

    def _owner_key(self, tenant: str, doc_id: str) -> str:
        return f"{cache_prefix()}:emb-owner:{tenant}:{doc_id}"

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(list(texts))

    def embed_batch(self, texts: List[str], *, batch_size: int = 32, owner: Optional[Tuple[str, str]] = None) -> List[List[float]]:
        if not texts:
            return []
        keys = [self.key(text, owner[0] if owner else None) for text in texts]
        store = self.store
        cached = store.mget(keys)
        vectors: List[Optional[List[float]]] = [None] * len(texts)
        saved_seconds = 0.0
        missing: Dict[str, List[int]] = {}
        for index, raw in enumerate(cached):
            if raw:
                (seconds,) = _HEADER.unpack_from(raw)
                vectors[index] = np.frombuffer(raw[_HEADER.size :], dtype=np.float32).astype(float).tolist()
                saved_seconds += seconds
            else:
                missing.setdefault(keys[index], []).append(index)
        hits = len(texts) - sum(len(v) for v in missing.values())
        if hits:
            telemetry.cache_lookup("embedding", True, model_seconds=saved_seconds, detail=f"{hits} of {len(texts)} texts")
        if missing:
            telemetry.cache_lookup("embedding", False, detail=f"{len(missing)} of {len(texts)} texts")
            order = list(missing)
            started = time.perf_counter()
            fresh = self._inner.embed_batch([texts[missing[key][0]] for key in order], batch_size=batch_size)
            per_text = (time.perf_counter() - started) / max(1, len(order))
            entries = {}
            for key, vector in zip(order, fresh):
                # Stored as float32. The miss returns the stored value too, so a text has one
                # vector whether it came from the cache or not.
                packed = np.asarray(vector, dtype=np.float32)
                for index in missing[key]:
                    vectors[index] = packed.astype(float).tolist()
                entries[key] = _HEADER.pack(per_text) + packed.tobytes()
            store.mset(entries, ttl=self._ttl)
        if owner is not None:
            store.sadd(self._owner_key(*owner), keys)
        return [v for v in vectors if v is not None]

    def purge_owner(self, tenant: str, doc_ids: Iterable[str]) -> int:
        """Delete the vectors a document put in the cache, and the list of them."""
        store, removed = self.store, 0
        for doc_id in doc_ids:
            owner = self._owner_key(tenant, doc_id)
            removed += store.delete(store.smembers(owner))
            store.delete([owner])
        return removed
