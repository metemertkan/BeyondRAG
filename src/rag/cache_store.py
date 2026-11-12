"""Where the caches live: Redis when ``REDIS_URL`` is set, process memory otherwise.

Every key starts with ``RAG_CACHE_PREFIX`` (default ``beyondrag``), so a test run or a
benchmark can share a Redis with the app without touching its entries.

Cache entries always carry a TTL. With ``maxmemory-policy volatile-lru`` Redis evicts only
keys that have one, so anything that must not be evicted is written without it.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)


def cache_prefix() -> str:
    return os.getenv("RAG_CACHE_PREFIX", "beyondrag").strip() or "beyondrag"


class CacheStore(Protocol):
    kind: str

    def get(self, key: str) -> Optional[bytes]: ...
    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]: ...
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None: ...
    def mset(self, items: Dict[str, bytes], ttl: Optional[int] = None) -> None: ...
    def delete(self, keys: Iterable[str]) -> int: ...
    def incr(self, key: str) -> int: ...
    def incrby(self, key: str, amount: int) -> int: ...
    def sadd(self, key: str, members: Iterable[str]) -> None: ...
    def smembers(self, key: str) -> List[str]: ...
    def keys(self, pattern: str) -> List[str]: ...
    def vector_add(self, key: str, vector: Sequence[float], scope: str, payload: bytes, ttl: Optional[int]) -> None: ...
    def vector_search(self, vector: Sequence[float], scope: str, k: int = 3) -> List[Tuple[str, float, bytes]]: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class MemoryStore:
    """One process, nothing shared. Used by the tests and when no Redis is configured."""

    kind = "memory"

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._vectors: Dict[str, Tuple[List[float], str, bytes, Optional[float]]] = {}
        self._lock = threading.Lock()

    def _alive(self, expires: Optional[float]) -> bool:
        return expires is None or expires > time.time()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            item = self._data.get(key)
            if item is None or not self._alive(item[1]):
                self._data.pop(key, None)
                return None
            return item[0] if isinstance(item[0], bytes) else str(item[0]).encode()

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        return [self.get(key) for key in keys]

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + ttl if ttl else None)

    def mset(self, items: Dict[str, bytes], ttl: Optional[int] = None) -> None:
        for key, value in items.items():
            self.set(key, value, ttl)

    def delete(self, keys: Iterable[str]) -> int:
        removed = 0
        with self._lock:
            for key in keys:
                removed += (self._data.pop(key, None) is not None) + (self._vectors.pop(key, None) is not None)
        return removed

    def incr(self, key: str) -> int:
        return self.incrby(key, 1)

    def incrby(self, key: str, amount: int) -> int:
        with self._lock:
            value = int(self._data.get(key, (0, None))[0]) + int(amount)
            self._data[key] = (value, None)
            return value

    def sadd(self, key: str, members: Iterable[str]) -> None:
        with self._lock:
            current = self._data.get(key, (set(), None))[0]
            self._data[key] = (set(current) | set(members), None)

    def smembers(self, key: str) -> List[str]:
        with self._lock:
            value = self._data.get(key, (set(), None))[0]
            return sorted(value) if isinstance(value, set) else []

    def keys(self, pattern: str) -> List[str]:
        with self._lock:
            return [k for k in list(self._data) + list(self._vectors) if fnmatch.fnmatchcase(k, pattern)]

    def vector_add(self, key: str, vector: Sequence[float], scope: str, payload: bytes, ttl: Optional[int]) -> None:
        with self._lock:
            self._vectors[key] = (list(vector), scope, payload, time.time() + ttl if ttl else None)

    def vector_search(self, vector: Sequence[float], scope: str, k: int = 3) -> List[Tuple[str, float, bytes]]:
        with self._lock:
            hits = [
                (key, _cosine(vector, entry[0]), entry[2])
                for key, entry in self._vectors.items()
                if entry[1] == scope and self._alive(entry[3])
            ]
        return sorted(hits, key=lambda hit: -hit[1])[:k]


class RedisStore:
    """Redis 8: plain keys for exact entries, a vector index (FLAT, cosine) for the semantic cache."""

    kind = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._redis = redis.Redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=2.0)
        self._redis.ping()
        self._index_dims: Optional[int] = None
        self._index = f"{cache_prefix()}:semantic-index"
        self._vector_prefix = f"{cache_prefix()}:semantic:"

    def get(self, key: str) -> Optional[bytes]:
        return self._redis.get(key)

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        return list(self._redis.mget(keys)) if keys else []

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        self._redis.set(key, value, ex=ttl or None)

    def mset(self, items: Dict[str, bytes], ttl: Optional[int] = None) -> None:
        if not items:
            return
        pipe = self._redis.pipeline(transaction=False)
        for key, value in items.items():
            pipe.set(key, value, ex=ttl or None)
        pipe.execute()

    def delete(self, keys: Iterable[str]) -> int:
        # An entry may have a vector too, under its own prefix. Both go.
        keys = list(keys)
        names = keys + [self._vector_prefix + key for key in keys]
        removed = 0
        for start in range(0, len(names), 500):
            removed += int(self._redis.delete(*names[start : start + 500]))
        return removed

    def incr(self, key: str) -> int:
        return int(self._redis.incr(key))

    def incrby(self, key: str, amount: int) -> int:
        return int(self._redis.incrby(key, int(amount)))

    def sadd(self, key: str, members: Iterable[str]) -> None:
        members = list(members)
        if members:
            self._redis.sadd(key, *members)

    def smembers(self, key: str) -> List[str]:
        return sorted(m.decode() for m in self._redis.smembers(key))

    def keys(self, pattern: str) -> List[str]:
        return [k.decode() for k in self._redis.scan_iter(match=pattern, count=1000)]

    def _ensure_index(self, dims: int) -> None:
        if self._index_dims == dims:
            return
        from redis.commands.search.field import TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        try:
            self._redis.ft(self._index).info()
        except Exception:
            self._redis.ft(self._index).create_index(
                [
                    TagField("scope"),
                    VectorField("vector", "FLAT", {"TYPE": "FLOAT32", "DIM": dims, "DISTANCE_METRIC": "COSINE"}),
                ],
                definition=IndexDefinition(prefix=[self._vector_prefix], index_type=IndexType.HASH),
            )
        self._index_dims = dims

    def vector_add(self, key: str, vector: Sequence[float], scope: str, payload: bytes, ttl: Optional[int]) -> None:
        import numpy as np

        self._ensure_index(len(vector))
        name = self._vector_prefix + key
        self._redis.hset(
            name, mapping={"vector": np.asarray(vector, dtype=np.float32).tobytes(), "scope": scope, "body": payload}
        )
        if ttl:
            self._redis.expire(name, ttl)

    def vector_search(self, vector: Sequence[float], scope: str, k: int = 3) -> List[Tuple[str, float, bytes]]:
        import numpy as np
        from redis.commands.search.query import Query

        self._ensure_index(len(vector))
        # The scope is a hash of tenant, roles and corpus generation: an exact tag match.
        query = (
            Query(f"(@scope:{{{scope}}})=>[KNN {k} @vector $vec AS distance]")
            .sort_by("distance")
            .return_fields("distance", "body")
            .dialect(2)
        )
        result = self._redis.ft(self._index).search(
            query, query_params={"vec": np.asarray(vector, dtype=np.float32).tobytes()}
        )
        hits = []
        for doc in result.docs:
            # "payload" is reserved by redis-py's search results, hence "body".
            payload = doc.body if isinstance(doc.body, bytes) else str(doc.body).encode()
            hits.append((doc.id[len(self._vector_prefix):], 1.0 - float(doc.distance), payload))
        return hits


_STORE: Optional[CacheStore] = None
_STORE_LOCK = threading.Lock()


def get_cache_store() -> CacheStore:
    """The shared store. Falls back to memory, with a warning, when Redis is configured but down."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            url = os.getenv("REDIS_URL", "").strip()
            if url:
                try:
                    _STORE = RedisStore(url)
                except Exception as exc:
                    logger.warning("Redis at %s is not reachable; caches use process memory: %s", url, exc)
                    _STORE = MemoryStore()
            else:
                _STORE = MemoryStore()
        return _STORE


def set_cache_store(store: Optional[CacheStore]) -> None:
    """For tests and the benchmark."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store
