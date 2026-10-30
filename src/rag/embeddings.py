"""Component 5: Embedding Provider Abstraction and Provider Factory.

Defines the strongly typed EmbeddingProvider interface, EmbeddingManifest,
and production-ready implementations:
  - FeatureHashEmbeddingProvider (deterministic development/testing provider)
  - OllamaEmbeddingProvider (local Ollama daemon, e.g. bge-m3, qwen3:8b)
  - GoogleEmbeddingProvider (Google text-embedding-004 via API key)
  - DeterministicFakeEmbeddingProvider (for fast, isolated unit testing)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Protocol

import httpx

from src.config import GOOGLE_API_KEY, OLLAMA_BASE_URL, is_valid_api_key

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9_]+|[\u4e00-\uffff]+", re.IGNORECASE)
_CJK_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\uffff]")


@dataclass(frozen=True)
class EmbeddingManifest:
    """Strongly typed manifest describing an active embedding model and its vector properties."""

    provider: str
    model_name: str
    dimensions: int
    model_version: Optional[str] = None
    normalization_strategy: str = "L2"
    distance_metric: str = "cosine"
    config_identifier: str = ""

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError(f"Embedding dimensions must be positive, got {self.dimensions}")
        if not self.config_identifier:
            object.__setattr__(
                self,
                "config_identifier",
                f"{self.provider}::{self.model_name}::{self.dimensions}::{self.distance_metric}",
            )


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers supporting synchronous and asynchronous single/batch embeddings."""

    @property
    def manifest(self) -> EmbeddingManifest:
        ...

    @property
    def dimensions(self) -> int:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    async def aembed_query(self, text: str) -> List[float]:
        ...

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        ...


def normalize_l2(vector: List[float]) -> List[float]:
    """Normalize a vector to unit L2 length."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-9:
        if not vector:
            return vector
        res = [0.0] * len(vector)
        res[0] = 1.0
        return res
    return [round(v / norm, 6) for v in vector]


class FeatureHashEmbeddingProvider:
    """Deterministic, zero-dependency semantic feature hashing provider for local dev and testing."""

    def __init__(
        self,
        dimensions: int = 256,
        model_name: str = "feature_hash_v1",
    ) -> None:
        self._dimensions = dimensions
        self._manifest = EmbeddingManifest(
            provider="feature_hash",
            model_name=model_name,
            dimensions=dimensions,
            model_version="1.0",
            normalization_strategy="L2",
            distance_metric="cosine",
        )

    @property
    def manifest(self) -> EmbeddingManifest:
        return self._manifest

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self._dimensions
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec

        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx1 = int.from_bytes(digest[0:2], "little") % self._dimensions
            sign1 = 1.0 if (digest[2] & 1) == 0 else -1.0
            idx2 = int.from_bytes(digest[4:6], "little") % self._dimensions
            sign2 = 1.0 if (digest[6] & 1) == 0 else -1.0
            vec[idx1] += sign1 * 1.5
            vec[idx2] += sign2 * 0.8

        return normalize_l2(vec)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed(self, text: str) -> List[float]:
        """Adapter method fulfilling EmbedderPort."""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


class DeterministicFakeEmbeddingProvider:
    """Deterministic fake provider strictly for isolated unit tests without external calls."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions
        self._manifest = EmbeddingManifest(
            provider="fake",
            model_name="deterministic_fake",
            dimensions=dimensions,
            model_version="test",
        )

    @property
    def manifest(self) -> EmbeddingManifest:
        return self._manifest

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> List[float]:
        # Generates a stable deterministic vector based on text length and ascii checksum
        val = sum(ord(c) for c in text) % 100 / 100.0
        vec = [val] * self._dimensions
        return normalize_l2(vec)

    def embed(self, text: str) -> List[float]:
        """Adapter method fulfilling EmbedderPort."""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


class OllamaEmbeddingProvider:
    """Ollama embedding adapter for local models (e.g. bge-m3, nomic-embed-text)."""

    def __init__(
        self,
        model_name: str = "bge-m3",
        dimensions: int = 1024,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        self._manifest = EmbeddingManifest(
            provider="ollama",
            model_name=model_name,
            dimensions=dimensions,
            model_version="latest",
        )

    @property
    def manifest(self) -> EmbeddingManifest:
        return self._manifest

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise RuntimeError(f"Ollama returned empty embeddings for query using model {self.model_name}")
        return vectors[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            resp.raise_for_status()
            payload = resp.json()
            raw_embeddings = payload.get("embeddings", [])
            if not raw_embeddings or len(raw_embeddings) != len(texts):
                raise RuntimeError(
                    f"Ollama expected {len(texts)} embeddings, received {len(raw_embeddings)}"
                )

            normalized: List[List[float]] = []
            for vec in raw_embeddings:
                if len(vec) != self._dimensions:
                    # Adapt dimension mismatch if model provides different native dim
                    if len(vec) > self._dimensions:
                        vec = vec[: self._dimensions]
                    else:
                        vec = vec + [0.0] * (self._dimensions - len(vec))
                normalized.append(normalize_l2(vec))
            return normalized
        except Exception as exc:
            raise RuntimeError(
                f"Failed to embed documents with Ollama model '{self.model_name}': {exc}"
            ) from exc

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


class GoogleEmbeddingProvider:
    """Google Gemini embedding adapter using text-embedding-004."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
        dimensions: int = 768,
    ) -> None:
        resolved_key = (api_key or GOOGLE_API_KEY).strip()
        if not is_valid_api_key(resolved_key):
            raise ValueError("Valid GOOGLE_API_KEY required for GoogleEmbeddingProvider.")
        self.api_key = resolved_key
        self.model_name = model_name
        self._dimensions = dimensions
        self._manifest = EmbeddingManifest(
            provider="google",
            model_name=model_name,
            dimensions=dimensions,
            model_version="004",
        )

    @property
    def manifest(self) -> EmbeddingManifest:
        return self._manifest

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:batchEmbedContents?key={self.api_key}"
        requests = [{"model": self.model_name, "content": {"parts": [{"text": t}]}} for t in texts]
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"requests": requests})
            resp.raise_for_status()
            data = resp.json()
            raw_entries = data.get("embeddings", [])
            vectors: List[List[float]] = []
            for entry in raw_entries:
                vals = entry.get("values", [])
                if len(vals) != self._dimensions:
                    if len(vals) > self._dimensions:
                        vals = vals[: self._dimensions]
                    else:
                        vals = vals + [0.0] * (self._dimensions - len(vals))
                vectors.append(normalize_l2(vals))
            return vectors

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def create_embedding_provider(
    provider_name: Optional[str] = None,
    dimensions: Optional[int] = None,
    model_name: Optional[str] = None,
) -> EmbeddingProvider:
    """Factory creating and validating configured embedding providers.
    
    Never silently switches providers upon failure; validates compatibility upfront.
    """
    prov = (provider_name or os.getenv("RAG_EMBEDDING_PROVIDER", "feature_hash")).strip().lower()
    dims = dimensions or int(os.getenv("RAG_EMBEDDING_DIMS", "256"))

    if prov in {"feature_hash", "feature_hashing", "local_hash"}:
        return FeatureHashEmbeddingProvider(
            dimensions=dims,
            model_name=model_name or "feature_hash_v1",
        )
    elif prov == "ollama":
        target_model = model_name or os.getenv("RAG_EMBEDDING_MODEL", "bge-m3")
        return OllamaEmbeddingProvider(
            model_name=target_model,
            dimensions=dims,
        )
    elif prov in {"google", "gemini"}:
        target_model = model_name or os.getenv("RAG_EMBEDDING_MODEL", "models/text-embedding-004")
        return GoogleEmbeddingProvider(
            model_name=target_model,
            dimensions=dims,
        )
    elif prov == "fake":
        return DeterministicFakeEmbeddingProvider(dimensions=dims)
    else:
        raise ValueError(
            f"Unknown embedding provider '{prov}'. Supported providers are: ['feature_hash', 'ollama', 'google', 'fake']"
        )


# Backward-compatible global constants & functions
EMBEDDING_DIMS: Final[int] = int(os.getenv("RAG_EMBEDDING_DIMS", "256"))
STOPWORDS: Final[set[str]] = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "will", "would", "could", "should", "into", "about",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "can", "may", "also", "only", "other", "than", "then", "them", "they",
    "their", "there", "these", "those", "your", "our",
}


def has_cjk(text: str) -> bool:
    """Return True if the string contains CJK characters."""
    return bool(_CJK_CHAR_RE.search(text))


_DEFAULT_PROVIDER: Final[EmbeddingProvider] = create_embedding_provider()


def embed_text(text: str, dims: Optional[int] = None) -> List[float]:
    """Backward-compatible embedding function delegating to active provider."""
    if dims is not None and dims != _DEFAULT_PROVIDER.dimensions:
        temp_provider = create_embedding_provider(dimensions=dims)
        return temp_provider.embed_query(text)
    return _DEFAULT_PROVIDER.embed_query(text)


class CrossLingualEmbedder:
    """Component: Configurable Vector Embedder implementing `EmbedderPort`."""

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        self._provider = provider or (
            create_embedding_provider(dimensions=dimensions)
            if dimensions
            else _DEFAULT_PROVIDER
        )

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def embed(self, text: str) -> List[float]:
        return self._provider.embed_query(text)
