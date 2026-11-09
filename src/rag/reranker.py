"""Cross-encoder re-ranking for the second stage of retrieval.

A bi-encoder embeds the query and the chunk separately, so it measures similarity. A
cross-encoder reads both together and scores whether the chunk answers the query. It is too
slow to run over a whole index, so it only sees the candidates that the first stage returned.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Final, List, Optional, Sequence

DEFAULT_RERANKER_MODEL: Final[str] = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base").strip()


class CrossEncoderUnavailable(RuntimeError):
    """The optional dependency or the model could not be loaded."""


class FastEmbedCrossEncoder:
    """Local ONNX cross-encoder. Install with ``pip install .[rerank]``."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        loader: Optional[Callable[[str], Any]] = None,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._loader = loader
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._loader is not None:
            self._model = self._loader(self._model_name)
            return self._model
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise CrossEncoderUnavailable("Install the reranker with: pip install .[rerank]") from exc
        try:
            self._model = TextCrossEncoder(self._model_name)
        except Exception as exc:
            raise CrossEncoderUnavailable(f"Could not load reranker {self._model_name}: {exc}") from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        return [float(value) for value in self._load().rerank(query, list(texts), batch_size=self._batch_size)]
