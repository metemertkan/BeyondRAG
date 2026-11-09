"""Pure ranking helpers: Reciprocal Rank Fusion and context ordering. No I/O."""

from __future__ import annotations

from typing import Any, Dict, Final, List, Sequence, TypeVar

RRF_K: Final[int] = 60
T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Dict[str, Any]]],
    *,
    key: str = "chunk_id",
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """Merge rankings using rank positions only.

    BM25 scores and cosine similarities are on different scales, so adding them lets one
    side dominate. RRF needs no tuning between them: score = sum of 1 / (k + rank).
    """
    scores: Dict[Any, float] = {}
    best: Dict[Any, Dict[str, Any]] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            identity = item[key]
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            best.setdefault(identity, item)
    ordered = sorted(scores, key=lambda identity: scores[identity], reverse=True)
    return [{**best[identity], "score": round(scores[identity], 6)} for identity in ordered]


def reorder_for_attention(ranked: Sequence[T]) -> List[T]:
    """Place the best items at the edges of the context and the weakest in the middle.

    Input is best-first. Output: 1st, 3rd, 5th, ..., 6th, 4th, 2nd, so the top item comes
    first, the second comes last, and the lowest-ranked item sits in the middle.
    """
    front = list(ranked[0::2])
    back = list(ranked[1::2])
    return front + back[::-1]
