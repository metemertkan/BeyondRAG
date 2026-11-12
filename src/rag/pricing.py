"""Turn tokens into an illustrative dollar figure.

Everything here runs on local models, which cost CPU time, not money. To make token counts
comparable with a hosted bill, ``config/model_prices.json`` maps each local model to a hosted
model of a similar tier and gives that model's list price, with the source and the date it
was read. The figure is an estimate and is labelled as one wherever it appears.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "model_prices.json"


@lru_cache(maxsize=1)
def price_table() -> Dict[str, Any]:
    path = Path(os.getenv("RAG_PRICE_TABLE", str(_DEFAULT)))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"per_million_tokens": {}, "local_to_hosted": {}}


def price_for(model: str) -> Optional[Dict[str, float]]:
    table = price_table()
    hosted = table.get("local_to_hosted", {}).get(model, model)
    return table.get("per_million_tokens", {}).get(hosted)


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = price_for(model)
    if not price:
        return 0.0
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
