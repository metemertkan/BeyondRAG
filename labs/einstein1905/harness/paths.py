"""Where things live. Heavy data stays outside the repo, in E1905_HOME."""

from __future__ import annotations

import os
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
E1905_HOME = Path(os.environ.get("E1905_HOME", "/home/server/Projects/AIContentCreator/einstein1905-data"))
MODELS = E1905_HOME / "models"
SAMPLES = E1905_HOME / "samples"
RESULTS = LAB / "results"
PROBES = LAB / "probes"

# label -> (Hugging Face repo, pinned revision, checkpoint step)
HISTORICAL = {
    "g1905": ("mhla/gpt1905-d34", "54ccc781fd576194e7737f4aa9f0d90aa08d1096", 19103),
    "g1964": ("mhla/gpt1964-d34", "271bcf68f5124c862ea72df7a66523cd0bd10229", None),
    "g1900": ("mhla/gpt1900-d34-22btok", None, None),
}
