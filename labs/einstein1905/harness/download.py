"""Download one historical model at a pinned revision and record what arrived.

Only the weights of one step, its meta file, the tokenizer and the loading code are fetched.
The repos also hold tens of GB of optimizer state, which is never needed here.

    uv run python -m harness.download g1905
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from harness.paths import HISTORICAL, MODELS, RESULTS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def pick_step(repo: str, revision: str | None) -> int:
    """The highest step that has a model file."""
    files = HfApi().list_repo_files(repo, revision=revision)
    steps = [int(f[len("model_"):-len(".pt")]) for f in files if f.startswith("model_") and f.endswith(".pt")]
    return max(steps)


def main(label: str) -> None:
    repo, revision, step = HISTORICAL[label]
    if revision is None:
        revision = HfApi().model_info(repo).sha
    if step is None:
        step = pick_step(repo, revision)
    target = MODELS / label
    snapshot_download(
        repo,
        revision=revision,
        local_dir=target,
        allow_patterns=[f"model_{step:06d}.pt", f"meta_{step:06d}.json", "tokenizer/*", "nanochat/*", "README.md"],
    )
    record = {
        "repo": repo,
        "revision": revision,
        "step": step,
        "files": {
            str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(target.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and "__pycache__" not in p.parts
        },
    }
    out = RESULTS / "models.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data[label] = record
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{label}: step {step}, revision {revision[:8]}, {len(record['files'])} files")


if __name__ == "__main__":
    main(sys.argv[1])
