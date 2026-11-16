"""Gate G0: how fast is the 3.29B-parameter model on this CPU?

Measures scoring (tokens per second through a full forward pass) and sampling (new tokens per
second with the key/value cache), with 12 and 16 threads, and checks on the real model that
the cached and uncached code paths give the same log-probabilities.

    nice -n 5 uv run python -m harness.bench [label]
"""

from __future__ import annotations

import json
import platform
import sys
import time

import torch

from harness.load_model import load
from harness.paths import RESULTS
from harness.sample import sample_tokens
from harness.score import score_tokens

TEXT = (
    "The experiments of Michelson and Morley were made to detect the motion of the earth through the luminiferous ether. "
    "A beam of light was divided into two parts, which travelled at right angles to each other and were then reunited, "
    "so that any difference in the time of their journeys would displace the interference fringes. No displacement of "
    "the expected amount was observed, and the result has been the subject of much discussion among physicists. "
)


def main(label="g1905"):
    model, tok, _ = load(label)
    bos = tok.get_bos_token_id()
    body = tok.encode(TEXT * 4)
    out = {"label": label, "cpu": platform.processor() or "AMD Ryzen 9 3900X", "torch": torch.__version__, "scoring": [], "sampling": []}

    for threads in (12, 16):
        torch.set_num_threads(threads)
        for batch, length in ((8, 64), (32, 64), (8, 256)):
            pairs = [([bos] + body[i : i + length // 2 - 1], body[i + length // 2 - 1 : i + length - 1]) for i in range(batch)]
            t0 = time.time()
            score_tokens(model, pairs, token_budget=batch * length)
            dt = time.time() - t0
            row = {"threads": threads, "batch": batch, "length": length, "seconds": round(dt, 2), "tokens_per_s": round(batch * length / dt, 1)}
            out["scoring"].append(row)
            print("score ", row, flush=True)

    prompt = [bos] + tok.encode("The velocity of light, as measured by an observer in uniform motion,")
    check = None
    for threads in (12, 16):
        torch.set_num_threads(threads)
        for batch in (1, 16):
            new = 24
            t0 = time.time()
            samples = sample_tokens(model, prompt, n=batch, max_tokens=new, temperature=1.0, base_seed=1905, batch_size=batch, stop_id=None)
            dt = time.time() - t0
            row = {"threads": threads, "batch": batch, "new_tokens": new, "seconds": round(dt, 2), "tokens_per_s": round(batch * new / dt, 1), "per_stream": round(new / dt, 2)}
            out["sampling"].append(row)
            print("sample", row, flush=True)
            if batch == 16 and check is None:
                check = samples
    scored = score_tokens(model, [(prompt, s["tokens"]) for s in check[:4]])
    diffs = [abs(a["logprob"] - b["logprob"]) for a, b in zip(scored, check[:4])]
    out["cached_vs_uncached_max_abs_diff"] = max(diffs)
    out["example"] = tok.decode(prompt[1:]) + " ‖" + tok.decode(check[0]["tokens"])
    print("cached vs uncached log-probability, max abs diff:", max(diffs))
    print(out["example"])
    assert max(diffs) < 1e-3
    best = max(r["tokens_per_s"] for r in out["scoring"])
    out["gate_g0"] = {"threshold_tokens_per_s": 20, "best_scoring_tokens_per_s": best, "passed": best >= 20}
    (RESULTS / "g0_speed.json").write_text(json.dumps(out, indent=2) + "\n")
    print("G0:", out["gate_g0"])


if __name__ == "__main__":
    main(*sys.argv[1:])
