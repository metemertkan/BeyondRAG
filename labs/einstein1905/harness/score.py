"""Log-probability of a continuation given a stem.

`score_tokens` works on token ids, `score` on text. Both return one dict per pair:
    logprob         summed natural-log probability of the continuation tokens
    n_tokens        number of continuation tokens
    token_logprobs  one value per continuation token (reused for the surprisal measurements)

Batches are padded on the right. Attention is causal, so padding cannot change any earlier
position, and padded positions are never read.

Text convention: the stem and the continuation are tokenised separately and concatenated, so
the split between them is exact. A continuation that follows a word starts with a space.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


@torch.inference_mode()
def score_tokens(model, pairs: Sequence[tuple[Sequence[int], Sequence[int]]], token_budget: int = 2048, pad_id: int = 0, progress=None):
    """pairs: (stem_ids, continuation_ids). The stem must not be empty (it starts with bos)."""
    max_len = model.config.sequence_len
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    out: list[dict | None] = [None] * len(pairs)
    i = 0
    while i < len(order):
        longest = len(pairs[order[i]][0]) + len(pairs[order[i]][1])
        batch = [order[i]]
        j = i + 1
        while j < len(order):  # sorted by length, so the last one added is the longest
            nxt = len(pairs[order[j]][0]) + len(pairs[order[j]][1])
            if nxt * (len(batch) + 1) > token_budget:
                break
            batch.append(order[j])
            longest = nxt
            j += 1
        assert longest <= max_len, f"sequence of {longest} tokens exceeds the context of {max_len}"
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        for row, k in enumerate(batch):
            stem, cont = pairs[k]
            assert len(stem) >= 1 and len(cont) >= 1, "stem and continuation must be non-empty"
            seq = list(stem) + list(cont)
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        logp = torch.log_softmax(model(ids), dim=-1)  # (B, T, V); position t predicts token t+1
        for row, k in enumerate(batch):
            stem, cont = pairs[k]
            pos = torch.arange(len(stem) - 1, len(stem) + len(cont) - 1)
            lp = logp[row, pos, torch.tensor(list(cont))].double()
            out[k] = {"logprob": float(lp.sum()), "n_tokens": len(cont), "token_logprobs": [float(x) for x in lp]}
        if progress is not None:
            progress(len(batch), int(ids.numel()))
        i = j
    return out


def encode_pair(tok, stem: str, continuation: str):
    return tok.encode(stem, prepend=tok.get_bos_token_id()), tok.encode(continuation)


def score(model, tok, pairs: Iterable[tuple[str, str]], token_budget: int = 2048, progress=None):
    """pairs: (stem text, continuation text)."""
    return score_tokens(model, [encode_pair(tok, s, c) for s, c in pairs], token_budget=token_budget, pad_id=tok.get_bos_token_id(), progress=progress)


def crossed(model, tok, items: Sequence[dict], token_budget: int = 2048, progress=None):
    """Crossed-pair test. Each item has stem_a, comp_a, stem_b, comp_b.

    Correct when lp(a|a) + lp(b|b) > lp(b|a) + lp(a|b). The two continuations appear on both
    sides of the comparison, so their length and frequency cancel.
    """
    pairs = []
    for it in items:
        pairs += [(it["stem_a"], it["comp_a"]), (it["stem_b"], it["comp_b"]), (it["stem_a"], it["comp_b"]), (it["stem_b"], it["comp_a"])]
    res = score(model, tok, pairs, token_budget=token_budget, progress=progress)
    out = []
    for n in range(len(items)):
        aa, bb, ab, ba = (res[4 * n + k]["logprob"] for k in range(4))
        out.append({"aa": aa, "bb": bb, "ab": ab, "ba": ba, "margin": aa + bb - ab - ba, "correct": aa + bb > ab + ba})
    return out
