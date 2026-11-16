"""Batched sampling with a key/value cache, reproducible per sample.

Every sample has its own random generator, seeded with `base_seed + index`. Sample 7 is the
same text whether it is drawn alone or in a batch of 16 (up to floating-point ties).

Each sample also carries the log-probability of its tokens under the model at temperature 1
(`logprob`, `token_logprobs`). The unit test scores the sampled tokens with `harness.score`
and checks that the two code paths, cached and uncached, agree.

bos is the document separator, so a sampled bos ends that sample. It is not part of the text
and not part of `logprob`.
"""

from __future__ import annotations

import torch


class KVCache:
    """The interface the model's attention expects (same layout as nanochat's engine.KVCache)."""

    def __init__(self, batch_size, seq_len, config, dtype=torch.float32):
        head_dim = config.n_embd // config.n_head
        self.n_layers = config.n_layer
        shape = (config.n_layer, batch_size, seq_len, config.n_kv_head, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype)
        self.v_cache = torch.zeros(shape, dtype=dtype)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)

    def get_pos(self):
        return int(self.cache_seqlens[0])

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, n):
        self.cache_seqlens += n

    def prefill(self, other):
        pos = other.get_pos()
        self.k_cache[:, :, :pos] = other.k_cache[:, :, :pos]
        self.v_cache[:, :, :pos] = other.v_cache[:, :, :pos]
        self.cache_seqlens.fill_(pos)


@torch.inference_mode()
def sample_tokens(model, prompt_ids, n, max_tokens, temperature=1.0, top_k=None, base_seed=0, batch_size=16, stop_id=None, progress=None):
    """Draw n continuations of prompt_ids. Returns one dict per sample: tokens, logprob, token_logprobs, seed, finished."""
    cfg = model.config
    P = len(prompt_ids)
    assert P >= 1 and P + max_tokens <= cfg.sequence_len, "prompt plus max_tokens exceeds the context"
    # The prompt is run once, at batch 1, and its keys and values are copied into every batch.
    pre = KVCache(1, P, cfg)
    first_logits = model(torch.tensor([list(prompt_ids)], dtype=torch.long), kv_cache=pre)[:, -1, :]
    results = []
    for start in range(0, n, batch_size):
        B = min(batch_size, n - start)
        cache = KVCache(B, P + max_tokens, cfg)
        cache.prefill(pre)
        gens = [torch.Generator().manual_seed(base_seed + start + b) for b in range(B)]
        toks = [[] for _ in range(B)]
        lps = [[] for _ in range(B)]
        done = [False] * B
        logits = first_logits.expand(B, -1)
        for step in range(max_tokens):
            logp = torch.log_softmax(logits, dim=-1)
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1)
            else:
                scaled = logits / temperature
                if top_k:
                    kth = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                probs = torch.softmax(scaled, dim=-1)
                nxt = torch.stack([torch.multinomial(probs[b], 1, generator=gens[b])[0] for b in range(B)])
            for b in range(B):
                if done[b]:
                    continue
                t = int(nxt[b])
                if stop_id is not None and t == stop_id:
                    done[b] = True
                    continue
                toks[b].append(t)
                lps[b].append(float(logp[b, t]))
            if progress is not None:
                progress(B - sum(done))
            if all(done) or step == max_tokens - 1:
                break
            logits = model(nxt.view(B, 1), kv_cache=cache)[:, -1, :]
        for b in range(B):
            results.append({"seed": base_seed + start + b, "tokens": toks[b], "logprob": float(sum(lps[b])), "token_logprobs": lps[b], "finished": done[b]})
    return results


def sample(model, tok, prompt, n, max_tokens, temperature=1.0, top_k=None, base_seed=0, batch_size=16, progress=None):
    """Text in, text out. Adds `text` (the continuation only) to each result."""
    bos = tok.get_bos_token_id()
    ids = tok.encode(prompt, prepend=bos)
    out = sample_tokens(model, ids, n, max_tokens, temperature, top_k, base_seed, batch_size, stop_id=bos, progress=progress)
    for r in out:
        r["text"] = tok.decode(r["tokens"])
    return out
