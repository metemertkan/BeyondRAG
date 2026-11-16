"""Load a nanochat checkpoint on the CPU in 32-bit floats.

The model card's quick start assumes CUDA and bfloat16 autocast. Three things differ here:

* `init_weights()` is skipped. It would fill 13 GB with random numbers that the checkpoint
  then replaces. Only the rotary tables it builds are needed, so those are built directly.
* The checkpoint stores the token and value embeddings in bfloat16. They are cast to float32,
  because bfloat16 arithmetic is slow on a CPU. The values are unchanged by the cast.
* The rotary tables stay in bfloat16, as in training (the forward pass asserts it). They are
  promoted to float32 when multiplied with the queries and keys.

The loading code (`nanochat/`) is the copy shipped in the model repo. It is imported from the
download folder and is not modified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from harness.paths import HISTORICAL, MODELS

CODE_FROM = "g1905"  # all three historical models ship the same nanochat code (checked by hash)


def _nanochat():
    code_dir = str(MODELS / CODE_FROM)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer

    return GPT, GPTConfig, RustBPETokenizer


def set_threads(n: int = 12) -> None:
    """The server has 12 physical cores. More threads than cores does not help matrix products."""
    torch.set_num_threads(n)


def _finish(model):
    """Float32 weights, bfloat16 rotary tables, eval mode."""
    model.float()
    head_dim = model.config.n_embd // model.config.n_head
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=torch.device("cpu"))
    model.cos, model.sin = cos, sin
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_tokenizer(label: str = CODE_FROM):
    _, _, RustBPETokenizer = _nanochat()
    return RustBPETokenizer.from_directory(str(MODELS / label / "tokenizer"))


def load(label: str, verbose: bool = True):
    """Return (model, tokenizer, meta) for one of the historical models."""
    GPT, GPTConfig, _ = _nanochat()
    model_dir = MODELS / label
    metas = sorted(model_dir.glob("meta_*.json"))
    assert metas, f"no checkpoint in {model_dir}; run `python -m harness.download {label}`"
    meta_path = metas[-1]
    step = meta_path.stem.split("_")[1]
    meta = json.loads(meta_path.read_text())
    t0 = time.time()
    config = GPTConfig(**meta["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    state = torch.load(model_dir / f"model_{step}.pt", map_location="cpu", mmap=True, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    _finish(model)
    if verbose:
        counts = model.num_scaling_params()
        print(f"{label}: step {int(step)}, loaded in {time.time() - t0:.1f}s, " + ", ".join(f"{k}={v / 1e6:.1f}M" for k, v in counts.items()))
    return model, load_tokenizer(label), meta


def tiny(seed: int = 0, n_layer: int = 2, n_embd: int = 64, n_head: int = 2, sequence_len: int = 256):
    """A small random model with the same code path, for tests. Uses the real tokenizer."""
    GPT, GPTConfig, _ = _nanochat()
    config = GPTConfig(sequence_len=sequence_len, vocab_size=32768, n_layer=n_layer, n_head=n_head, n_kv_head=n_head, n_embd=n_embd, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:  # init leaves the output projections at zero; give every matrix some signal
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 if "lm_head" in name else 0.1))
    return _finish(model), load_tokenizer()


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "g1905"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The luminiferous aether"
    set_threads()
    model, tok, _ = load(label)
    ids = tok.encode(prompt, prepend=tok.get_bos_token_id())
    print(prompt, end="", flush=True)
    t0, n = time.time(), 0
    for t in model.generate(ids, max_tokens=100, temperature=0.8, seed=1905):
        print(tok.decode([t]), end="", flush=True)
        n += 1
    print(f"\n[{n} tokens in {time.time() - t0:.1f}s, no cache]")
