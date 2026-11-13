"""Scoring and sampling must agree with each other and must not depend on batching.

These run on a small random model that uses the same code as the real one (rotary embeddings,
value embeddings, the key/value cache), so they finish in seconds.
"""

import pytest
import torch

from harness.load_model import tiny
from harness.sample import sample_tokens
from harness.score import crossed, score, score_tokens


@pytest.fixture(scope="module")
def mt():
    torch.set_num_threads(4)
    return tiny(seed=0)


def _prompt(tok):
    return tok.encode("The velocity of light in a vacuum is", prepend=tok.get_bos_token_id())


def test_scoring_reproduces_sampler_logprob(mt):
    model, tok = mt
    prompt = _prompt(tok)
    samples = sample_tokens(model, prompt, n=6, max_tokens=24, temperature=1.0, base_seed=11, batch_size=3)
    assert all(len(s["tokens"]) == 24 for s in samples)
    assert len({tuple(s["tokens"]) for s in samples}) > 1, "the random model should not be deterministic"
    scored = score_tokens(model, [(prompt, s["tokens"]) for s in samples])
    for s, r in zip(samples, scored):
        assert r["n_tokens"] == 24
        assert abs(r["logprob"] - s["logprob"]) < 1e-3
        assert max(abs(a - b) for a, b in zip(r["token_logprobs"], s["token_logprobs"])) < 1e-3


def test_sample_is_the_same_in_any_batch(mt):
    model, tok = mt
    prompt = _prompt(tok)
    alone = sample_tokens(model, prompt, n=5, max_tokens=16, base_seed=3, batch_size=1)
    together = sample_tokens(model, prompt, n=5, max_tokens=16, base_seed=3, batch_size=5)
    assert [s["tokens"] for s in alone] == [s["tokens"] for s in together]
    shifted = sample_tokens(model, prompt, n=1, max_tokens=16, base_seed=5, batch_size=1)
    assert shifted[0]["tokens"] == alone[2]["tokens"], "seed base+index identifies a sample"


def test_greedy_and_top_k(mt):
    model, tok = mt
    prompt = _prompt(tok)
    a = sample_tokens(model, prompt, n=2, max_tokens=8, temperature=0.0)
    assert a[0]["tokens"] == a[1]["tokens"]
    k1 = sample_tokens(model, prompt, n=1, max_tokens=8, temperature=1.0, top_k=1)
    assert k1[0]["tokens"] == a[0]["tokens"], "top_k=1 is greedy"


def test_padding_does_not_change_scores(mt):
    model, tok = mt
    pairs = [
        ("Light is propagated through", " the luminiferous ether"),
        ("A rigid rod at rest in a moving system has, as measured in the system at rest, a length which is", " shorter"),
        ("Heat", " is a mode of motion, as Tyndall wrote, and the mechanical equivalent of heat was measured by Joule"),
    ]
    together = score(model, tok, pairs, token_budget=4096)
    for pair, r in zip(pairs, together):
        alone = score(model, tok, [pair])[0]
        assert abs(alone["logprob"] - r["logprob"]) < 1e-3
        assert alone["n_tokens"] == r["n_tokens"]


def test_stop_token_ends_a_sample(mt):
    model, tok = mt
    prompt = _prompt(tok)
    greedy = sample_tokens(model, prompt, n=1, max_tokens=8, temperature=0.0)[0]["tokens"]
    stopped = sample_tokens(model, prompt, n=1, max_tokens=8, temperature=0.0, stop_id=greedy[3])[0]
    assert stopped["tokens"] == greedy[:3] and stopped["finished"]


def test_crossed_pair_is_symmetric(mt):
    model, tok = mt
    item = {"stem_a": "The planet nearest the sun is", "comp_a": " Mercury", "stem_b": "The largest planet is", "comp_b": " Jupiter"}
    swapped = {"stem_a": item["stem_b"], "comp_a": item["comp_b"], "stem_b": item["stem_a"], "comp_b": item["comp_a"]}
    r1, r2 = crossed(model, tok, [item, swapped])
    assert abs(r1["margin"] - r2["margin"]) < 1e-3
    assert r1["correct"] == r2["correct"]
