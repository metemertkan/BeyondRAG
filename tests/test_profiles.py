"""Unit tests for retrieval profiles, configuration overrides, and precedence."""

import pytest
from src.rag.profiles import (
    IndexTenantStrategy,
    RETRIEVAL_PROFILES,
    resolve_retrieval_config,
)


def test_profile_resolution_precedence():
    # 1. Precision profile uses its configured top_k (2) and params
    p_prec = resolve_retrieval_config("precision")
    assert p_prec.top_k == 2
    assert p_prec.num_candidates_multiplier == 6
    assert p_prec.content_boost == 2.5
    assert p_prec.min_should_match == "3<75%"
    assert p_prec.synthesis_mode == "fast"

    # 2. High recall profile uses top_k (8) and llm synthesis
    p_recall = resolve_retrieval_config("high_recall")
    assert p_recall.top_k == 8
    assert p_recall.synthesis_mode == "llm"

    # 3. Explicit caller top_k override takes precedence over profile default
    p_override = resolve_retrieval_config("precision", top_k_override=10)
    assert p_override.top_k == 10
    assert p_override.content_boost == 2.5  # maintains other precision profile attributes

    # 4. Explicit caller synthesis_mode override
    p_synth = resolve_retrieval_config("default", synthesis_mode_override="llm")
    assert p_synth.synthesis_mode == "llm"


def test_unknown_profile_raises_error():
    with pytest.raises(ValueError, match="Unknown retrieval profile"):
        resolve_retrieval_config("non_existent_profile")


def test_tenant_strategy():
    shared_strat = IndexTenantStrategy(prefix="kb", tenant_mode="shared")
    assert shared_strat.resolve_index_name("tenant_a") == "kb"

    isolated_strat = IndexTenantStrategy(prefix="kb", tenant_mode="per_tenant")
    assert isolated_strat.resolve_index_name("tenant_a") == "kb-tenant_a"
