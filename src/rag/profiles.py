"""Component 8: Strongly Typed Retrieval Profiles and Parameter Resolution.

Supports configurable multi-tenant or shared Elasticsearch indices, and use-case-specific
retrieval profiles (e.g., precision, high-recall, executive_summary) overriding top_k,
candidate multipliers, minimum_should_match, and BM25 boost parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Final, Optional

DEFAULT_INDEX_PREFIX: Final[str] = os.getenv("RAG_INDEX_PREFIX", "hk-rbac-rag-kb").strip()
DEFAULT_TENANT_MODE: Final[str] = os.getenv("RAG_TENANT_MODE", "shared").strip().lower()  # "shared" or "per_tenant"
DEFAULT_TENANT_ID: Final[str] = os.getenv("RAG_DEFAULT_TENANT", "default").strip()


@dataclass(frozen=True)
class IndexTenantStrategy:
    """Strategy for deriving Elasticsearch index names based on tenant configuration."""

    prefix: str = DEFAULT_INDEX_PREFIX
    tenant_mode: str = DEFAULT_TENANT_MODE
    default_tenant: str = DEFAULT_TENANT_ID

    def resolve_index_name(self, tenant_id: Optional[str] = None) -> str:
        tenant = (tenant_id or self.default_tenant).strip().lower()
        if self.tenant_mode == "per_tenant":
            safe_tenant = "".join(c for c in tenant if c.isalnum() or c in ("-", "_"))
            return f"{self.prefix}-{safe_tenant}"
        return self.prefix


@dataclass(frozen=True)
class RetrievalProfile:
    """Retrieval configuration tailored to a specific use case."""

    name: str
    top_k: int
    num_candidates_multiplier: int
    content_boost: float
    title_boost: float
    min_should_match: str
    audit_masked_size: int
    default_synthesis_mode: str


@dataclass(frozen=True)
class ResolvedRetrievalConfig:
    """Final, strongly typed retrieval parameters resolved exactly once per query."""

    profile_name: str
    top_k: int
    num_candidates_multiplier: int
    content_boost: float
    title_boost: float
    min_should_match: str
    audit_masked_size: int
    synthesis_mode: str

    @property
    def num_candidates(self) -> int:
        return max(20, self.top_k * self.num_candidates_multiplier)


RETRIEVAL_PROFILES: Final[Dict[str, RetrievalProfile]] = {
    "default": RetrievalProfile(
        name="default",
        top_k=4,
        num_candidates_multiplier=4,
        content_boost=2.0,
        title_boost=1.2,
        min_should_match="2<60%",
        audit_masked_size=10,
        default_synthesis_mode="fast",
    ),
    "precision": RetrievalProfile(
        name="precision",
        top_k=2,
        num_candidates_multiplier=6,
        content_boost=2.5,
        title_boost=1.5,
        min_should_match="3<75%",
        audit_masked_size=5,
        default_synthesis_mode="fast",
    ),
    "high_recall": RetrievalProfile(
        name="high_recall",
        top_k=8,
        num_candidates_multiplier=4,
        content_boost=1.8,
        title_boost=1.0,
        min_should_match="1<40%",
        audit_masked_size=20,
        default_synthesis_mode="llm",
    ),
    "executive_summary": RetrievalProfile(
        name="executive_summary",
        top_k=5,
        num_candidates_multiplier=5,
        content_boost=2.2,
        title_boost=1.4,
        min_should_match="2<50%",
        audit_masked_size=10,
        default_synthesis_mode="llm",
    ),
}


def resolve_retrieval_config(
    profile_name: Optional[str] = None,
    top_k_override: Optional[int] = None,
    synthesis_mode_override: Optional[str] = None,
) -> ResolvedRetrievalConfig:
    """Resolve retrieval configuration exactly once with clean precedence rules."""
    name_key = (profile_name or "default").strip().lower()
    if name_key not in RETRIEVAL_PROFILES:
        raise ValueError(
            f"Unknown retrieval profile '{profile_name}'. Valid profiles: {list(RETRIEVAL_PROFILES.keys())}"
        )

    base = RETRIEVAL_PROFILES[name_key]
    effective_top_k = top_k_override if (top_k_override is not None and top_k_override > 0) else base.top_k
    effective_synth = synthesis_mode_override if synthesis_mode_override else base.default_synthesis_mode

    return ResolvedRetrievalConfig(
        profile_name=base.name,
        top_k=effective_top_k,
        num_candidates_multiplier=base.num_candidates_multiplier,
        content_boost=base.content_boost,
        title_boost=base.title_boost,
        min_should_match=base.min_should_match,
        audit_masked_size=base.audit_masked_size,
        synthesis_mode=effective_synth,
    )


# Backward-compatible helper
def get_retrieval_profile(
    name: Optional[str] = None,
    top_k_override: Optional[int] = None,
) -> RetrievalProfile:
    resolved = resolve_retrieval_config(profile_name=name, top_k_override=top_k_override)
    return RetrievalProfile(
        name=resolved.profile_name,
        top_k=resolved.top_k,
        num_candidates_multiplier=resolved.num_candidates_multiplier,
        content_boost=resolved.content_boost,
        title_boost=resolved.title_boost,
        min_should_match=resolved.min_should_match,
        audit_masked_size=resolved.audit_masked_size,
        default_synthesis_mode=resolved.synthesis_mode,
    )
