"""Pure Domain Module: Role-Based Access Control (RBAC) & Configurable Authorization Policy.

Supports configurable baseline roles, custom public sentinel tags, role hierarchies
(e.g., admin -> [finance, operation]), and any/all access evaluation modes without
infrastructure dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Sequence, Set

DEFAULT_PUBLIC_SENTINEL: Final[str] = os.getenv("RAG_PUBLIC_SENTINEL", "__public__").strip()
DEFAULT_BASELINE_ROLES: Final[tuple[str, ...]] = tuple(
    r.strip().lower()
    for r in os.getenv("RAG_BASELINE_ROLES", "finance,operation").split(",")
    if r.strip()
)
_PUBLIC_SYNONYMS: Final[frozenset[str]] = frozenset({"public", "__public__", "none", "untagged"})


@dataclass(frozen=True)
class DocumentACL:
    """Resolved access control state for a document."""

    is_public: bool
    visible_tags: List[str]
    acl_tags: List[str]


@dataclass
class AuthorizationPolicy:
    """Configurable enterprise authorization policy."""

    public_sentinel: str = DEFAULT_PUBLIC_SENTINEL
    baseline_roles: Sequence[str] = DEFAULT_BASELINE_ROLES
    role_hierarchy: Dict[str, Sequence[str]] = field(default_factory=lambda: {
        "admin": ("finance", "operation"),
        "super_admin": ("admin", "finance", "operation"),
    })
    match_mode: str = "any"  # "any" or "all"

    def normalize_tag(self, tag: str) -> str:
        clean = "".join(c for c in tag.strip().lower() if c.isalnum() or c in ("-", "_"))
        return clean

    def normalize_tags(self, raw_tags: Sequence[str] | str | None) -> List[str]:
        if raw_tags is None:
            return []
        if isinstance(raw_tags, str):
            parts = [p.strip().lower() for p in raw_tags.replace(";", ",").split(",")]
        else:
            parts = [str(p).strip().lower() for p in raw_tags]

        cleaned: list[str] = []
        for tag in parts:
            if not tag or tag in _PUBLIC_SYNONYMS:
                continue
            sanitized = self.normalize_tag(tag)
            if sanitized and sanitized not in cleaned:
                cleaned.append(sanitized)
        return cleaned

    def expand_user_roles(self, user_roles: Sequence[str] | str | None) -> List[str]:
        """Expand user roles according to role hierarchy (e.g. admin -> finance, operation)."""
        normalized_inputs = self.normalize_tags(user_roles)
        expanded: Set[str] = set()

        def _traverse(role: str) -> None:
            if role in expanded:
                return
            expanded.add(role)
            for child in self.role_hierarchy.get(role, ()):
                _traverse(child)

        for r in normalized_inputs:
            _traverse(r)
        return sorted(expanded)

    def resolve_document_acl(self, raw_tags: Sequence[str] | str | None) -> DocumentACL:
        normalized = self.normalize_tags(raw_tags)
        is_public = len(normalized) == 0
        acl_tags = [self.public_sentinel] if is_public else normalized
        return DocumentACL(
            is_public=is_public,
            visible_tags=[] if is_public else normalized,
            acl_tags=acl_tags,
        )

    def build_effective_acl_tags(
        self,
        user_roles: Sequence[str] | str | None,
    ) -> tuple[List[str], List[str]]:
        expanded_roles = self.expand_user_roles(user_roles)
        effective_acl_tags = [self.public_sentinel] + expanded_roles
        return expanded_roles, effective_acl_tags

    def extract_visible_tags(self, acl_tags: Iterable[str]) -> List[str]:
        return [t for t in acl_tags if t != self.public_sentinel]

    def collect_available_tags(self, documents: Iterable[dict]) -> List[str]:
        tags = set(self.baseline_roles)
        for doc in documents:
            for tag in doc.get("tags", []):
                if tag and tag != self.public_sentinel:
                    tags.add(str(tag))
        return sorted(tags)


DEFAULT_AUTH_POLICY: Final[AuthorizationPolicy] = AuthorizationPolicy()

# Backward-compatible function exports
PUBLIC_SENTINEL_TAG: Final[str] = DEFAULT_AUTH_POLICY.public_sentinel
BASELINE_ENTERPRISE_TAGS: Final[tuple[str, ...]] = tuple(DEFAULT_AUTH_POLICY.baseline_roles)


def normalize_tags(raw_tags: Sequence[str] | str | None) -> List[str]:
    return DEFAULT_AUTH_POLICY.normalize_tags(raw_tags)


def resolve_document_acl(raw_tags: Sequence[str] | str | None) -> DocumentACL:
    return DEFAULT_AUTH_POLICY.resolve_document_acl(raw_tags)


def build_effective_acl_tags(user_roles: Sequence[str] | str | None) -> tuple[List[str], List[str]]:
    return DEFAULT_AUTH_POLICY.build_effective_acl_tags(user_roles)


def extract_visible_tags(acl_tags: Iterable[str]) -> List[str]:
    return DEFAULT_AUTH_POLICY.extract_visible_tags(acl_tags)


def collect_available_tags(documents: Iterable[dict]) -> List[str]:
    return DEFAULT_AUTH_POLICY.collect_available_tags(documents)
