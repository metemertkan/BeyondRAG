"""Pure Domain Module: Role-Based Access Control (RBAC) & ACL Tag Management.

Contains zero infrastructure dependencies (no Elasticsearch, LLM, or web framework imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, List, Sequence

PUBLIC_SENTINEL_TAG: Final[str] = "__public__"
BASELINE_ENTERPRISE_TAGS: Final[tuple[str, ...]] = ("finance", "operation")
_PUBLIC_SYNONYMS: Final[frozenset[str]] = frozenset({"public", "__public__", "none", "untagged"})


@dataclass(frozen=True)
class DocumentACL:
    """Resolved access control state for a document."""

    is_public: bool
    visible_tags: List[str]
    acl_tags: List[str]


def normalize_tags(raw_tags: Sequence[str] | str | None) -> List[str]:
    """Normalize a list or comma/semicolon-separated string of RBAC tags.

    Returns an empty list if no restricted tags are present (meaning the document is PUBLIC).
    """
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
        sanitized = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
        if sanitized and sanitized not in cleaned:
            cleaned.append(sanitized)
    return cleaned


def resolve_document_acl(raw_tags: Sequence[str] | str | None) -> DocumentACL:
    """Resolve raw user-supplied tags into a canonical DocumentACL value object."""
    normalized = normalize_tags(raw_tags)
    is_public = len(normalized) == 0
    acl_tags = [PUBLIC_SENTINEL_TAG] if is_public else normalized
    return DocumentACL(
        is_public=is_public,
        visible_tags=[] if is_public else normalized,
        acl_tags=acl_tags,
    )


def build_effective_acl_tags(user_roles: Sequence[str] | str | None) -> tuple[List[str], List[str]]:
    """Return (cleaned_user_roles, effective_acl_tags_with_public_sentinel)."""
    cleaned_roles = normalize_tags(user_roles)
    effective_acl_tags = [PUBLIC_SENTINEL_TAG] + cleaned_roles
    return cleaned_roles, effective_acl_tags


def extract_visible_tags(acl_tags: Iterable[str]) -> List[str]:
    """Strip the internal public sentinel tag from stored ACL tags."""
    return [t for t in acl_tags if t != PUBLIC_SENTINEL_TAG]


def collect_available_tags(documents: Iterable[dict]) -> List[str]:
    """Return sorted unique RBAC tags across all indexed documents plus enterprise defaults."""
    tags = set(BASELINE_ENTERPRISE_TAGS)
    for doc in documents:
        for tag in doc.get("tags", []):
            if tag and tag != PUBLIC_SENTINEL_TAG:
                tags.add(str(tag))
    return sorted(tags)
