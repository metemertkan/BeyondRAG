"""Tenant scope: who is asking, and the one tenant their request may touch.

Roles say what a user may read inside a company. They say nothing about which company.
Two tenants both have an ``employee`` role, so an ACL filter alone lets one tenant's
employees read the other's documents. The tenant is a second, mandatory dimension.

Two rules hold everywhere in the data path:

  - A read names exactly one tenant. There is no default, no wildcard and no way to ask
    for "all tenants" (``require_tenant``).
  - A write without a tenant lands in the ``default`` tenant, as it always has
    (``normalize_tenant``). It can never spread across tenants.

The tenant comes from a verified token (see ``auth.py``), never from a request body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Tuple

DEFAULT_TENANT: Final[str] = "default"
MAX_TENANT_LENGTH: Final[int] = 32


class TenantScopeError(ValueError):
    """Raised when a read has no tenant, or names something that is not one tenant."""


class TenantNotFoundError(LookupError):
    """The object does not exist for this tenant.

    Raised both when it does not exist at all and when it belongs to another tenant, so
    a caller cannot tell the two apart and probe for other tenants' document ids.
    """


def normalize_tenant(tenant: str | None) -> str:
    """Clean a tenant id for writes. Empty means the ``default`` tenant."""
    raw = (tenant or DEFAULT_TENANT).strip().lower() or DEFAULT_TENANT
    cleaned = "".join(c for c in raw if c.isalnum() or c in ("-", "_"))[:MAX_TENANT_LENGTH]
    return cleaned or DEFAULT_TENANT


def require_tenant(tenant: Any) -> str:
    """Validate a tenant id for reads. Refuses anything that cleaning would change.

    ``"*"``, ``"acme,globex"`` and ``"acme*"`` are rejected instead of cleaned, because a
    cleaned wildcard silently becomes a different tenant, and an uncleaned one becomes an
    Elasticsearch index pattern.
    """
    if not isinstance(tenant, str) or not tenant.strip():
        raise TenantScopeError("A tenant is required. Searches never run across tenants.")
    candidate = tenant.strip().lower()
    if candidate != normalize_tenant(candidate) or not candidate[0].isalnum():
        raise TenantScopeError(f"Not a valid tenant id: {tenant!r}")
    return candidate


def tenant_scoped(kind: str, tenant: str | None, value: str) -> str:
    """Hash input for tombstones, unique per tenant.

    The default tenant keeps the earlier ``kind:value`` form, so tombstones written before
    tenants existed still block re-ingestion. Other tenants use a JSON list, which cannot
    collide with that form or with another tenant whatever characters the value contains.
    """
    scope = normalize_tenant(tenant)
    if scope == DEFAULT_TENANT:
        return f"{kind}:{value}"
    return json.dumps([kind, scope, value], ensure_ascii=False)


def tenant_digest(tenant: str | None, length: int = 10) -> str:
    return hashlib.sha256(normalize_tenant(tenant).encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Built from a verified token and nothing else."""

    tenant: str
    roles: Tuple[str, ...] = ()
    subject: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant", require_tenant(self.tenant))
        object.__setattr__(self, "roles", tuple(str(r).strip().lower() for r in self.roles if str(r).strip()))
