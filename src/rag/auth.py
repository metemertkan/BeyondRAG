"""Signed identity: the tenant and roles come from a token, not from the request body.

Before this module the dashboard sent ``user_roles`` in the chat request and ``tenant`` in
the upload form, so "authorization" trusted whatever the client typed. Now both are claims
in an HS256 JSON Web Token that the server verifies on every request.

What is checked, and why:

  - The algorithm is pinned to HS256. A token with ``alg: none`` is refused.
  - ``exp``, ``sub`` and ``tenant`` are required. A token without a tenant is not a token
    for "all tenants"; it is invalid.
  - ``aud`` must be this service, so a token minted for another system is refused.

Modes (``RAG_AUTH_MODE``):

  - ``demo`` (default): the app can mint tokens for the dashboard's persona picker, and
    generates a signing secret under ``workspace/`` if none is configured. This shows how
    tenancy behaves. It is not authentication, because anyone can ask for any persona.
  - ``required``: no minting endpoint. ``RAG_AUTH_SECRET`` must be set, and tokens come
    from your identity provider or gateway.

A production system would verify RS256 or ES256 tokens against the provider's published
keys instead of sharing a secret. The checks above stay the same.
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Final, Iterable, Optional

import jwt

from src.config import WORKSPACE_DIR
from src.rag.tenancy import Principal, TenantScopeError, require_tenant

ALGORITHM: Final[str] = "HS256"
AUDIENCE: Final[str] = "beyondrag"
AUTH_MODES: Final[tuple[str, ...]] = ("demo", "required")
DEFAULT_AUTH_SECRET_PATH: Final[Path] = WORKSPACE_DIR / ".auth_secret"
DEFAULT_TOKEN_TTL_SECONDS: Final[int] = 3600
MIN_SECRET_BYTES: Final[int] = 32


class AuthNotConfiguredError(RuntimeError):
    """No usable signing secret. The API must answer 503, not fall back to open access."""


class InvalidTokenError(ValueError):
    """The token is missing, malformed, expired, wrongly signed or lacks a tenant."""


def auth_mode() -> str:
    mode = os.getenv("RAG_AUTH_MODE", "demo").strip().lower()
    if mode not in AUTH_MODES:
        raise AuthNotConfiguredError(f"Unknown RAG_AUTH_MODE {mode!r}. Valid: {AUTH_MODES}")
    return mode


def load_auth_secret(secret_path: Path = DEFAULT_AUTH_SECRET_PATH) -> bytes:
    """Return the signing secret, generating one only in demo mode."""
    configured = os.getenv("RAG_AUTH_SECRET", "").strip()
    if configured:
        secret = configured.encode("utf-8")
        if len(secret) < MIN_SECRET_BYTES:
            raise AuthNotConfiguredError(f"RAG_AUTH_SECRET must be at least {MIN_SECRET_BYTES} bytes.")
        return secret
    if auth_mode() != "demo":
        raise AuthNotConfiguredError("RAG_AUTH_SECRET is not set and RAG_AUTH_MODE is 'required'.")
    if secret_path.exists():
        return secret_path.read_bytes()
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(48).encode("utf-8")
    secret_path.touch(mode=0o600)
    secret_path.write_bytes(secret)
    return secret


def mint_token(
    tenant: str,
    roles: Iterable[str] = (),
    *,
    subject: str = "demo-user",
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
    secret: Optional[bytes] = None,
) -> str:
    """Sign a token. Used by the demo endpoint, the tests and the benchmark."""
    principal = Principal(tenant=tenant, roles=tuple(roles), subject=subject)
    now = int(time.time())
    claims: Dict[str, Any] = {
        "sub": principal.subject or "demo-user",
        "tenant": principal.tenant,
        "roles": list(principal.roles),
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + int(ttl_seconds),
    }
    return jwt.encode(claims, secret or load_auth_secret(), algorithm=ALGORITHM)


def verify_token(token: str, *, secret: Optional[bytes] = None) -> Principal:
    """Verify signature and claims, and return who is calling."""
    if not token or not token.strip():
        raise InvalidTokenError("Missing bearer token.")
    try:
        claims = jwt.decode(
            token.strip(),
            secret or load_auth_secret(),
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            options={"require": ["exp", "sub", "tenant"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(f"Invalid token: {exc}") from exc
    roles = claims.get("roles") or []
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise InvalidTokenError("Invalid token: roles must be a list of strings.")
    try:
        return Principal(tenant=require_tenant(claims.get("tenant")), roles=tuple(roles), subject=str(claims["sub"]))
    except TenantScopeError as exc:
        raise InvalidTokenError(f"Invalid token: {exc}") from exc
