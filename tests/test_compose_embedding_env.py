"""Compose forwards RAG_EMBEDDING_* into the app service (static file check)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"

REQUIRED_KEYS = (
    "RAG_EMBEDDING_PROVIDER",
    "RAG_EMBEDDING_DIMS",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_API_KEY",
    "RAG_EMBEDDING_API_BASE",
)


def test_docker_compose_forwards_rag_embedding_env_vars():
    text = COMPOSE.read_text(encoding="utf-8")
    app_block_start = text.index("beyondrag-app:")
    # Truncate at next top-level service (elasticsearch) to scope assertions.
    es_idx = text.find("\n  elasticsearch:", app_block_start)
    if es_idx < 0:
        es_idx = text.find("\nelasticsearch:", app_block_start)
    assert es_idx > app_block_start, "could not locate elasticsearch service after beyondrag-app"
    app_block = text[app_block_start:es_idx]
    for key in REQUIRED_KEYS:
        assert f"{key}=" in app_block or f"{key}:" in app_block, (
            f"docker-compose.yml beyondrag-app must forward {key}"
        )
        assert f"${{{key}" in app_block, (
            f"docker-compose.yml should interpolate host/env .env via ${{{key}:-...}}"
        )


def test_env_example_documents_rag_embedding_vars():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in REQUIRED_KEYS:
        assert key in text
    assert "remote_api" in text
    assert "feature_hash" in text


def test_readme_documents_compose_embedding_forwarding():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "rag_embedding_provider" in readme
    assert "remote_api" in readme
    # Episode 2 / compose forwarding called out
    assert "docker compose" in readme or "docker-compose" in readme
