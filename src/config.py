"""Dual-LLM Router and runtime environment configuration for hk-fde-content-engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import httpx
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
WORKSPACE_DIR: Final[Path] = PROJECT_ROOT / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY: Final[str] = os.getenv("GOOGLE_API_KEY", "").strip()
USE_LOCAL_ONLY: Final[bool] = os.getenv("USE_LOCAL_ONLY", "false").strip().lower() == "true"
LANGGRAPH_STRICT_MSGPACK: Final[bool] = (
    os.getenv("LANGGRAPH_STRICT_MSGPACK", "true").strip().lower() == "true"
)

POSTGRES_URI: Final[str] = os.getenv(
    "POSTGRES_URI",
    "postgresql://fde_admin:fde_password@localhost:5432/langgraph_state",
).strip()
ELASTICSEARCH_URL: Final[str] = os.getenv(
    "ELASTICSEARCH_URL",
    "http://localhost:9200",
).strip()
OLLAMA_BASE_URL: Final[str] = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip()

DEVTO_API_KEY: Final[str] = os.getenv("DEVTO_API_KEY", "").strip()
HASHNODE_TOKEN: Final[str] = os.getenv("HASHNODE_TOKEN", "").strip()
HASHNODE_PUBLICATION_ID: Final[str] = os.getenv("HASHNODE_PUBLICATION_ID", "").strip()

PRO_ROLES: Final[set[str]] = {"architect", "coder", "blog"}
FLASH_ROLES: Final[set[str]] = {"manager", "shorts"}
PLACEHOLDER_KEYS: Final[set[str]] = {
    "",
    "your_google_gemini_api_key_here",
    "changeme",
    "placeholder",
}


def is_valid_api_key(key_value: str) -> bool:
    """Return True if the provided API key is non-empty and not a template placeholder."""
    normalized = key_value.strip()
    if not normalized:
        return False
    if normalized.lower() in PLACEHOLDER_KEYS:
        return False
    if normalized.lower().startswith("your_"):
        return False
    return True


class ResilientChatRouter(RunnableWithFallbacks):
    """RunnableWithFallbacks subclass that propagates .with_structured_output() across primary and fallback LLMs."""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        primary_structured = self.runnable.with_structured_output(schema, **kwargs)
        fallback_structured = [
            fb.with_structured_output(schema, **kwargs) for fb in self.fallbacks
        ]
        return primary_structured.with_fallbacks(fallback_structured)


def _detect_host_ollama_models(base_url: str) -> list[str]:
    """Query the local Ollama daemon (/api/tags) and return available model names."""
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            payload = resp.json()
            return [
                str(item.get("name", "")).strip()
                for item in payload.get("models", [])
                if item.get("name")
            ]
    except Exception:
        return []


def get_llm(role: str = "architect", temperature: float = 0.2) -> BaseChatModel:
    """Instantiate a resilient Dual-LLM router for the specified agent role.

    Primary models via ChatGoogleGenerativeAI:
      - gemini-2.5-pro for roles in ["architect", "coder", "blog"]
      - gemini-2.5-flash for roles in ["manager", "shorts"]

    Fallback models via ChatOllama:
      - qwen2.5-coder:14b for role == "coder"
      - llama3.2:latest for all other roles
      - Also binds any already-pulled host Ollama model (e.g., qwen3:8b) as a secondary
        local fallback if the target tag is not yet pulled on the host Ollama daemon.

    If USE_LOCAL_ONLY="true" or GOOGLE_API_KEY is missing/placeholder, routes
    directly to the local Ollama router. Otherwise returns gemini_llm.with_fallbacks([ollama_llm]).
    """
    normalized_role = role.strip().lower()

    if normalized_role in PRO_ROLES:
        gemini_model_name = "gemini-2.5-pro"
    elif normalized_role in FLASH_ROLES:
        gemini_model_name = "gemini-2.5-flash"
    else:
        gemini_model_name = "gemini-2.5-pro"

    env_ollama_model = os.getenv("OLLAMA_MODEL")
    installed_models = _detect_host_ollama_models(OLLAMA_BASE_URL)

    # Determine default Ollama model: env override > installed lightweight models > default tag
    if env_ollama_model:
        ollama_model_name = env_ollama_model
    elif installed_models:
        selected = None
        for pref in ("qwen2.5:1.5b", "qwen2.5:0.5b", "llama3.2:1b", "qwen3:8b"):
            if pref in installed_models:
                selected = pref
                break
        ollama_model_name = selected or installed_models[0]
    else:
        ollama_model_name = (
            "qwen2.5-coder:14b" if normalized_role == "coder" else "llama3.2:latest"
        )

    ollama_llm = ChatOllama(
        model=ollama_model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        reasoning=False,
    )

    fallback_chain: list[BaseChatModel] = [ollama_llm]
    if installed_models and ollama_model_name not in installed_models:
        fallback_chain.insert(
            0,
            ChatOllama(
                model=installed_models[0],
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                reasoning=False,
            ),
        )

    if USE_LOCAL_ONLY or not is_valid_api_key(GOOGLE_API_KEY):
        if len(fallback_chain) > 1:
            return ResilientChatRouter(
                runnable=fallback_chain[0],
                fallbacks=fallback_chain[1:],
            )  # type: ignore[return-value]
        return ollama_llm

    gemini_llm = ChatGoogleGenerativeAI(
        model=gemini_model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
    bound = gemini_llm.with_fallbacks(fallback_chain)
    return ResilientChatRouter(
        runnable=bound.runnable,
        fallbacks=bound.fallbacks,
    )  # type: ignore[return-value]
