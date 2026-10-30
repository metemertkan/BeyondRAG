"""Dynamic LLM Cross-Lingual (English <-> Chinese) Dense Vector & Concept Engine for BeyondRAG.

Instead of hardcoding domain translations in source code, BeyondRAG uses a local
Ollama LLM (e.g. `qwen3:8b`) at **PDF Upload / Ingestion Time** to extract domain
concepts and generate bilingual (English + Traditional/Simplified Chinese) trigger
mappings on the fly.

Extracted mappings are persisted in `workspace/dynamic_concepts.json` (keyed by PDF
content hash) and projected into a shared 256-dimensional L2-normalized semantic
vector space (`int8_hnsw` cosine in Elasticsearch 8.15).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import httpx

from src.config import OLLAMA_BASE_URL, WORKSPACE_DIR, _detect_host_ollama_models

EMBEDDING_PROVIDER: Final[str] = os.getenv("RAG_EMBEDDING_PROVIDER", "feature_hash").strip().lower()
EMBEDDING_MODEL_NAME: Final[str] = os.getenv("RAG_EMBEDDING_MODEL", "bge-m3").strip()
EMBEDDING_DIMS: Final[int] = int(os.getenv("RAG_EMBEDDING_DIMS", "256"))
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9_]+|[\u4e00-\uffff]+", re.IGNORECASE)
_CJK_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\uffff]")

DYNAMIC_CONCEPTS_PATH: Final[Path] = WORKSPACE_DIR / "dynamic_concepts.json"

STOPWORDS: Final[set[str]] = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "into",
    "about",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "can",
    "may",
    "also",
    "only",
    "other",
    "than",
    "then",
    "them",
    "they",
    "their",
    "there",
    "these",
    "those",
    "your",
    "our",
}


_GENERIC_CJK_BIGRAMS: Final[set[str]] = {
    "政策",
    "規定",
    "规定",
    "什麼",
    "什么",
    "請問",
    "请问",
    "公司",
    "香港",
    "要求",
    "多少",
    "時間",
    "时间",
    "工作",
    "管理",
    "系統",
    "系统",
    "標準",
    "标准",
    "部門",
    "部门",
}


def has_cjk(text: str) -> bool:
    """Return True if the string contains Chinese/CJK characters."""
    return bool(_CJK_CHAR_RE.search(text))


def _matches_zh_triggers(text: str, zh_triggers: List[str]) -> bool:
    """Return True if text matches a Chinese trigger phrase or shares a domain CJK 2-gram."""
    if not has_cjk(text):
        return False
    for zt in zh_triggers:
        if not zt:
            continue
        if zt in text:
            return True
        # Check domain-specific 2-character Chinese root compounds inside the LLM trigger
        if len(zt) >= 2:
            for i in range(len(zt) - 1):
                bg = zt[i : i + 2]
                if len(_CJK_CHAR_RE.findall(bg)) == 2 and bg not in _GENERIC_CJK_BIGRAMS:
                    if bg in text:
                        return True
    return False


class DynamicConceptRegistry:
    """Manages LLM-generated English <-> Chinese concept mappings extracted at PDF ingestion time."""

    def __init__(self, storage_path: Path = DYNAMIC_CONCEPTS_PATH) -> None:
        self.storage_path = storage_path
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._query_cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._documents = raw.get("documents", {})
                self._query_cache = raw.get("query_cache", {})
        except Exception:
            self._documents = {}
            self._query_cache = {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self._documents,
            "query_cache": self._query_cache,
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all_concepts(self) -> List[Tuple[str, List[str], List[str], str]]:
        """Return all dynamically extracted (concept_id, en_triggers, zh_triggers, en_expansion) tuples."""
        concepts: List[Tuple[str, List[str], List[str], str]] = []
        seen_ids: set[str] = set()
        for doc_entry in self._documents.values():
            for item in doc_entry.get("concepts", []):
                cid = str(item.get("concept_id", "")).strip()
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                en_t = [str(x).lower().strip() for x in item.get("en_triggers", []) if str(x).strip()]
                zh_t = [str(x).strip() for x in item.get("zh_triggers", []) if str(x).strip()]
                en_exp = str(item.get("en_expansion", "")).strip()
                if en_t or zh_t:
                    concepts.append((cid, en_t, zh_t, en_exp))
        return concepts

    def get_document_concepts(self, content_hash: str) -> List[Dict[str, Any]] | None:
        entry = self._documents.get(content_hash)
        if entry and isinstance(entry.get("concepts"), list) and entry["concepts"]:
            return entry["concepts"]
        return None

    def register_document_concepts(
        self,
        content_hash: str,
        filename: str,
        concepts: List[Dict[str, Any]],
        model_used: str,
    ) -> None:
        self._documents[content_hash] = {
            "filename": filename,
            "model_used": model_used,
            "concepts": concepts,
        }
        self._save()

    def get_cached_query_translation(self, query: str) -> str | None:
        return self._query_cache.get(query.strip())

    def cache_query_translation(self, query: str, english_keywords: str) -> None:
        self._query_cache[query.strip()] = english_keywords.strip()
        self._save()


_REGISTRY: DynamicConceptRegistry | None = None


def get_concept_registry() -> DynamicConceptRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DynamicConceptRegistry()
    return _REGISTRY


def _select_ollama_model() -> str:
    """Pick the best available local Ollama model for bilingual concept extraction."""
    available = _detect_host_ollama_models(OLLAMA_BASE_URL)
    for preferred in ("qwen3:8b", "qwen2.5:7b", "llama3.1:8b"):
        if preferred in available:
            return preferred
    return available[0] if available else "qwen3:8b"


def extract_bilingual_concepts_at_ingestion(
    filename: str,
    full_text: str,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Use local Ollama at PDF upload time to generate English <-> Chinese concept mappings dynamically.

    If this exact PDF text (by SHA-256 hash) was already processed by Ollama, returns
    the persisted dynamic mappings from `workspace/dynamic_concepts.json` immediately.
    """
    registry = get_concept_registry()
    content_hash = hashlib.sha256(full_text.strip().encode("utf-8")).hexdigest()[:16]

    if not force_refresh:
        cached = registry.get_document_concepts(content_hash)
        if cached:
            model_name = registry._documents.get(content_hash, {}).get("model_used", "ollama-cached")
            return cached, model_name

    model_name = _select_ollama_model()
    prompt_text = (
        f"Document: {filename}\n"
        f"Text:\n{full_text[:2500]}\n\n"
        "Extract 3 distinct domain concepts from the text so users can search this English PDF using "
        "Traditional Chinese (Hong Kong), Simplified Chinese, or English.\n"
        "For each concept:\n"
        "- Give a unique `concept_id` (e.g. `concept:cash_reserve`, `concept:fx_hedging`)\n"
        "- `en_triggers`: array of short lowercase English keywords, numbers, and acronyms from the text\n"
        "- `zh_triggers`: array of separate Chinese keywords in Traditional HK Chinese and Simplified Chinese\n"
        "- `en_expansion`: space-separated English keywords from the text\n"
        "Return ONLY JSON:\n"
        '{"concepts": [{"concept_id": "concept:unique_name", "en_triggers": ["kw1", "kw2"], "zh_triggers": ["繁體詞", "简体词"], "en_expansion": "english keywords"}]}'
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Extract bilingual EN<->ZH search concepts as compact JSON.",
            },
            {"role": "user", "content": prompt_text},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 480},
    }

    extracted_concepts: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            raw_list = parsed.get("concepts", [])
            if isinstance(raw_list, list):
                for idx, item in enumerate(raw_list, start=1):
                    if not isinstance(item, dict):
                        continue
                    raw_cid = str(item.get("concept_id") or "").strip().lower()
                    slug = re.sub(r"[^a-z0-9_]+", "_", raw_cid.replace("concept:", "")).strip("_")
                    if not slug or slug in {"topic", "example", "unique_name", "id"}:
                        fallback_src = str(item.get("en_expansion") or f"c{idx}").lower()
                        slug = re.sub(r"[^a-z0-9_]+", "_", fallback_src).strip("_")[:28]
                    cid = f"concept:{content_hash[:6]}_{idx}_{slug}"

                    en_t: List[str] = []
                    for x in item.get("en_triggers", []):
                        for part in re.split(r"[/,;]+", str(x).lower()):
                            clean_p = part.strip()
                            if clean_p and clean_p not in en_t:
                                en_t.append(clean_p)

                    zh_t: List[str] = []
                    for x in item.get("zh_triggers", []):
                        for part in re.split(r"[/,、;（）() ]+", str(x)):
                            clean_z = part.strip()
                            if clean_z and clean_z not in zh_t:
                                zh_t.append(clean_z)

                    raw_exp = str(item.get("en_expansion") or "").strip()
                    en_exp = f"{raw_exp} {' '.join(en_t)}".strip()
                    if en_t or zh_t:
                        extracted_concepts.append(
                            {
                                "concept_id": cid,
                                "en_triggers": en_t,
                                "zh_triggers": zh_t,
                                "en_expansion": en_exp,
                            }
                        )
    except Exception:
        extracted_concepts = []

    if extracted_concepts:
        registry.register_document_concepts(
            content_hash=content_hash,
            filename=filename,
            concepts=extracted_concepts,
            model_used=f"ollama/{model_name}",
        )

    return extracted_concepts, f"ollama/{model_name}"


def translate_query_with_ollama(query: str) -> str:
    """Translate an unseen Chinese query into English search keywords via local Ollama (with disk cache)."""
    registry = get_concept_registry()
    cached = registry.get_cached_query_translation(query)
    if cached:
        return cached

    model_name = _select_ollama_model()
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Translate the user's Chinese search query into concise English search keywords. Return ONLY JSON: {\"english_keywords\": \"...\"}",
            },
            {"role": "user", "content": query},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 120},
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            parsed = json.loads(resp.json().get("message", {}).get("content", "{}"))
            kw = str(parsed.get("english_keywords", "")).strip()
            if kw:
                registry.cache_query_translation(query, kw)
                return kw
    except Exception:
        return ""
    return ""


def expand_cross_lingual_query(query: str) -> Tuple[str, List[str]]:
    """Expand a Chinese or bilingual query using dynamically extracted PDF concepts (plus Ollama fallback)."""
    lowered = query.lower()
    matched_concepts: List[str] = []
    english_expansions: List[str] = []

    dynamic_concepts = get_concept_registry().get_all_concepts()
    for concept_id, en_triggers, zh_triggers, en_expansion in dynamic_concepts:
        if _matches_zh_triggers(query, zh_triggers) or any(et in lowered for et in en_triggers):
            matched_concepts.append(concept_id)
            if has_cjk(query) and en_expansion:
                english_expansions.append(en_expansion)

    # Fallback: if a CJK query didn't hit any trigger directly, translate on the fly via local Ollama
    if has_cjk(query) and not english_expansions:
        translated_kw = translate_query_with_ollama(query)
        if translated_kw:
            english_expansions.append(translated_kw)
            lowered_kw = translated_kw.lower()
            for concept_id, en_triggers, _, _ in dynamic_concepts:
                if any(et in lowered_kw for et in en_triggers) and concept_id not in matched_concepts:
                    matched_concepts.append(concept_id)

    if english_expansions:
        expanded = f"{query} {' '.join(english_expansions)}"
    else:
        expanded = query

    return expanded, matched_concepts


def _extract_features(text: str) -> list[tuple[str, float]]:
    """Extract lexical, character n-gram, and dynamically generated cross-lingual concept features."""
    lowered = text.lower().strip()
    features: list[tuple[str, float]] = []

    # 1. Inject high-weight Dynamic Concept dimensions generated by local Ollama at PDF ingestion time
    for concept_id, en_triggers, zh_triggers, _ in get_concept_registry().get_all_concepts():
        if any(et in lowered for et in en_triggers) or _matches_zh_triggers(text, zh_triggers):
            features.append((concept_id, 5.5))
            features.append((f"{concept_id}:anchor", 4.0))

    # 2. Extract English words and CJK character unigrams/bigrams
    raw_chunks = _TOKEN_RE.findall(lowered)
    en_tokens: list[str] = []
    for chunk in raw_chunks:
        if has_cjk(chunk):
            for ch in chunk:
                features.append((f"cjk1:{ch}", 1.5))
            for i in range(len(chunk) - 1):
                features.append((f"cjk2:{chunk[i:i+2]}", 2.4))
        elif chunk not in STOPWORDS:
            en_tokens.append(chunk)

    for tok in en_tokens:
        features.append((f"u:{tok}", 2.0))
        if len(tok) >= 4:
            for i in range(len(tok) - 2):
                features.append((f"c3:{tok[i:i+3]}", 0.5))

    for i in range(len(en_tokens) - 1):
        features.append((f"b:{en_tokens[i]}_{en_tokens[i+1]}", 1.3))

    if not features:
        features.append(("__empty__", 1.0))

    return features


def _embed_via_ollama(text: str, model_name: str = EMBEDDING_MODEL_NAME, dims: int = EMBEDDING_DIMS) -> Optional[List[float]]:
    """Compute dense embeddings via Ollama /api/embed if configured."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
                json={"model": model_name, "input": text},
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and isinstance(embeddings[0], list):
                    vec = embeddings[0]
                    # Resize or pad if dimension mismatch
                    if len(vec) > dims:
                        vec = vec[:dims]
                    elif len(vec) < dims:
                        vec = vec + [0.0] * (dims - len(vec))
                    norm = math.sqrt(sum(v * v for v in vec))
                    return [round(v / (norm + 1e-9), 6) for v in vec]
    except Exception:
        pass
    return None


def embed_text(text: str, dims: int = EMBEDDING_DIMS) -> List[float]:
    """Compute an L2-normalized dense vector for Elasticsearch int8_hnsw.
    
    Routes to Ollama embedding model when RAG_EMBEDDING_PROVIDER='ollama',
    otherwise uses dynamic semantic feature hashing.
    """
    if EMBEDDING_PROVIDER == "ollama":
        ollama_vec = _embed_via_ollama(text, dims=dims)
        if ollama_vec:
            return ollama_vec

    vec = [0.0] * dims
    features = _extract_features(text)

    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx1 = int.from_bytes(digest[0:2], "little") % dims
        sign1 = 1.0 if (digest[2] & 1) == 0 else -1.0
        idx2 = int.from_bytes(digest[4:6], "little") % dims
        sign2 = 1.0 if (digest[6] & 1) == 0 else -1.0

        vec[idx1] += sign1 * weight
        vec[idx2] += sign2 * (weight * 0.6)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        vec[0] = 1.0
        return vec

    return [round(v / norm, 6) for v in vec]


class CrossLingualEmbedder:
    """Component: Configurable Vector Embedder implementing `EmbedderPort`."""

    def __init__(
        self,
        provider: str = EMBEDDING_PROVIDER,
        dimensions: int = EMBEDDING_DIMS,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.provider = provider
        self.dimensions = dimensions
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        return embed_text(text, dims=self.dimensions)
