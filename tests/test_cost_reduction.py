"""Episode 7: caches, routing and token optimization. Offline: in-memory store, scripted models."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.rag import telemetry
from src.rag.answer_cache import AnswerCache, CachePolicy, guard_allows, normalize_question
from src.rag.cache_store import MemoryStore, set_cache_store
from src.rag.embedding_cache import CachingEmbedder
from src.rag.embeddings import CrossLingualEmbedder, create_embedding_provider
from src.rag.guardrails import GuardrailEngine, GuardrailPolicy, TokenLedger
from src.rag.llm_cache import StoreLLMCache
from src.rag.pipeline import PipelineOptions
from src.rag.routing import is_single_fact, model_for
from src.rag.synthesizer import GroundedAnswerSynthesizer, PromptPolicy, compress
from src.rag.trust import AnswerVerifier


@pytest.fixture(autouse=True)
def fresh_store() -> Any:
    store = MemoryStore()
    set_cache_store(store)
    yield store
    set_cache_store(None)


class CountingProvider:
    """feature_hash vectors, with a count of calls and texts."""

    def __init__(self) -> None:
        self._inner = create_embedding_provider("feature_hash")
        self.calls = 0
        self.texts = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def manifest(self) -> Any:
        return self._inner.manifest

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        self.texts += len(texts)
        return self._inner.embed_documents(texts)


# --- Embedding cache -----------------------------------------------------------------------


def test_embedding_cache_serves_repeats_and_batches_misses():
    provider = CountingProvider()
    embedder = CachingEmbedder(CrossLingualEmbedder(provider))
    first = embedder.embed_batch(["alpha", "beta", "alpha"], owner=("acme", "doc1"))
    assert provider.calls == 1 and provider.texts == 2  # one call, and the repeat inside the batch is free
    again = embedder.embed_batch(["beta", "alpha"], owner=("acme", "doc2"))
    assert provider.calls == 1 and again == [first[1], first[0]]


def test_cached_vectors_match_uncached_ones():
    plain = CrossLingualEmbedder(create_embedding_provider("feature_hash"))
    cached = CachingEmbedder(CrossLingualEmbedder(create_embedding_provider("feature_hash")))
    for _ in range(2):
        assert [round(x, 5) for x in cached.embed("Rotate passwords every 180 days")] == [
            round(x, 5) for x in plain.embed("Rotate passwords every 180 days")
        ]


def test_embedding_cache_is_per_tenant_and_purged_with_the_document(fresh_store: MemoryStore):
    provider = CountingProvider()
    embedder = CachingEmbedder(CrossLingualEmbedder(provider))
    embedder.embed_batch(["shared text"], owner=("acme", "doc1"))
    embedder.embed_batch(["shared text"], owner=("globex", "doc9"))
    assert provider.calls == 2  # no hit across tenants, so no timing signal either
    assert embedder.purge_owner("acme", ["doc1"]) == 1
    embedder.embed_batch(["shared text"], owner=("globex", "doc9"))
    assert provider.calls == 2  # Globex's copy survived Acme's erasure


# --- LLM call cache and cost accounting ---------------------------------------------------------


def _fake_model(*replies: str, cache: Any = None, input_tokens: int = 50, output_tokens: int = 7) -> Any:
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    return GenericFakeChatModel(
        messages=iter([AIMessage(content=r, usage_metadata=usage) for r in replies]),
        callbacks=[telemetry.genai_handler()],
        cache=cache,
    )


def test_an_llm_cache_hit_is_saved_not_spent_and_not_charged_to_the_budget():
    ledger = TokenLedger()
    GuardrailEngine(GuardrailPolicy(enabled=True), ledger=ledger)  # registers the ledger as a usage listener
    model = _fake_model("x", "x", cache=StoreLLMCache())
    records = []
    for _ in range(2):
        with telemetry.request_scope("budget-tenant", "llm") as record:
            with telemetry.stage("query_transform"):
                model.invoke("same prompt")
            record.outcome = "answered"
        records.append(record.to_dict())
    assert records[0]["tokens"]["input"] == 50 and records[1]["tokens"]["input"] == 0
    assert records[1]["saved"]["input_tokens"] == 50 and records[1]["llm_calls"][0]["cached"]
    assert ledger.usage("budget-tenant")["total"] == 57


# --- Answer cache --------------------------------------------------------------------------


def test_normalized_questions_share_a_key():
    assert normalize_question("  What is the  Cash Reserve? ") == normalize_question("what is the cash reserve")


@pytest.mark.parametrize(
    "asked, cached, allowed",
    [
        ("What is the minimum cash reserve?", "What's the minimum reserve of cash?", True),
        ("What is the minimum cash reserve?", "What is the maximum cash reserve?", False),
        ("What was revenue in Q3?", "What was revenue in Q4?", False),
        ("What changed in the 2026 policy?", "What changed in the 2023 policy?", False),
        ("Is travel to Shenzhen approved?", "Is travel to Shenzhen not approved?", False),
        ("What is the reserve in HKD?", "What is the reserve in EUR?", False),
        ("What does Acme require?", "What does Globex require?", False),
    ],
)
def test_the_guard_rejects_near_misses(asked: str, cached: str, allowed: bool):
    assert guard_allows(asked, cached)[0] is allowed


def _cache(**policy: Any) -> AnswerCache:
    embedder = CachingEmbedder(CrossLingualEmbedder(create_embedding_provider("feature_hash")))
    return AnswerCache(CachePolicy(**{"exact": True, **policy}), embedder=embedder)


def _ctx(cache: AnswerCache, question: str, tenant: str = "acme", acl: tuple = ("finance",)) -> Any:
    return cache.context(question, tenant, acl, mode="llm", options={}, top_k=4, profile=None, model="m")


def test_exact_hits_need_the_same_tenant_roles_and_generation():
    cache = _cache()
    cache.store_answer(_ctx(cache, "What is the reserve?"), {"answer": "HKD 450M", "authorized_chunks": []}, {"input_tokens": 700})
    assert cache.lookup(_ctx(cache, "what is the reserve")) is not None
    assert cache.lookup(_ctx(cache, "What is the reserve?", acl=("__public__",))) is None
    assert cache.lookup(_ctx(cache, "What is the reserve?", tenant="globex")) is None
    cache.on_change("acme", "write", ["doc1"])
    assert cache.lookup(_ctx(cache, "What is the reserve?")) is None


def test_a_removal_deletes_the_tenants_answers(fresh_store: MemoryStore):
    cache = _cache()
    cache.store_answer(_ctx(cache, "q1"), {"answer": "a", "authorized_chunks": []}, {})
    cache.store_answer(_ctx(cache, "q2", tenant="globex"), {"answer": "b", "authorized_chunks": []}, {})
    cache.on_change("acme", "remove", ["doc1"])
    assert not [k for k in fresh_store.keys("*:ans:acme:*")]
    assert [k for k in fresh_store.keys("*:ans:globex:*")]


def test_an_answer_never_outlives_its_sources_expiry():
    cache = _cache()
    expired = {"answer": "a", "authorized_chunks": [{"expires_at": "2000-01-01"}]}
    cache.store_answer(_ctx(cache, "q"), expired, {})
    assert cache.lookup(_ctx(cache, "q")) is None


def test_semantic_hits_respect_the_threshold_and_the_guard():
    cache = _cache(semantic=True, threshold=0.5)
    cache.store_answer(_ctx(cache, "What is the minimum cash reserve for treasury?"), {"answer": "HKD 450M", "authorized_chunks": []}, {})
    hit = cache.lookup(_ctx(cache, "What's the minimum cash reserve for the treasury?"))
    assert hit is not None and hit["cache"]["layer"] == "answer_semantic"
    assert cache.lookup(_ctx(cache, "What is the maximum cash reserve for treasury?")) is None


# --- Service ---------------------------------------------------------------------------


def _service(tmp_path: Any, model: Any, *, answer_cache: AnswerCache | None = None, guardrails: Any = None) -> Any:
    from src.rag.chunker import create_chunker
    from src.rag.es_client import ElasticsearchIndexManager
    from src.rag.extractor import DocumentParserRegistry
    from src.rag.ingestor import PdfIngestor
    from src.rag.repository import DocumentRepository
    from src.rag.retriever import HybridRetriever
    from src.rag.service import RBACKnowledgeBaseService

    provider = create_embedding_provider("feature_hash")
    embedder = CachingEmbedder(CrossLingualEmbedder(provider))
    manager = ElasticsearchIndexManager(es_url="http://non-existent-es:9200", dimensions=provider.dimensions)
    repository = DocumentRepository(es_manager=manager, uploads_dir=tmp_path / "u", registry_path=tmp_path / "r.json")
    cross_lingual = SimpleNamespace(
        is_cross_lingual=lambda text: False,
        extract_concepts_at_ingestion=lambda filename, full_text, tenant=None: ([], "none"),
        prepare_query=lambda question, allow_llm_fallback=False, tenant=None: (question, []),
    )
    service = RBACKnowledgeBaseService(
        index_manager=manager,
        repository=repository,
        ingestor=PdfIngestor(DocumentParserRegistry(), create_chunker("sentence", max_tokens=100, overlap_tokens=0), cross_lingual, embedder, repository),
        cross_lingual=cross_lingual,
        embedder=embedder,
        retriever=HybridRetriever(manager),
        synthesizer=GroundedAnswerSynthesizer(lambda **_: model),
        seeder=object(),  # type: ignore[arg-type]
        guardrails=guardrails or GuardrailEngine(GuardrailPolicy()),
        answer_cache=answer_cache or AnswerCache(CachePolicy(exact=True), embedder=embedder),
    )
    service.ingest_pdf_bytes("policy.md", b"Employees must rotate their network password every 180 days.", source_key="policy")
    return service


def _ask(service: Any, question: str = "rotate network password") -> Dict[str, Any]:
    return service.query_and_chat(question, synthesis_mode="llm", pipeline_options=PipelineOptions(), tenant="default")


def test_a_repeated_question_is_answered_from_the_cache_for_free(tmp_path: Any):
    service = _service(tmp_path, _fake_model("- Rotate every 180 days [S1]", "- unused [S1]"))
    first, second = _ask(service), _ask(service)
    assert first["telemetry"]["outcome"] == "answered" and first["telemetry"]["tokens"]["input"] == 50
    assert second["telemetry"]["outcome"] == "answered_cached" and second["telemetry"]["tokens"]["input"] == 0
    assert second["telemetry"]["saved"]["input_tokens"] == 50 and second["answer"] == first["answer"]
    assert second["cache"]["layer"] == "answer_exact"


def test_a_new_version_of_a_source_makes_the_cached_answer_unreachable(tmp_path: Any):
    service = _service(tmp_path, _fake_model("- Rotate every 180 days [S1]", "- Rotate every 90 days [S1]"))
    _ask(service)
    service.ingest_pdf_bytes("policy.md", b"Employees must rotate their network password every 90 days.", source_key="policy")
    assert "90 days" in _ask(service)["answer"]


def test_deleting_a_document_deletes_the_answers_built_on_it(tmp_path: Any, fresh_store: MemoryStore):
    service = _service(tmp_path, _fake_model("- Rotate every 180 days [S1]"))
    _ask(service)
    assert fresh_store.keys("*:ans:default:*")
    doc_id = service.list_documents(tenant="default")[0]["doc_id"]
    service.delete_document(doc_id, tenant="default")
    assert not fresh_store.keys("*:ans:default:*") and not fresh_store.keys("*:emb-owner:default:*")


def test_degraded_answers_are_not_cached(tmp_path: Any):
    def down(_: Any) -> Any:
        raise ConnectionError("model down")

    service = _service(tmp_path, RunnableLambda(down))
    assert _ask(service)["telemetry"]["outcome"] == "degraded_llm_fallback"
    assert _ask(service)["telemetry"]["outcome"] == "degraded_llm_fallback"


def test_a_cached_answer_is_served_even_over_budget(tmp_path: Any):
    ledger = TokenLedger()
    engine = GuardrailEngine(GuardrailPolicy(enabled=True, daily_token_budget=10), ledger=ledger)
    service = _service(tmp_path, _fake_model("- Rotate every 180 days [S1]"), guardrails=engine)
    _ask(service)  # spends 57 tokens, over the budget of 10
    assert _ask(service)["telemetry"]["outcome"] == "answered_cached"


# --- Routing ---------------------------------------------------------------------------------


def test_routing_is_off_unless_asked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RAG_ROUTING", raising=False)
    assert model_for("verify") is None
    monkeypatch.setenv("RAG_ROUTING", "static")
    monkeypatch.setenv("RAG_ROUTE_VERIFY", "small")
    monkeypatch.setenv("RAG_MODEL_SMALL", "tiny:1b")
    with telemetry.request_scope("acme", "llm") as record:
        assert model_for("verify") == "tiny:1b"
        record.outcome = "answered"
    assert record.routes == [{"task": "verify", "tier": "small", "model": "tiny:1b", "stage": "none"}]


@pytest.mark.parametrize(
    "question, single",
    [
        ("What is the minimum cash reserve?", True),
        ("How many days until a password expires?", True),
        ("Why did the policy change and what does it mean for travel?", False),
        ("Compare the 2023 and 2026 password policies", False),
    ],
)
def test_single_fact_questions(question: str, single: bool):
    assert is_single_fact(question) is single


def _recording_factory(replies: Dict[str, str], seen: List[str]) -> Any:
    def factory(**kwargs: Any) -> Any:
        model = kwargs.get("model") or "default"

        def respond(_: Any) -> Any:
            seen.append(model)
            return SimpleNamespace(content=replies[model])

        return RunnableLambda(respond)

    return factory


_CHUNKS = [{"chunk_id": "a", "filename": "policy.md", "page_number": 1, "tags": [], "is_public": True,
            "content": "Employees must rotate their network password every 180 days."}]


def _synth(factory: Any) -> str:
    synthesizer = GroundedAnswerSynthesizer(factory)
    return synthesizer.synthesize("rotate password", "rotate password", [], False, [], _CHUNKS, [], synthesis_mode="llm", ranked_by_model=True)


def test_the_cascade_keeps_a_good_small_answer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_ROUTING", "cascade")
    monkeypatch.setenv("RAG_MODEL_SMALL", "small-m")
    monkeypatch.setenv("RAG_MODEL_LARGE", "large-m")
    seen: List[str] = []
    answer = _synth(_recording_factory({"small-m": "- Rotate every 180 days [S1]", "large-m": "x"}, seen))
    assert seen == ["small-m"] and "180" in answer


@pytest.mark.parametrize("small_reply", ["NOT_FOUND", "- Rotate every 90 days [S1]", "- Rotate every 180 days [S7]"])
def test_the_cascade_escalates_when_cheap_checks_fail(monkeypatch: pytest.MonkeyPatch, small_reply: str):
    monkeypatch.setenv("RAG_ROUTING", "cascade")
    monkeypatch.setenv("RAG_MODEL_SMALL", "small-m")
    monkeypatch.setenv("RAG_MODEL_LARGE", "large-m")
    seen: List[str] = []
    answer = _synth(_recording_factory({"small-m": small_reply, "large-m": "- Rotate every 180 days [S1]"}, seen))
    assert seen == ["small-m", "large-m"] and "180" in answer


# --- Token optimization -----------------------------------------------------------------------


def test_compression_keeps_the_sentence_with_the_answer():
    text = ("The handbook covers many topics. Offices open at nine. " * 6) + "Passwords must be rotated every 180 days. Parking is free."
    short = compress(text, {"password", "rotat"}, 120)
    assert "180 days" in short and len(short) <= 160


def test_compact_headers_drop_trust_details_without_a_conflict():
    seen: Dict[str, str] = {}

    def factory(**_: Any) -> Any:
        return RunnableLambda(lambda prompt: seen.setdefault("p", prompt.to_string()) and SimpleNamespace(content="- ok [S1]"))

    GroundedAnswerSynthesizer(factory, prompt_policy=PromptPolicy(compact_sources=True)).synthesize(
        "rotate password", "rotate password", [], False, [], _CHUNKS, [], synthesis_mode="llm", ranked_by_model=True
    )
    assert "[S1] policy.md p.1" in seen["p"] and "Trust tier" not in seen["p"] and "trust tier" not in seen["p"]


def test_compact_verification_skips_the_llm_when_values_settle_it():
    calls: List[int] = []

    class Spy:
        def verify(self, sentences: Any, sources: Any) -> Dict[int, Any]:
            calls.append(len(sentences))
            return {}

    sources = {"S1": {"text": "Rotate every 180 days."}}
    report = AnswerVerifier(Spy(), compact=True).report("Rotate every 180 days [S1].", sources)  # type: ignore[arg-type]
    assert not calls and report.sentences[0].status == "supported" and "llm_skipped" in report.checks
    AnswerVerifier(Spy(), compact=True).report("Rotation is mandatory [S1].", sources)  # type: ignore[arg-type]
    assert calls == [1]


# --- Redis (skipped without one) ------------------------------------------------------------------


def _redis_store(monkeypatch: pytest.MonkeyPatch) -> Any:
    import os
    import uuid

    from src.rag.cache_store import RedisStore

    monkeypatch.setenv("RAG_CACHE_PREFIX", f"beyondrag-pytest-{uuid.uuid4().hex[:8]}")
    try:
        return RedisStore(os.getenv("REDIS_TEST_URL", "redis://localhost:6379/0"))
    except Exception:
        pytest.skip("No Redis reachable")


def test_redis_store_keys_ttl_and_vector_scope(monkeypatch: pytest.MonkeyPatch):
    store = _redis_store(monkeypatch)
    from src.rag.cache_store import cache_prefix

    try:
        store.set(f"{cache_prefix()}:a", b"1", ttl=60)
        assert store.mget([f"{cache_prefix()}:a", f"{cache_prefix()}:b"]) == [b"1", None]
        store.vector_add("k1", [1.0, 0.0, 0.0], "scopeA", b"payload-a", 60)
        store.vector_add("k2", [0.99, 0.1, 0.0], "scopeB", b"payload-b", 60)
        hits = store.vector_search([1.0, 0.0, 0.0], "scopeA", k=3)
        assert [(key, payload) for key, _, payload in hits] == [("k1", b"payload-a")] and hits[0][1] > 0.99
        assert store.delete(["k1"]) == 1 and store.vector_search([1.0, 0.0, 0.0], "scopeA") == []
    finally:
        store.delete(store.keys(f"{cache_prefix()}:*"))
        store._redis.ft(f"{cache_prefix()}:semantic-index").dropindex(delete_documents=True)


def test_shared_ledger_counts_like_the_file_ledger(fresh_store: MemoryStore):
    from src.rag.guardrails import SharedTokenLedger

    one, two = SharedTokenLedger(fresh_store), SharedTokenLedger(fresh_store)  # two app replicas
    one.add("acme", 100, 20)
    two.add("acme", 5, 5)
    assert one.usage("acme") == two.usage("acme") == {"input": 105, "output": 25, "total": 130}
    assert two.forget_tenant("acme") == 1 and one.usage("acme")["total"] == 0


def test_an_index_reset_keeps_embeddings_but_drops_answers(fresh_store: MemoryStore):
    cache = _cache()
    cache.store_answer(_ctx(cache, "q"), {"answer": "a", "authorized_chunks": []}, {})
    cache.on_change("acme", "reset", ["doc1"])
    assert not fresh_store.keys("*:ans:acme:*")
