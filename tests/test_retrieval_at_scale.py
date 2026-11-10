"""Episode 3: retrieval at scale. Offline by default; live-Elasticsearch tests skip themselves."""

from __future__ import annotations

import math
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

from src.rag.chunker import create_chunker
from src.rag.embeddings import CrossLingualEmbedder, create_embedding_provider
from src.rag.es_client import ElasticsearchIndexManager, VECTOR_INDEX_TYPES
from src.rag.extractor import DocumentParserRegistry
from src.rag.fusion import reciprocal_rank_fusion, reorder_for_attention
from src.rag.ingestor import PdfIngestor
from src.rag.multi_hop import MultiHopPlanner
from src.rag.parent_child import ParentChildSplitter, expand_to_parents
from src.rag.pipeline import PipelineOptions, PipelineResult, RetrievalPipeline
from src.rag.query_transform import LlmQueryTransformer, QueryVariant
from src.rag.reranker import CrossEncoderUnavailable, FastEmbedCrossEncoder
from src.rag.repository import DocumentRepository
from src.rag.retriever import HybridRetriever
from src.rag.service import RBACKnowledgeBaseService
from src.rag.synthesizer import GroundedAnswerSynthesizer


# --- Reciprocal Rank Fusion ----------------------------------------------------------


def _ranked(*ids: str) -> List[Dict[str, Any]]:
    return [{"chunk_id": chunk_id, "score": 1000.0 - index} for index, chunk_id in enumerate(ids)]


def test_rrf_rewards_appearing_in_both_lists_over_winning_one():
    fused = reciprocal_rank_fusion([_ranked("lexical-only", "shared"), _ranked("dense-only", "shared")])
    assert fused[0]["chunk_id"] == "shared"


def test_rrf_uses_ranks_so_a_huge_score_scale_cannot_dominate():
    lexical = [{"chunk_id": "a", "score": 99999.0}, {"chunk_id": "b", "score": 88888.0}]
    dense = [{"chunk_id": "b", "score": 0.91}, {"chunk_id": "a", "score": 0.90}]
    fused = reciprocal_rank_fusion([lexical, dense])
    assert math.isclose(fused[0]["score"], fused[1]["score"])


def test_rrf_score_matches_the_formula():
    fused = reciprocal_rank_fusion([_ranked("a"), _ranked("x", "a")], k=60)
    expected = round(1 / 61 + 1 / 62, 6)
    assert next(item for item in fused if item["chunk_id"] == "a")["score"] == expected


# --- Lost in the middle --------------------------------------------------------------


def test_reorder_puts_best_first_second_last_and_weakest_in_the_middle():
    assert reorder_for_attention([1, 2, 3, 4, 5]) == [1, 3, 5, 4, 2]
    assert reorder_for_attention([1, 2, 3, 4, 5, 6]) == [1, 3, 5, 6, 4, 2]


@pytest.mark.parametrize("items", [[], [1], [1, 2]])
def test_reorder_handles_short_lists(items: List[int]):
    assert reorder_for_attention(items) == items


def test_reorder_keeps_every_item():
    items = list(range(15))
    assert sorted(reorder_for_attention(items)) == items


def test_llm_prompt_uses_the_reordered_context():
    seen: Dict[str, Any] = {}

    def fake_llm(**_: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        def respond(prompt: Any) -> SimpleNamespace:
            seen["prompt"] = prompt.to_string()
            return SimpleNamespace(content="answer")

        return RunnableLambda(respond)

    chunks = [
        {"chunk_id": name, "filename": f"{name}.md", "page_number": 1, "tags": [], "content": f"text {name}", "is_public": True}
        for name in ("first", "second", "third", "fourth")
    ]
    synthesizer = GroundedAnswerSynthesizer(llm_factory=fake_llm)
    synthesizer.synthesize("q", "text", [], False, [], chunks, [], synthesis_mode="llm")
    prompt = seen["prompt"]
    order = sorted(chunks, key=lambda c: prompt.index(c["filename"]))
    assert [c["chunk_id"] for c in order] == ["first", "third", "fourth", "second"]


def test_fast_answer_still_quotes_the_best_chunk():
    chunks = [
        {"chunk_id": "best", "filename": "best.md", "page_number": 1, "tags": [], "content": "cash buffer best", "is_public": True},
        {"chunk_id": "next", "filename": "next.md", "page_number": 1, "tags": [], "content": "cash buffer next", "is_public": True},
    ]
    answer = GroundedAnswerSynthesizer().synthesize("cash buffer", "cash buffer", [], False, [], chunks, [])
    assert "best.md" in answer


# --- Exact match analyzer and index options -------------------------------------------


def _captured_mapping(**kwargs: Any) -> Dict[str, Any]:
    manager = ElasticsearchIndexManager(es_url="http://non-existent-es:9200", **kwargs)
    captured: Dict[str, Any] = {}

    class Indices:
        def exists(self, index: str) -> bool:
            return False

        def create(self, **body: Any) -> None:
            captured.update(body)

    manager.client = SimpleNamespace(indices=Indices(), ping=lambda: True)  # type: ignore[assignment]
    manager.ensure_index()
    return captured


def test_vector_index_type_is_configurable_and_defaults_to_int8():
    assert _captured_mapping()["mappings"]["properties"]["embedding"]["index_options"]["type"] == "int8_hnsw"
    for vector_type in VECTOR_INDEX_TYPES:
        mapping = _captured_mapping(vector_index_type=vector_type)
        assert mapping["mappings"]["properties"]["embedding"]["index_options"]["type"] == vector_type


def test_unknown_vector_index_type_is_refused():
    with pytest.raises(ValueError, match="Unknown vector index type"):
        ElasticsearchIndexManager(es_url="http://non-existent-es:9200", vector_index_type="float128")


def test_mapping_adds_a_code_preserving_subfield_and_unindexed_parent():
    mapping = _captured_mapping()
    properties = mapping["mappings"]["properties"]
    assert properties["content"]["fields"]["codes"]["analyzer"] == "code_preserving"
    assert properties["parent_content"]["index"] is False
    tokenizer = mapping["settings"]["analysis"]["tokenizer"]["code_tokenizer"]
    assert "-" not in tokenizer["tokenize_on_chars"]


# --- In-memory hybrid retrieval -------------------------------------------------------


class ConstantEmbedder:
    """Every text gets the same vector, so only the lexical side can tell them apart."""

    dimensions = 4

    def embed(self, text: str) -> List[float]:
        return [0.5, 0.5, 0.5, 0.5]


def _memory_retriever(chunks: Sequence[Dict[str, Any]]) -> HybridRetriever:
    manager = ElasticsearchIndexManager(es_url="http://non-existent-es:9200", dimensions=4)
    manager.memory_chunks.extend(chunks)
    return HybridRetriever(manager)


def _record(chunk_id: str, content: str, **extra: Any) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id,
        "filename": f"{chunk_id}.md",
        "page_number": 1,
        "content": content,
        "acl_tags": ["__public__"],
        "tenant": "default",
        "embedding": [0.5, 0.5, 0.5, 0.5],
        **extra,
    }


def test_hybrid_finds_an_exact_code_when_dense_cannot_separate_documents():
    retriever = _memory_retriever(
        [
            _record("generic", "General error handling manual for all services"),
            _record("wrong-code", "Error ERR-1024 means the disk is full"),
            _record("right-code", "Error ERR-1042 means the token expired"),
        ]
    )
    vector = ConstantEmbedder().embed("")
    hybrid = retriever.search("ERR-1042", vector, ["__public__"], top_k=1, fusion="rrf", tenant="default")
    assert hybrid.authorized_chunks[0]["chunk_id"] == "right-code"


def test_acl_filters_every_retriever_mode():
    retriever = _memory_retriever(
        [_record("secret", "ACME-X200 price", acl_tags=["finance"]), _record("open", "ACME-X200 manual")]
    )
    vector = ConstantEmbedder().embed("")
    for mode in ("rrf", "dense", "lexical"):
        outcome = retriever.search("ACME-X200", vector, ["__public__"], top_k=5, fusion=mode, tenant="default")
        assert [c["chunk_id"] for c in outcome.authorized_chunks] == ["open"], mode


def test_superseded_and_expired_chunks_stay_out_of_lexical_results():
    retriever = _memory_retriever(
        [
            _record("old", "ACME-X200 old price", is_current=False),
            _record("expired", "ACME-X200 expired price", expires_at="2020-01-01"),
            _record("now", "ACME-X200 current price"),
        ]
    )
    outcome = retriever.search("ACME-X200", None, ["__public__"], top_k=5, fusion="lexical", tenant="default")
    assert [c["chunk_id"] for c in outcome.authorized_chunks] == ["now"]


def test_unknown_fusion_mode_is_refused():
    retriever = _memory_retriever([])
    with pytest.raises(ValueError, match="Unknown fusion mode"):
        retriever.search("q", None, ["__public__"], fusion="magic", tenant="default")


# --- Query transformation -------------------------------------------------------------


class ScriptedLlm:
    """Returns the next structured answer. An Exception in the script is raised."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.prompts: List[str] = []

    def __call__(self, **_: Any) -> "ScriptedLlm":
        return self

    def with_structured_output(self, schema: Any) -> "ScriptedLlm":
        self.schema = schema
        return self

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return self.schema(**answer)


def test_off_mode_makes_no_llm_call():
    llm = ScriptedLlm()
    plan = LlmQueryTransformer(llm).plan("q", "q", "off")
    assert [variant.kind for variant in plan.variants] == ["original"] and not llm.prompts


def test_rewrites_keep_the_original_query_and_drop_duplicates():
    llm = ScriptedLlm({"queries": ["authentication timeout", "APP KEEPS CRASHING", "authentication timeout", "login session expiry", "a", "b"]})
    plan = LlmQueryTransformer(llm).plan("app keeps crashing", "app keeps crashing", "multi")
    assert plan.variants[0] == QueryVariant("original", "app keeps crashing")
    rewrites = [v.text for v in plan.variants if v.kind == "rewrite"]
    assert rewrites[0] == "authentication timeout" and rewrites.count("authentication timeout") == 1
    assert len(rewrites) <= 3


def test_hyde_is_used_for_vectors_only():
    llm = ScriptedLlm({"passage": "The authentication service times out."})
    hyde = LlmQueryTransformer(llm).plan("why", "why", "hyde").variants[1]
    assert (hyde.kind, hyde.lexical, hyde.dense) == ("hyde", False, True)


def test_llm_failure_falls_back_to_the_original_query_and_is_reported():
    llm = ScriptedLlm(RuntimeError("model down"), RuntimeError("model down"))
    plan = LlmQueryTransformer(llm).plan("q", "q", "both")
    assert [variant.kind for variant in plan.variants] == ["original"]
    assert len(plan.errors) == 2


def test_failed_transform_is_not_cached():
    llm = ScriptedLlm(RuntimeError("down"), {"queries": ["good"]})
    transformer = LlmQueryTransformer(llm)
    assert len(transformer.plan("q", "q", "multi").variants) == 1
    assert len(transformer.plan("q", "q", "multi").variants) == 2


def test_glossary_expands_known_acronyms_in_the_prompt():
    llm = ScriptedLlm({"queries": []})
    LlmQueryTransformer(llm, glossary={"SSO": "single sign-on"}).plan("SSO broken?", "SSO broken?", "multi")
    assert "SSO = single sign-on" in llm.prompts[0]


# --- Parent-child ---------------------------------------------------------------------


LONG_TEXT = "\n\n".join(
    " ".join(f"Sentence {paragraph}.{sentence} says something about topic {paragraph}." for sentence in range(30))
    for paragraph in range(6)
)


def test_children_are_small_and_parents_contain_them():
    splitter = ParentChildSplitter(parent_tokens=300, child_tokens=60, child_overlap=10)
    children = splitter.split(LONG_TEXT)
    tokenizer = create_chunker("sentence", max_tokens=60, overlap_tokens=0).config.tokenizer
    assert len(children) > len({child.parent_index for child in children})
    for child in children:
        assert tokenizer.count_tokens(child.text) <= 60
        assert tokenizer.count_tokens(child.parent_text) <= 300
        assert child.text.split()[-1] in child.parent_text


def test_child_must_be_smaller_than_parent():
    with pytest.raises(ValueError, match="must be smaller"):
        ParentChildSplitter(parent_tokens=100, child_tokens=100)


def test_expansion_returns_the_parent_once_and_keeps_the_best_rank():
    ranked = [
        {"chunk_id": "c1", "content": "child one", "parent_id": "p1", "parent_content": "PARENT ONE"},
        {"chunk_id": "c2", "content": "child two", "parent_id": "p2", "parent_content": "PARENT TWO"},
        {"chunk_id": "c3", "content": "child three", "parent_id": "p1", "parent_content": "PARENT ONE"},
        {"chunk_id": "t1", "content": "| a table |"},
    ]
    expanded, merged = expand_to_parents(ranked)
    assert [c["content"] for c in expanded] == ["PARENT ONE", "PARENT TWO", "| a table |"]
    assert expanded[0]["matched_children"] == ["child one", "child three"]
    assert merged == 1


@pytest.fixture
def stack(tmp_path: Any) -> SimpleNamespace:
    """A real ingestor and repository on the in-memory store, with parent-child on."""
    provider = create_embedding_provider("feature_hash")
    embedder = CrossLingualEmbedder(provider)
    manager = ElasticsearchIndexManager(es_url="http://non-existent-es:9200", dimensions=provider.dimensions)
    repository = DocumentRepository(es_manager=manager, uploads_dir=tmp_path / "uploads", registry_path=tmp_path / "registry.json")
    cross_lingual = SimpleNamespace(
        is_cross_lingual=lambda text: False,
        extract_concepts_at_ingestion=lambda filename, full_text, tenant=None: ([], "none"),
        prepare_query=lambda question, allow_llm_fallback=False, tenant=None: (question, []),
    )
    ingestor = PdfIngestor(
        extractor=DocumentParserRegistry(),
        chunker=create_chunker("sentence", max_tokens=100, overlap_tokens=10),
        cross_lingual=cross_lingual,
        embedder=embedder,
        repository=repository,
        parent_child=ParentChildSplitter(parent_tokens=120, child_tokens=30, child_overlap=5),
    )
    retriever = HybridRetriever(manager)
    service = RBACKnowledgeBaseService(
        index_manager=manager,
        repository=repository,
        ingestor=ingestor,
        cross_lingual=cross_lingual,
        embedder=embedder,
        retriever=retriever,
        synthesizer=GroundedAnswerSynthesizer(),
        seeder=object(),  # type: ignore[arg-type]
        reranker=None,
    )
    return SimpleNamespace(service=service, manager=manager, repository=repository, retriever=retriever, embedder=embedder)


def _policy(fact: str, name: str = "alpha") -> bytes:
    filler = " ".join(f"Background sentence {i} about {name} operations." for i in range(40))
    return f"{filler} Rule: the {name} limit is {fact}. {filler}".encode("utf-8")


def test_only_children_are_embedded_and_each_carries_its_parent(stack: SimpleNamespace):
    stack.service.ingest_pdf_bytes("alpha.md", _policy("HKD 500"))
    records = stack.manager.memory_chunks
    assert all(record["embedding"] and record["parent_id"] and record["parent_content"] for record in records)
    assert len({record["parent_id"] for record in records}) < len(records)


def test_retrieval_returns_the_parent_to_the_model(stack: SimpleNamespace):
    stack.service.ingest_pdf_bytes("alpha.md", _policy("HKD 500"))
    result = stack.service.query_and_chat("alpha limit rule", pipeline_options=PipelineOptions(), tenant="default")
    chunk = result["authorized_chunks"][0]
    assert len(chunk["content"]) > len(chunk["matched_children"][0])
    assert "parent_content" not in chunk
    assert result["retrieval_trace"]["parents"]["expanded"] >= 1


def test_parent_inherits_the_acl_of_its_child(stack: SimpleNamespace):
    stack.service.ingest_pdf_bytes("secret.md", _policy("HKD 900", "gamma"), tags=["finance"])
    blocked = stack.service.query_and_chat("gamma limit rule", user_roles=[], pipeline_options=PipelineOptions(), tenant="default")
    assert blocked["authorized_chunks"] == []
    allowed = stack.service.query_and_chat("gamma limit rule", user_roles=["finance"], pipeline_options=PipelineOptions(), tenant="default")
    assert "HKD 900" in " ".join(c["content"] for c in allowed["authorized_chunks"])


def test_erasure_removes_parents_with_their_children(stack: SimpleNamespace):
    person = "maya.chen@example.com"
    stack.service.ingest_pdf_bytes("hr.md", _policy(f"owned by {person}", "delta"), source_key="hr")
    assert any(person in record["parent_content"] for record in stack.manager.memory_chunks)
    assert all(person in record["subject_ids"] for record in stack.manager.memory_chunks if person in record["parent_content"])
    erasure = stack.service.erasure_service()
    receipt = erasure.execute(erasure.plan_subject(person))
    assert receipt.verified
    assert not any(person in str(record) for record in stack.manager.memory_chunks)


def test_a_new_version_retires_old_parents(stack: SimpleNamespace):
    stack.service.ingest_pdf_bytes("alpha.md", _policy("HKD 500"), source_key="alpha")
    stack.service.ingest_pdf_bytes("alpha.md", _policy("HKD 700"), source_key="alpha")
    result = stack.service.query_and_chat("alpha limit rule", pipeline_options=PipelineOptions(), tenant="default")
    blob = " ".join(c["content"] for c in result["authorized_chunks"])
    assert "HKD 700" in blob and "HKD 500" not in blob


def test_lab_documents_load_without_llm_concept_extraction(stack: SimpleNamespace):
    from src.rag.seeding import seed_lab_documents

    calls: List[str] = []
    stack.service._ingestor._cross_lingual = SimpleNamespace(  # noqa: SLF001
        extract_concepts_at_ingestion=lambda filename, full_text, tenant=None: calls.append(filename) or ([], "llm")
    )
    seeded = seed_lab_documents(stack.service)
    assert len(seeded) > 40 and not calls
    assert seed_lab_documents(stack.service) == []
    finance = next(d for d in seeded if d["filename"].startswith("revenue"))
    assert finance["tags"] == ["finance"]


# --- Two-stage re-ranking -------------------------------------------------------------


class KeywordReranker:
    """Fake cross-encoder: a text is relevant if it holds the keyword."""

    model_name = "fake-cross-encoder"

    def __init__(self, keyword: str = "price") -> None:
        self.keyword = keyword
        self.seen: List[str] = []

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        self.seen = list(texts)
        return [8.0 if self.keyword in text else -8.0 for text in texts]


def _pipeline(chunks: Sequence[Dict[str, Any]], reranker: Any = None, llm: Any = None) -> RetrievalPipeline:
    return RetrievalPipeline(
        _memory_retriever(chunks),
        ConstantEmbedder(),
        reranker=reranker,
        transformer=LlmQueryTransformer(llm or ScriptedLlm()),
    )


def test_reranker_promotes_the_relevant_chunk_from_deep_in_the_first_stage():
    chunks = [_record(f"history-{i}", f"history of the product part {i}") for i in range(20)]
    chunks.append(_record("answer", "the product price is 400 dollars"))
    reranker = KeywordReranker()
    result = _pipeline(chunks, reranker).run("product", "product", ["__public__"], top_k=3, options=PipelineOptions(rerank=True, candidates=50), tenant="default")
    assert result.chunks[0]["chunk_id"] == "answer"
    assert len(reranker.seen) == 21
    assert result.trace["rerank"]["status"] == "applied"
    moved = result.trace["rerank"]["moved"][0]
    assert moved["to"] == 1 and moved["from"] > 1


def test_rerank_scores_are_positive_so_trust_tiers_still_work():
    chunks = [
        _record("draft", "price draft", trust_tier="draft_notes"),
        _record("policy", "price policy", trust_tier="official_policy"),
    ]
    result = _pipeline(chunks, KeywordReranker()).run("price", "price", ["__public__"], top_k=2, options=PipelineOptions(rerank=True), tenant="default")
    assert [c["chunk_id"] for c in result.chunks] == ["policy", "draft"]
    assert all(c["score"] > 0 for c in result.chunks)


def test_reranker_failure_keeps_first_stage_order_and_says_so():
    class Broken:
        model_name = "broken"

        def score(self, query: str, texts: Sequence[str]) -> List[float]:
            raise CrossEncoderUnavailable("no model")

    chunks = [_record("a", "product a"), _record("b", "product b")]
    result = _pipeline(chunks, Broken()).run("product", "product", ["__public__"], top_k=2, options=PipelineOptions(rerank=True), tenant="default")
    assert len(result.chunks) == 2
    assert result.trace["rerank"] == {"status": "failed", "reason": "no model"}


def test_missing_reranker_is_reported_not_silent():
    result = _pipeline([_record("a", "product")]).run("product", "product", ["__public__"], options=PipelineOptions(rerank=True), tenant="default")
    assert result.trace["rerank"]["status"] == "skipped"


def test_reranker_is_not_called_unless_requested():
    reranker = KeywordReranker()
    _pipeline([_record("a", "product")], reranker).run("product", "product", ["__public__"], tenant="default")
    assert reranker.seen == []


def test_acl_is_enforced_before_the_reranker_sees_anything():
    reranker = KeywordReranker()
    chunks = [_record("secret", "secret price", acl_tags=["finance"]), _record("open", "open price")]
    _pipeline(chunks, reranker).run("price", "price", ["__public__"], options=PipelineOptions(rerank=True), tenant="default")
    assert reranker.seen == ["open price"]


def test_fastembed_adapter_scores_through_its_loader():
    class Model:
        def rerank(self, query: str, texts: List[str], batch_size: int) -> List[float]:
            return [1.0 for _ in texts]

    assert FastEmbedCrossEncoder("m", loader=lambda name: Model()).score("q", ["a", "b"]) == [1.0, 1.0]
    assert FastEmbedCrossEncoder("m", loader=lambda name: Model()).score("q", []) == []


# --- Pipeline with query variants -----------------------------------------------------


def test_rewrites_find_what_the_raw_query_cannot():
    llm = ScriptedLlm({"queries": ["authentication timeout"]})
    chunks = [_record("auth", "Authentication Service Timeout Exceptions")]
    pipeline = RetrievalPipeline(_memory_retriever(chunks), _NoVectors(), transformer=LlmQueryTransformer(llm))
    assert pipeline.run("app crashes", "app crashes", ["__public__"], options=PipelineOptions(fusion="lexical"), tenant="default").chunks == []
    llm = ScriptedLlm({"queries": ["authentication timeout"]})
    pipeline = RetrievalPipeline(_memory_retriever(chunks), _NoVectors(), transformer=LlmQueryTransformer(llm))
    result = pipeline.run("app crashes", "app crashes", ["__public__"], options=PipelineOptions(fusion="lexical", query_mode="multi"), tenant="default")
    assert [c["chunk_id"] for c in result.chunks] == ["auth"]
    assert [variant["kind"] for variant in result.trace["variants"]] == ["original", "rewrite"]


class _NoVectors:
    def embed(self, text: str) -> List[float]:
        raise AssertionError("lexical searches must not embed")


def test_hyde_text_never_reaches_the_lexical_search():
    searched: List[Any] = []

    class Spy:
        def search(self, **kwargs: Any) -> Any:
            searched.append((kwargs["expanded_query"], kwargs["fusion"]))
            return SimpleNamespace(authorized_chunks=[], masked_restricted_docs=[], latency_ms=0, query_dsl={}, rankings={})

    llm = ScriptedLlm({"passage": "invented wording"})
    RetrievalPipeline(Spy(), ConstantEmbedder(), transformer=LlmQueryTransformer(llm)).run(
        "q", "q", ["__public__"], options=PipelineOptions(query_mode="hyde"), tenant="default"
    )
    assert ("invented wording", "dense") in searched


def test_every_variant_uses_the_callers_acl_tags():
    tags: List[Any] = []

    class Spy:
        def search(self, **kwargs: Any) -> Any:
            tags.append(kwargs["effective_acl_tags"])
            return SimpleNamespace(authorized_chunks=[], masked_restricted_docs=[], latency_ms=0, query_dsl={}, rankings={})

    llm = ScriptedLlm({"queries": ["a", "b"]}, {"passage": "c"})
    RetrievalPipeline(Spy(), ConstantEmbedder(), transformer=LlmQueryTransformer(llm)).run(
        "q", "q", ["__public__", "finance"], options=PipelineOptions(query_mode="both"), tenant="default"
    )
    assert len(tags) == 4 and all(value == ["__public__", "finance"] for value in tags)


# --- Multi-hop ------------------------------------------------------------------------


def _retrieved(*items: Dict[str, Any], masked: Any = ()) -> PipelineResult:
    return PipelineResult(list(items), list(masked), 1.0, {}, {}, "")


def _chunk(chunk_id: str, content: str) -> Dict[str, Any]:
    return {"chunk_id": chunk_id, "filename": f"{chunk_id}.md", "content": content}


def test_second_hop_searches_with_the_answer_from_the_first():
    llm = ScriptedLlm(
        {"steps": ["Which product launched in Q3?", "What was {1} revenue in Europe?"]},
        {"found": True, "answer": "Helios"},
    )
    searches: List[str] = []

    def retrieve(text: str) -> PipelineResult:
        searches.append(text)
        return _retrieved(_chunk("launch" if len(searches) == 1 else "revenue", "facts"))

    result = MultiHopPlanner(llm).run("How did the Q3 launch affect European revenue?", retrieve)
    assert searches == ["Which product launched in Q3?", "What was Helios revenue in Europe?"]
    assert [c["chunk_id"] for c in result.chunks] == ["launch", "revenue"]
    assert result.hops[0].answer == "Helios" and not result.stopped_reason


def test_chain_stops_when_a_hop_finds_nothing_instead_of_guessing():
    llm = ScriptedLlm({"steps": ["first?", "second about {1}?"]}, {"found": False, "answer": ""})
    searches: List[str] = []

    def retrieve(text: str) -> PipelineResult:
        searches.append(text)
        return _retrieved(_chunk("x", "unrelated"))

    result = MultiHopPlanner(llm).run("q", retrieve)
    assert searches == ["first?"]
    assert "found no answer" in result.stopped_reason


def test_no_chunks_for_a_hop_makes_no_llm_answer_call():
    llm = ScriptedLlm({"steps": ["first?", "second {1}?"]})
    result = MultiHopPlanner(llm).run("q", lambda text: _retrieved())
    assert len(llm.prompts) == 1 and "found no answer" in result.stopped_reason


def test_planner_failure_falls_back_to_one_search():
    llm = ScriptedLlm(RuntimeError("down"))
    searches: List[str] = []

    def retrieve(text: str) -> PipelineResult:
        searches.append(text)
        return _retrieved(_chunk("x", "text"))

    result = MultiHopPlanner(llm).run("original question", retrieve)
    assert searches == ["original question"] and result.hops[0].found


def test_hops_are_capped():
    llm = ScriptedLlm({"steps": ["a", "b", "c", "d", "e"]})
    assert len(MultiHopPlanner(llm).decompose("q", 3)) == 3


def test_unknown_placeholder_stops_the_chain():
    llm = ScriptedLlm({"steps": ["about {7}?"]})
    searches: List[str] = []
    result = MultiHopPlanner(llm).run("q", lambda text: searches.append(text) or _retrieved())
    assert searches == [] and "isn't available" in result.stopped_reason


def test_restricted_second_hop_is_masked_and_cannot_leak(stack: SimpleNamespace):
    stack.service.ingest_pdf_bytes("launch.md", b"The Q3 launch product was Helios, shipped to all regions.")
    stack.service.ingest_pdf_bytes("revenue.md", b"Helios revenue in Europe was EUR 4,200,000.", tags=["finance"])
    llm = ScriptedLlm(
        {"steps": ["Which product launched in Q3?", "What was {1} revenue in Europe?"]},
        {"found": True, "answer": "Helios"},
    )
    stack.service._multi_hop = MultiHopPlanner(llm)  # noqa: SLF001
    result = stack.service.query_and_chat(
        "How did the Q3 launch affect European revenue?",
        user_roles=[],
        pipeline_options=PipelineOptions(multi_hop=True),
        tenant="default",
    )
    blob = " ".join(c["content"] for c in result["authorized_chunks"]) + result["answer"]
    assert "4,200,000" not in blob
    assert len(result["retrieval_trace"]["hops"]) == 2


def test_multi_hop_chunks_reach_the_answer_with_the_right_acl(stack: SimpleNamespace):
    stack.service.ingest_pdf_bytes("launch.md", b"The Q3 launch product was Helios, shipped to all regions.")
    stack.service.ingest_pdf_bytes("revenue.md", b"Helios revenue in Europe was EUR 4,200,000.", tags=["finance"])
    llm = ScriptedLlm(
        {"steps": ["Which product launched in Q3?", "What was {1} revenue in Europe?"]},
        {"found": True, "answer": "Helios"},
    )
    stack.service._multi_hop = MultiHopPlanner(llm)  # noqa: SLF001
    result = stack.service.query_and_chat(
        "How did the Q3 launch affect European revenue?",
        user_roles=["finance"],
        pipeline_options=PipelineOptions(multi_hop=True),
        tenant="default",
    )
    assert {"launch.md", "revenue.md"} <= {c["filename"] for c in result["authorized_chunks"]}


# --- Against a real Elasticsearch -----------------------------------------------------


@pytest.fixture
def live_index() -> Any:
    manager = ElasticsearchIndexManager(index_name=f"beyondrag-test-{uuid.uuid4().hex[:10]}", dimensions=4)
    if not manager.is_online():
        pytest.skip("Elasticsearch is not reachable")
    manager.ensure_index()
    yield manager
    manager.client.indices.delete(index=manager.index_name, ignore_unavailable=True)


def _index(manager: ElasticsearchIndexManager, *records: Dict[str, Any]) -> None:
    for record in records:
        manager.client.index(index=manager.index_name, id=record["chunk_id"], document=record)
    manager.client.indices.refresh(index=manager.index_name)


def _tokens(manager: ElasticsearchIndexManager, analyzer: str, text: str) -> List[str]:
    result = manager.client.indices.analyze(index=manager.index_name, analyzer=analyzer, text=text)
    return [token["token"] for token in result["tokens"]]


def test_live_standard_analyzer_splits_codes_and_ours_does_not(live_index: Any):
    assert _tokens(live_index, "multilingual", "ACME-X200") == ["acme", "x200"]
    assert _tokens(live_index, "code_preserving", "See ACME-X200, or (ERR-1042).") == ["see", "acme-x200", "or", "err-1042"]


def test_live_sentence_end_does_not_stick_to_the_code(live_index: Any):
    assert _tokens(live_index, "code_preserving", "the code is ERR-1042.") == ["the", "code", "is", "err-1042"]
    assert _tokens(live_index, "code_preserving", "version v2.1.3 ships") == ["version", "v2.1.3", "ships"]


def test_live_exact_code_beats_a_document_with_the_same_parts(live_index: Any):
    _index(
        live_index,
        _record("parts", "ACME sells X200 cables and many ACME adapters for the X200 line", is_current=True, ingestion_status="INDEXED"),
        _record("whole", "ACME-X200 weighs four kilograms", is_current=True, ingestion_status="INDEXED"),
    )
    outcome = HybridRetriever(live_index).search("ACME-X200", [0.5, 0.5, 0.5, 0.5], ["__public__"], top_k=2, fusion="lexical", tenant="default")
    assert outcome.authorized_chunks[0]["chunk_id"] == "whole"


def test_live_rrf_returns_rankings_from_both_searches(live_index: Any):
    _index(live_index, _record("a", "ERR-1042 token expired"), _record("b", "general manual"))
    outcome = HybridRetriever(live_index).search("ERR-1042", [0.5, 0.5, 0.5, 0.5], ["__public__"], top_k=2, tenant="default")
    assert outcome.rankings["lexical"] == ["a"]
    assert set(outcome.rankings["dense"]) == {"a", "b"}
    assert outcome.authorized_chunks[0]["chunk_id"] == "a"


def test_live_acl_applies_to_both_searches_in_rrf(live_index: Any):
    _index(live_index, _record("secret", "ERR-1042 secret", acl_tags=["finance"]), _record("open", "ERR-1042 open"))
    outcome = HybridRetriever(live_index).search("ERR-1042", [0.5, 0.5, 0.5, 0.5], ["__public__"], top_k=5, tenant="default")
    assert [c["chunk_id"] for c in outcome.authorized_chunks] == ["open"]
    assert "secret" not in outcome.rankings["lexical"] + outcome.rankings["dense"]


def test_fast_answer_quotes_the_matching_sentence_of_a_long_parent():
    filler = "This section applies to all regional entities and is reviewed every year. "
    chunk = {"chunk_id": "a", "filename": "a.md", "page_number": 1, "tags": [], "is_public": True, "content": filler * 4 + "Error ERR-1393 is raised by the billing service."}
    answer = GroundedAnswerSynthesizer().synthesize("ERR-1393", "ERR-1393", [], False, [], [chunk], [])
    assert "ERR-1393 is raised" in answer


def test_fast_answer_only_reports_conflicts_about_the_quoted_source():
    def chunk(name: str, sentence: str) -> Dict[str, Any]:
        return {"chunk_id": name, "filename": f"{name}.md", "page_number": 1, "tags": [], "is_public": True, "content": sentence}

    chunks = [
        chunk("quoted", "The Aurora router costs 289 dollars."),
        chunk("other-a", "The ACME widget price list says the widget costs 100 dollars for standard customers today."),
        chunk("other-b", "The ACME widget price list says the widget costs 900 dollars for standard customers today."),
    ]
    answer = GroundedAnswerSynthesizer().synthesize("Aurora router cost", "Aurora router cost", [], False, [], chunks, [])
    assert "quoted.md" in answer and "other-a.md" not in answer
