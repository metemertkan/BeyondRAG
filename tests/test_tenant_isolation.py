"""Episode 5: tenant isolation. Offline by default; the live-Elasticsearch test skips itself.

Two tenants, ``acme`` and ``globex``, share role names on purpose. Every test asks the same
thing in a different place: can one of them see, change, block or learn about the other?
"""

from __future__ import annotations

import base64
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import jwt
import pytest
from fastapi.testclient import TestClient

from src.rag import auth
from src.rag.auth import AuthNotConfiguredError, InvalidTokenError, mint_token, verify_token
from src.rag.chunker import create_chunker
from src.rag.cross_lingual import DynamicConceptRegistry
from src.rag.embeddings import CrossLingualEmbedder, create_embedding_provider
from src.rag.erasure import ErasedSourceError, ErasureService, SourceSyncHandler
from src.rag.es_client import ElasticsearchIndexManager
from src.rag.extractor import DocumentParserRegistry
from src.rag.extractor import ParsedTable
from src.rag.ingestor import PdfIngestor, generate_deterministic_doc_id
from src.rag.pipeline import PipelineOptions
from src.rag.profiles import IndexTenantStrategy
from src.rag.repository import DocumentRepository
from src.rag.retriever import HybridRetriever
from src.rag.seeding import DEMO_TENANT_SAMPLES, DemoDocumentSeeder
from src.rag.service import RBACKnowledgeBaseService
from src.rag.synthesizer import GroundedAnswerSynthesizer
from src.rag.tabular_store import TableQuery, TabularStore
from src.rag.tenancy import (
    Principal,
    TenantNotFoundError,
    TenantScopeError,
    normalize_tenant,
    require_tenant,
    tenant_scoped,
)

ACME_RESERVE = "Treasury must hold a minimum cash reserve of HKD 450,000,000 at all times."
GLOBEX_RESERVE = "Treasury must hold a minimum cash reserve of EUR 38,000,000 at all times."
GLOBEX_SECRET = "Project Nightjar has an acquisition budget of EUR 61,000,000."
QUESTION = "What is the minimum cash reserve treasury must hold?"


# --- Tenant ids ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None, 7, "*", "acme,globex", "acme*", "acme globex", "-acme", "_all", "a" * 33])
def test_a_read_needs_exactly_one_valid_tenant(bad: Any):
    with pytest.raises(TenantScopeError):
        require_tenant(bad)


def test_valid_tenant_ids_are_lowercased_and_kept():
    assert require_tenant(" Acme ") == "acme"
    assert require_tenant("globex-eu_2") == "globex-eu_2"


def test_a_write_without_a_tenant_lands_in_the_default_tenant():
    assert normalize_tenant(None) == normalize_tenant("") == "default"


def test_tombstone_keys_cannot_collide_across_tenants():
    assert tenant_scoped("source", None, "a.pdf") == "source:a.pdf"  # Unchanged for data written before tenants.
    keys = {
        tenant_scoped("source", "default", "acme:a.pdf"),
        tenant_scoped("source", "acme", "a.pdf"),
        tenant_scoped("source", "acm", "ea.pdf"),
        tenant_scoped("content", "acme", "a.pdf"),
    }
    assert len(keys) == 4


def test_a_principal_cannot_be_built_for_all_tenants():
    with pytest.raises(TenantScopeError):
        Principal(tenant="*", roles=("admin",))
    assert Principal(tenant="ACME", roles=(" Finance ", "")).roles == ("finance",)


# --- Signed identity -------------------------------------------------------------------

SECRET = b"unit-test-signing-secret-0123456789abcdef"


def _claims(**overrides: Any) -> Dict[str, Any]:
    now = int(time.time())
    claims = {"sub": "u1", "tenant": "acme", "roles": ["finance"], "aud": auth.AUDIENCE, "iat": now, "exp": now + 600}
    claims.update(overrides)
    return {key: value for key, value in claims.items() if value is not None}


def _b64(data: Dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).rstrip(b"=").decode()


def test_a_minted_token_round_trips_to_a_principal():
    principal = verify_token(mint_token("acme", ["finance"], subject="u1", secret=SECRET), secret=SECRET)
    assert principal == Principal(tenant="acme", roles=("finance",), subject="u1")


def test_token_without_a_signature_is_refused():
    unsigned = f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(_claims(tenant='globex', roles=['admin']))}."
    with pytest.raises(InvalidTokenError):
        verify_token(unsigned, secret=SECRET)


def test_changing_the_tenant_inside_a_token_breaks_it():
    header, _, signature = mint_token("acme", ["finance"], secret=SECRET).split(".")
    forged = f"{header}.{_b64(_claims(tenant='globex'))}.{signature}"
    with pytest.raises(InvalidTokenError):
        verify_token(forged, secret=SECRET)


@pytest.mark.parametrize(
    "claims",
    [
        _claims(exp=int(time.time()) - 5),
        _claims(tenant=None),
        _claims(tenant="*"),
        _claims(tenant="acme,globex"),
        _claims(sub=None),
        _claims(exp=None),
        _claims(aud="another-service"),
        _claims(aud=None),
        _claims(roles="admin"),
    ],
    ids=["expired", "no-tenant", "wildcard-tenant", "two-tenants", "no-subject", "no-expiry", "wrong-audience", "no-audience", "roles-not-a-list"],
)
def test_token_with_bad_claims_is_refused(claims: Dict[str, Any]):
    with pytest.raises(InvalidTokenError):
        verify_token(jwt.encode(claims, SECRET, algorithm="HS256"), secret=SECRET)


def test_token_signed_with_another_key_or_algorithm_is_refused():
    with pytest.raises(InvalidTokenError):
        verify_token(jwt.encode(_claims(), b"x" * 40, algorithm="HS256"), secret=SECRET)
    with pytest.raises(InvalidTokenError):
        verify_token(jwt.encode(_claims(), SECRET + b"0" * 32, algorithm="HS512"), secret=SECRET)


def test_required_mode_has_no_fallback_secret(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RAG_AUTH_MODE", "required")
    monkeypatch.delenv("RAG_AUTH_SECRET", raising=False)
    with pytest.raises(AuthNotConfiguredError):
        auth.load_auth_secret(tmp_path / ".auth_secret")
    assert not (tmp_path / ".auth_secret").exists()


def test_a_short_secret_is_refused(monkeypatch):
    monkeypatch.setenv("RAG_AUTH_SECRET", "too-short")
    with pytest.raises(AuthNotConfiguredError):
        auth.load_auth_secret()


def test_demo_mode_generates_a_private_secret_once(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RAG_AUTH_MODE", "demo")
    monkeypatch.delenv("RAG_AUTH_SECRET", raising=False)
    path = tmp_path / ".auth_secret"
    first = auth.load_auth_secret(path)
    assert len(first) >= auth.MIN_SECRET_BYTES and auth.load_auth_secret(path) == first
    assert path.stat().st_mode & 0o077 == 0


# --- A two-tenant stack on the in-memory store -----------------------------------------


class _NoConcepts:
    def is_cross_lingual(self, text: str) -> bool:
        return False

    def extract_concepts_at_ingestion(self, filename: str, full_text: str, tenant: str | None = None) -> Any:
        return [], "none"

    def prepare_query(self, question: str, allow_llm_fallback: bool = False, tenant: str | None = None) -> Any:
        return question, []


def _build(tmp_path: Path, manager: ElasticsearchIndexManager, embedder: CrossLingualEmbedder) -> SimpleNamespace:
    repository = DocumentRepository(
        es_manager=manager,
        uploads_dir=tmp_path / "uploads",
        registry_path=tmp_path / "registry.json",
        tabular_store=TabularStore(),
    )
    ingestor = PdfIngestor(
        extractor=DocumentParserRegistry(),
        chunker=create_chunker("sentence", max_tokens=100, overlap_tokens=10),
        cross_lingual=_NoConcepts(),
        embedder=embedder,
        repository=repository,
    )
    retriever = HybridRetriever(manager)
    service = RBACKnowledgeBaseService(
        index_manager=manager,
        repository=repository,
        ingestor=ingestor,
        cross_lingual=_NoConcepts(),
        embedder=embedder,
        retriever=retriever,
        synthesizer=GroundedAnswerSynthesizer(),
        seeder=object(),  # type: ignore[arg-type]
        reranker=None,
    )
    return SimpleNamespace(service=service, manager=manager, repository=repository, retriever=retriever, embedder=embedder)


@pytest.fixture
def stack(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    provider = create_embedding_provider("feature_hash")
    manager = ElasticsearchIndexManager(es_url="http://non-existent-es:9200", dimensions=provider.dimensions)
    built = _build(tmp_path, manager, CrossLingualEmbedder(provider))
    # Offboarding clears the concept registry. Point it at a file of this test's own.
    concepts = DynamicConceptRegistry(tmp_path / "concepts.json")
    monkeypatch.setattr("src.rag.cross_lingual.get_concept_registry", lambda: concepts)
    built.concepts = concepts
    return built


@pytest.fixture
def two_tenants(stack: SimpleNamespace) -> SimpleNamespace:
    """Same file names, same role names, different companies."""
    ingest = stack.service.ingest_pdf_bytes
    stack.acme_public = ingest("handbook.md", b"Acme staff work from the Hong Kong office.", tenant="acme")
    stack.acme_finance = ingest("treasury.md", ACME_RESERVE.encode(), tags=["finance"], tenant="acme")
    stack.globex_public = ingest("handbook.md", b"Globex staff work from the Rotterdam office.", tenant="globex")
    stack.globex_finance = ingest(
        "treasury.md", f"{GLOBEX_RESERVE} {GLOBEX_SECRET}".encode(), tags=["finance"], tenant="globex"
    )
    return stack


def _ask(stack: SimpleNamespace, tenant: str, roles: List[str], question: str = QUESTION) -> Dict[str, Any]:
    return stack.service.query_and_chat(question, user_roles=roles, pipeline_options=PipelineOptions(), tenant=tenant)


def _texts(result: Dict[str, Any]) -> str:
    return " ".join(chunk["content"] for chunk in result["authorized_chunks"])


# --- Retrieval -------------------------------------------------------------------------


def test_same_question_and_role_give_each_tenant_its_own_answer(two_tenants: SimpleNamespace):
    acme, globex = _ask(two_tenants, "acme", ["finance"]), _ask(two_tenants, "globex", ["finance"])
    assert "HKD 450,000,000" in _texts(acme) and "EUR" not in _texts(acme)
    assert "EUR 38,000,000" in _texts(globex) and "HKD" not in _texts(globex)
    assert {acme["tenant"], globex["tenant"]} == {"acme", "globex"}


def test_no_role_reaches_across_tenants(two_tenants: SimpleNamespace):
    for roles in (["finance"], ["admin"], ["super_admin"], ["finance", "operation", "admin", "super_admin"]):
        result = _ask(two_tenants, "acme", roles, "Project Nightjar acquisition budget")
        assert "Nightjar" not in _texts(result)
        assert all(chunk["filename"] in {"handbook.md", "treasury.md"} for chunk in result["authorized_chunks"])
        assert "Nightjar" not in json.dumps(result["masked_restricted_docs"])


def test_a_tenant_with_no_documents_gets_nothing(two_tenants: SimpleNamespace):
    assert _ask(two_tenants, "initech", ["super_admin"])["authorized_chunks"] == []


def test_every_stored_chunk_carries_its_tenant(two_tenants: SimpleNamespace):
    assert {record["tenant"] for record in two_tenants.manager.memory_chunks} == {"acme", "globex"}


def test_search_without_a_tenant_does_not_run(two_tenants: SimpleNamespace):
    vector = two_tenants.embedder.embed(QUESTION)
    with pytest.raises(TypeError):
        two_tenants.retriever.search(QUESTION, vector, ["__public__", "finance"])  # type: ignore[call-arg]
    for bad in ("", "*", "acme,globex", "acme*"):
        with pytest.raises(TenantScopeError):
            two_tenants.retriever.search(QUESTION, vector, ["__public__", "finance"], tenant=bad)
        with pytest.raises(TenantScopeError):
            _ask(two_tenants, bad, ["finance"])
    with pytest.raises(TypeError):
        two_tenants.service.query_and_chat(QUESTION, user_roles=["finance"])  # type: ignore[call-arg]


def test_tenant_filter_is_in_both_halves_of_the_hybrid_query(two_tenants: SimpleNamespace):
    outcome = two_tenants.retriever.search(QUESTION, two_tenants.embedder.embed(QUESTION), ["__public__"], tenant="acme")
    term = {"term": {"tenant": "acme"}}
    assert term in outcome.query_dsl["knn"]["filter"]
    assert term in outcome.query_dsl["query"]["bool"]["filter"]


def test_silo_mode_gives_each_tenant_a_concrete_index_and_keeps_the_filter(tmp_path: Path):
    provider = create_embedding_provider("feature_hash")
    manager = ElasticsearchIndexManager(
        es_url="http://non-existent-es:9200",
        index_name="kb",
        tenant_strategy=IndexTenantStrategy(prefix="kb", tenant_mode="per_tenant"),
        dimensions=provider.dimensions,
    )
    silo = _build(tmp_path, manager, CrossLingualEmbedder(provider))
    assert (manager.index_for("acme"), manager.index_for("globex")) == ("kb-acme", "kb-globex")
    silo.service.ingest_pdf_bytes("treasury.md", ACME_RESERVE.encode(), tags=["finance"], tenant="acme")
    silo.service.ingest_pdf_bytes("treasury.md", GLOBEX_RESERVE.encode(), tags=["finance"], tenant="globex")
    outcome = silo.retriever.search(QUESTION, silo.embedder.embed(QUESTION), ["finance"], tenant="globex")
    assert outcome.query_dsl["index"] == "kb-globex" and "*" not in outcome.query_dsl["index"]
    assert {"term": {"tenant": "globex"}} in outcome.query_dsl["knn"]["filter"]
    assert [c["content"] for c in outcome.authorized_chunks] == [GLOBEX_RESERVE]
    with pytest.raises(ValueError):
        ElasticsearchIndexManager(es_url="http://non-existent-es:9200", dimensions=4).drop_tenant_index("acme")


# --- Documents: identity, listing, change ----------------------------------------------


def test_same_file_name_in_two_tenants_does_not_supersede(two_tenants: SimpleNamespace):
    for tenant in ("acme", "globex"):
        docs = two_tenants.service.list_documents(tenant=tenant)
        assert sorted(d["filename"] for d in docs) == ["handbook.md", "treasury.md"]
        assert all(d.get("is_current", True) and d.get("status") != "SUPERSEDED" for d in docs)


def test_same_bytes_in_two_tenants_get_different_document_ids():
    body = b"identical bytes"
    ids = {generate_deterministic_doc_id("a.md", body, tenant=t) for t in (None, "acme", "globex")}
    assert len(ids) == 3
    # A default-tenant file named like another tenant's prefix must not collide with it.
    assert generate_deterministic_doc_id("acme-a.md", body) != generate_deterministic_doc_id("a.md", body, tenant="acme")


def test_listing_shows_one_tenant_only(two_tenants: SimpleNamespace):
    acme_ids = {d["doc_id"] for d in two_tenants.service.list_documents(tenant="acme")}
    globex_ids = {d["doc_id"] for d in two_tenants.service.list_documents(tenant="globex")}
    assert acme_ids == {two_tenants.acme_public["doc_id"], two_tenants.acme_finance["doc_id"]}
    assert not acme_ids & globex_ids
    assert two_tenants.service.list_documents(tenant="initech") == []
    # A tag invented by one tenant is not offered to another; only the built-in roles are shared.
    two_tenants.service.ingest_pdf_bytes("deal.md", b"Nightjar deal room notes.", tags=["nightjar-deal-team"], tenant="globex")
    assert "nightjar-deal-team" in two_tenants.service.list_available_tags(tenant="globex")
    assert "nightjar-deal-team" not in two_tenants.service.list_available_tags(tenant="acme")
    assert two_tenants.service.list_available_tags(tenant="initech") == ["finance", "operation"]


def test_another_tenants_document_looks_exactly_like_a_missing_one(two_tenants: SimpleNamespace):
    foreign, missing = two_tenants.acme_finance["doc_id"], "doc-does-not-exist"
    for doc_id in (foreign, missing):
        with pytest.raises(TenantNotFoundError):
            two_tenants.service.delete_document(doc_id, tenant="globex")
        with pytest.raises(TenantNotFoundError):
            two_tenants.service.update_document_tags(doc_id, ["__public__"], tenant="globex")
    # Nothing changed for the owner.
    assert "HKD 450,000,000" in _texts(_ask(two_tenants, "acme", ["finance"]))
    assert "HKD 450,000,000" not in _texts(_ask(two_tenants, "acme", []))
    with pytest.raises(TenantNotFoundError):
        two_tenants.repository.delete_document(foreign, tenant="globex")


def test_a_chosen_document_id_cannot_point_at_another_tenant(two_tenants: SimpleNamespace):
    with pytest.raises(ValueError, match="not available"):
        two_tenants.service.ingest_pdf_bytes(
            "treasury.md", b"overwrite attempt", doc_id=two_tenants.acme_finance["doc_id"], tenant="globex"
        )
    assert "HKD 450,000,000" in _texts(_ask(two_tenants, "acme", ["finance"]))


def test_a_source_key_only_names_a_document_inside_the_callers_tenant(two_tenants: SimpleNamespace):
    acme_key = two_tenants.acme_finance["source_key"]
    two_tenants.service.ingest_pdf_bytes(
        "takeover.md", b"Globex wrote this with Acme's source key.", tags=["finance"], tenant="globex", source_key=acme_key
    )
    acme_docs = two_tenants.service.list_documents(tenant="acme")
    assert all(d.get("is_current", True) and d.get("status") != "SUPERSEDED" for d in acme_docs)
    assert "HKD 450,000,000" in _texts(_ask(two_tenants, "acme", ["finance"]))


# --- Tables ----------------------------------------------------------------------------


def _table(amount: int) -> ParsedTable:
    return ParsedTable(table_id="tbl_budget", name="Budget", columns=["region", "amount"], rows=[["north", amount]])


def test_same_table_id_in_two_tenants_holds_separate_rows():
    store = TabularStore()
    store.save_table(_table(100), doc_id="d-acme", source_key="acme:b.xlsx", acl_tags=["finance"], tenant="acme")
    store.save_table(_table(999), doc_id="d-globex", source_key="globex:b.xlsx", acl_tags=["finance"], tenant="globex")
    query = TableQuery(table_id="tbl_budget")
    assert store.query(query, ["finance"], tenant="acme")["rows"] == [["north", 100]]
    assert store.query(query, ["finance"], tenant="globex")["rows"] == [["north", 999]]


def test_another_tenants_table_is_an_unknown_table():
    store = TabularStore()
    store.save_table(_table(100), doc_id="d-acme", source_key="acme:b.xlsx", acl_tags=["finance"], tenant="acme")
    for tenant in ("globex", None):
        with pytest.raises(KeyError):
            store.query(TableQuery(table_id="tbl_budget"), ["finance", "super_admin"], tenant=tenant)
    assert store.delete_tenant("globex") == 0
    assert store.delete_tenant("acme") == 1 and store.describe("tbl_budget", tenant="acme") is None


# --- Tombstones, events, erasure -------------------------------------------------------


def _erasure(stack: SimpleNamespace, tmp_path: Path) -> ErasureService:
    return ErasureService(stack.repository, concept_registry=stack.concepts, audit_log_path=tmp_path / "audit.jsonl")


def test_erasing_a_source_in_one_tenant_leaves_the_same_source_in_the_other(two_tenants: SimpleNamespace, tmp_path: Path):
    erasure = _erasure(two_tenants, tmp_path)
    plan = erasure.plan_source(two_tenants.acme_finance["source_key"], tenant="acme")
    receipt = erasure.execute(plan)
    assert receipt.verified
    assert "HKD" not in _texts(_ask(two_tenants, "acme", ["finance"]))
    assert "EUR 38,000,000" in _texts(_ask(two_tenants, "globex", ["finance"]))


def test_one_tenant_cannot_plan_an_erasure_of_another_tenants_source(two_tenants: SimpleNamespace, tmp_path: Path):
    erasure = _erasure(two_tenants, tmp_path)
    plan = erasure.plan_source(two_tenants.acme_finance["source_key"], tenant="globex")
    assert plan.versions == [] and plan.chunk_count == 0 and plan.tenant == "globex"
    erasure.execute(plan)
    assert "HKD 450,000,000" in _texts(_ask(two_tenants, "acme", ["finance"]))


def test_a_tombstone_blocks_only_the_tenant_that_erased(two_tenants: SimpleNamespace, tmp_path: Path):
    erasure = _erasure(two_tenants, tmp_path)
    erasure.execute(erasure.plan_source(two_tenants.acme_finance["source_key"], tenant="acme"))
    with pytest.raises(ErasedSourceError):
        two_tenants.service.ingest_pdf_bytes("treasury.md", ACME_RESERVE.encode(), tags=["finance"], tenant="acme")
    # The same bytes under the same file name in another tenant: no block, and no hint that Acme ever held them.
    report = two_tenants.service.ingest_pdf_bytes("treasury_copy.md", ACME_RESERVE.encode(), tags=["finance"], tenant="globex")
    assert report["doc_id"]


def test_two_tenants_can_send_the_same_event_id(two_tenants: SimpleNamespace, tmp_path: Path):
    handler = SourceSyncHandler(two_tenants.repository, _erasure(two_tenants, tmp_path))
    first = handler.handle(
        event_id="evt-1", event_type="deleted", source_key=two_tenants.acme_public["source_key"], tenant="acme"
    )
    second = handler.handle(
        event_id="evt-1", event_type="deleted", source_key=two_tenants.globex_public["source_key"], tenant="globex"
    )
    assert first["status"] != "duplicate" and second["status"] != "duplicate"
    assert two_tenants.service.list_documents(tenant="acme")[0]["filename"] == "treasury.md"
    assert [d["filename"] for d in two_tenants.service.list_documents(tenant="globex")] == ["treasury.md"]


# --- Concept registry and query cache --------------------------------------------------


def test_concepts_and_cached_queries_stay_inside_their_tenant(tmp_path: Path):
    registry = DynamicConceptRegistry(tmp_path / "concepts.json")
    concept = {"concept_id": "nightjar", "zh_triggers": ["夜鹰"], "en_triggers": ["nightjar"], "expansion": "Project Nightjar acquisition"}
    registry.register_document_concepts("hash-1", "treasury.pdf", [concept], "test-model", tenant="globex")
    registry.cache_query_translation("夜鹰预算", "nightjar budget", tenant="globex")

    assert registry.get_document_concepts("hash-1", tenant="acme") is None
    assert registry.get_document_concepts("hash-1") is None
    assert registry.get_cached_query_translation("夜鹰预算", tenant="acme") is None
    assert registry.count_mentions("nightjar", tenant="acme") == 0
    assert registry.count_mentions("nightjar", tenant="globex") == 2

    reloaded = DynamicConceptRegistry(tmp_path / "concepts.json")
    assert reloaded.get_document_concepts("hash-1", tenant="globex") == [concept]
    assert reloaded.forget_tenant("globex") == 2
    assert "nightjar" not in (tmp_path / "concepts.json").read_text(encoding="utf-8").lower()


# --- Offboarding -----------------------------------------------------------------------


def test_offboarding_removes_one_tenant_and_nothing_else(two_tenants: SimpleNamespace):
    two_tenants.repository.tabular_store.save_table(
        _table(7), doc_id=two_tenants.globex_finance["doc_id"], source_key="globex:b.xlsx", acl_tags=["finance"], tenant="globex"
    )
    two_tenants.concepts.cache_query_translation("q", "nightjar", tenant="globex")
    receipt = two_tenants.service.offboard_tenant("globex")
    assert receipt["tenant"] == "globex" and receipt["documents"] == 2 and receipt["chunks"] >= 2
    assert receipt["tables"] == 1 and receipt["concept_entries"] == 1 and receipt["upload_files"] == 2

    assert two_tenants.service.list_documents(tenant="globex") == []
    assert _ask(two_tenants, "globex", ["super_admin"])["authorized_chunks"] == []
    assert {record["tenant"] for record in two_tenants.manager.memory_chunks} == {"acme"}
    assert "HKD 450,000,000" in _texts(_ask(two_tenants, "acme", ["finance"]))
    assert len(two_tenants.service.list_documents(tenant="acme")) == 2
    with pytest.raises(TenantScopeError):
        two_tenants.service.offboard_tenant("*")


def test_registry_reloads_each_document_into_its_tenant(two_tenants: SimpleNamespace, tmp_path: Path):
    reloaded = DocumentRepository(
        es_manager=two_tenants.manager, uploads_dir=tmp_path / "uploads", registry_path=tmp_path / "registry.json"
    )
    assert reloaded.tenant_of(two_tenants.acme_finance["doc_id"]) == "acme"
    assert reloaded.tenant_of(two_tenants.globex_finance["doc_id"]) == "globex"
    assert reloaded.get_current_doc_id(two_tenants.acme_finance["source_key"], "globex") is None


def test_a_demo_restart_reseeds_the_samples_and_forgets_what_the_reset_removed(stack: SimpleNamespace, tmp_path: Path):
    """Demo mode empties the index at every start. The registry on disk has to go with it.

    It used to survive, so after the first start the seeder found every sample file "already
    loaded", skipped it, and the app came up with documents in the list and nothing to search.
    """
    samples = sum(len(files) for files in DEMO_TENANT_SAMPLES.values())
    stack.service._seeder = DemoDocumentSeeder()
    assert len(stack.service.reindex_all_sample_pdfs()) == samples
    upload = stack.service.ingest_pdf_bytes("note.md", b"An Acme note uploaded after the start.", tenant="acme")
    erasure = _erasure(stack, tmp_path)
    erasure.execute(erasure.plan_source(upload["source_key"], tenant="acme"))
    kept = stack.service.ingest_pdf_bytes("memo.md", b"An Acme memo uploaded after the start.", tenant="acme")

    # A restart: a new process reads the registry from disk, then resets the index and seeds.
    restarted = _build(tmp_path, stack.manager, stack.embedder)
    restarted.service._seeder = DemoDocumentSeeder()
    assert restarted.repository.get_document_state(kept["doc_id"]) is not None
    assert len(restarted.service.reindex_all_sample_pdfs()) == samples

    assert {record["tenant"] for record in restarted.manager.memory_chunks} == set(DEMO_TENANT_SAMPLES)
    assert "EUR 61,000,000" in _texts(_ask(restarted, "globex", ["finance"], "Project Nightjar acquisition budget"))
    assert "EUR 61,000,000" not in _texts(_ask(restarted, "acme", ["finance"], "Project Nightjar acquisition budget"))
    # The memo's chunks went with the index, so it is not listed and its saved copy is gone.
    assert "memo.md" not in {doc["filename"] for doc in restarted.service.list_documents(tenant="acme")}
    assert not list((tmp_path / "uploads").glob(f"{kept['doc_id']}_*"))
    # An erasure outlives the reset: the erased source still cannot come back.
    with pytest.raises(ErasedSourceError):
        restarted.service.ingest_pdf_bytes("note.md", b"An Acme note uploaded after the start.", tenant="acme")


# --- The HTTP API ----------------------------------------------------------------------


@pytest.fixture
def api(two_tenants: SimpleNamespace, monkeypatch) -> Iterator[TestClient]:
    from src.web.app import app

    # "src.web.app" resolves to the FastAPI object re-exported by the package.
    monkeypatch.setattr(sys.modules["src.web.app"], "get_kb_service", lambda: two_tenants.service)
    monkeypatch.setenv("RAG_AUTH_MODE", "demo")
    yield TestClient(app)


def _bearer(tenant: str, *roles: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(tenant, roles)}"}


def _chat(api: TestClient, headers: Dict[str, str], question: str = QUESTION, **extra: Any) -> Dict[str, Any]:
    response = api.post("/api/chat", json={"question": question, **extra}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/api/chat"),
        ("get", "/api/documents"),
        ("post", "/api/upload"),
        ("put", "/api/documents/doc-x/tags"),
        ("delete", "/api/documents/doc-x"),
        ("post", "/api/seed"),
        ("post", "/api/seed-lab"),
    ],
)
def test_data_endpoints_refuse_a_request_without_a_valid_token(api: TestClient, method: str, path: str):
    expired = mint_token("acme", ["admin"], ttl_seconds=-5)
    for headers in ({}, {"Authorization": "Bearer not-a-token"}, {"Authorization": f"Basic {expired}"}, {"Authorization": f"Bearer {expired}"}):
        response = getattr(api, method)(path, headers=headers)
        assert response.status_code == 401
        assert response.json() == {"detail": "A valid bearer token is required."}


def test_roles_and_tenant_typed_into_the_request_are_ignored(api: TestClient):
    body = _chat(
        api, _bearer("acme"), user_roles=["finance", "admin", "super_admin"], tenant="globex", roles=["finance"]
    )
    assert body["tenant"] == "acme" and body["user_roles"] == []
    assert "450,000,000" not in _texts(body) and "EUR" not in _texts(body)
    spoofed = api.post(
        "/api/chat",
        json={"question": QUESTION},
        headers={**_bearer("acme", "finance"), "X-Tenant": "globex", "X-Tenant-Id": "globex"},
    ).json()
    assert spoofed["tenant"] == "acme" and "HKD 450,000,000" in _texts(spoofed)


def test_the_token_decides_which_company_answers(api: TestClient):
    assert "HKD 450,000,000" in _texts(_chat(api, _bearer("acme", "finance")))
    assert "EUR 38,000,000" in _texts(_chat(api, _bearer("globex", "finance")))
    listed = api.get("/api/documents", headers=_bearer("globex")).json()
    assert listed["tenant"] == "globex" and len(listed["documents"]) == 2
    assert all("Acme" not in json.dumps(doc) for doc in listed["documents"])


def test_changing_documents_needs_a_write_role(api: TestClient, two_tenants: SimpleNamespace):
    doc_id = two_tenants.acme_finance["doc_id"]
    for roles in ((), ("finance",), ("operation",)):
        headers = _bearer("acme", *roles)
        assert api.put(f"/api/documents/{doc_id}/tags", json={"tags": ["__public__"]}, headers=headers).status_code == 403
        assert api.delete(f"/api/documents/{doc_id}", headers=headers).status_code == 403
        upload = api.post("/api/upload", files={"file": ("x.md", b"text", "text/markdown")}, headers=headers)
        assert upload.status_code == 403
    # Still restricted: the member could not open it up for themselves.
    assert "450,000,000" not in _texts(_chat(api, _bearer("acme")))
    assert api.put(f"/api/documents/{doc_id}/tags", json={"tags": ["finance"]}, headers=_bearer("acme", "admin")).status_code == 200


def test_an_admin_of_one_tenant_is_nobody_in_another(api: TestClient, two_tenants: SimpleNamespace):
    foreign = two_tenants.acme_finance["doc_id"]
    admin = _bearer("globex", "admin", "super_admin")
    answers = [
        api.delete(f"/api/documents/{doc_id}", headers=admin) for doc_id in (foreign, "doc-does-not-exist")
    ] + [
        api.put(f"/api/documents/{doc_id}/tags", json={"tags": ["__public__"]}, headers=admin)
        for doc_id in (foreign, "doc-does-not-exist")
    ]
    assert {r.status_code for r in answers} == {404}
    assert len({r.text for r in answers}) == 1  # Byte-identical: no way to tell "exists elsewhere" from "never existed".
    assert "HKD 450,000,000" in _texts(_chat(api, _bearer("acme", "finance")))
    assert "450,000,000" not in _texts(_chat(api, _bearer("acme")))


def test_an_upload_lands_in_the_token_tenant_whatever_the_form_says(api: TestClient, two_tenants: SimpleNamespace):
    response = api.post(
        "/api/upload",
        files={"file": ("memo.md", b"Acme memo about the canteen menu.", "text/markdown")},
        data={"tenant": "globex", "tags": ""},
        headers=_bearer("acme", "admin"),
    )
    assert response.status_code == 200, response.text
    doc_id = response.json()["document"]["doc_id"]
    assert two_tenants.repository.tenant_of(doc_id) == "acme"
    assert "memo.md" not in json.dumps(api.get("/api/documents", headers=_bearer("globex")).json())


def test_health_says_nothing_about_the_data(api: TestClient):
    body = api.get("/api/health").json()
    assert set(body) == {"status", "elasticsearch_online", "auth_mode"}


def test_demo_token_exists_only_in_demo_mode(api: TestClient, monkeypatch):
    minted = api.post("/api/auth/demo-token", json={"tenant": "acme", "roles": ["admin"]}).json()
    assert minted["tenant"] == "acme" and minted["can_write"] is True
    assert set(minted["expanded_roles"]) >= {"admin", "finance", "operation"}
    assert verify_token(minted["token"]).tenant == "acme"
    assert api.post("/api/auth/demo-token", json={"tenant": "*"}).status_code == 422
    assert api.post("/api/auth/demo-token", json={"tenant": "acme", "roles": []}).json()["can_write"] is False

    monkeypatch.setenv("RAG_AUTH_MODE", "required")
    assert api.post("/api/auth/demo-token", json={"tenant": "acme", "roles": ["admin"]}).status_code == 404
    assert api.get("/api/auth/config").json() == {"mode": "required", "tenants": [], "roles": []}
    assert api.post("/api/seed", headers=_bearer("acme", "admin")).status_code == 404
    assert api.post("/api/seed-lab", headers=_bearer("acme", "admin")).status_code == 404
    # A token that was valid before the switch still works: only minting is gone.
    assert "HKD 450,000,000" in _texts(_chat(api, {"Authorization": f"Bearer {minted['token']}"}))


def test_no_secret_means_no_access_not_open_access(api: TestClient, monkeypatch):
    token_headers = _bearer("acme", "finance")
    monkeypatch.setenv("RAG_AUTH_MODE", "required")
    monkeypatch.delenv("RAG_AUTH_SECRET", raising=False)
    assert api.post("/api/chat", json={"question": QUESTION}, headers=token_headers).status_code == 503


def test_operator_endpoints_need_an_explicit_single_tenant(api: TestClient, monkeypatch):
    monkeypatch.setenv("RAG_GOVERNANCE_TOKEN", "right")
    headers = {"X-Governance-Token": "right"}
    assert api.post("/api/erasure", json={"source_key": "k"}, headers=headers).status_code == 422
    assert api.post("/api/erasure", json={"tenant": "*", "source_key": "k"}, headers=headers).status_code == 422
    event = {"event_id": "e", "event_type": "deleted", "source_key": "k"}
    assert api.post("/api/sync/events", json=event, headers=headers).status_code == 422
    assert api.post("/api/sync/events", json={**event, "tenant": "acme,globex"}, headers=headers).status_code == 422
    # A tenant's own token is not an operator credential.
    assert api.post("/api/erasure", json={"tenant": "acme", "source_key": "k"}, headers=_bearer("acme", "super_admin")).status_code == 401


def test_offboarding_endpoint_is_operator_only_and_asks_twice(api: TestClient, two_tenants: SimpleNamespace, monkeypatch):
    monkeypatch.delenv("RAG_GOVERNANCE_TOKEN", raising=False)
    assert api.post("/api/tenants/globex/offboard", json={"confirm": "globex"}).status_code == 503
    monkeypatch.setenv("RAG_GOVERNANCE_TOKEN", "right")
    headers = {"X-Governance-Token": "right"}
    assert api.post("/api/tenants/globex/offboard", json={"confirm": "globex"}, headers=_bearer("globex", "admin")).status_code == 401
    assert api.post("/api/tenants/globex/offboard", json={"confirm": "acme"}, headers=headers).status_code == 422
    assert api.post("/api/tenants/*/offboard", json={"confirm": "*"}, headers=headers).status_code == 422
    assert len(two_tenants.service.list_documents(tenant="globex")) == 2

    done = api.post("/api/tenants/globex/offboard", json={"confirm": "globex"}, headers=headers)
    assert done.status_code == 200 and done.json()["receipt"]["documents"] == 2
    assert api.get("/api/documents", headers=_bearer("globex")).json()["documents"] == []
    assert len(api.get("/api/documents", headers=_bearer("acme")).json()["documents"]) == 2


def test_browsers_on_other_sites_get_no_cors_grant_by_default(api: TestClient):
    response = api.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {key.lower() for key in response.headers}


# --- Live Elasticsearch ----------------------------------------------------------------


@pytest.fixture
def live(tmp_path: Path) -> Iterator[SimpleNamespace]:
    provider = create_embedding_provider("feature_hash")
    manager = ElasticsearchIndexManager(index_name=f"beyondrag-test-{uuid.uuid4().hex[:10]}", dimensions=provider.dimensions)
    if not manager.is_online():
        pytest.skip("Elasticsearch is not reachable")
    built = _build(tmp_path, manager, CrossLingualEmbedder(provider))
    yield built
    manager.client.indices.delete(index=manager.index_name, ignore_unavailable=True)


def test_live_pool_filters_search_audit_listing_and_offboarding_by_tenant(live: SimpleNamespace):
    ingest = live.service.ingest_pdf_bytes
    ingest("treasury.md", ACME_RESERVE.encode(), tags=["finance"], tenant="acme")
    ingest("treasury.md", f"{GLOBEX_RESERVE} {GLOBEX_SECRET}".encode(), tags=["finance"], tenant="globex")
    live.manager.client.indices.refresh(index=live.manager.index_name)

    acme = _ask(live, "acme", ["finance"])
    assert "HKD 450,000,000" in _texts(acme) and "EUR" not in _texts(acme)
    # No finance role: the audit names the caller's own hidden document, never the other tenant's.
    public = _ask(live, "acme", [], "Nightjar minimum cash reserve treasury")
    assert public["authorized_chunks"] == []
    assert len(public["masked_restricted_docs"]) == 1
    assert [d["filename"] for d in live.service.list_documents(tenant="globex")] == ["treasury.md"]

    receipt = live.repository.purge_tenant("globex")
    assert receipt["method"] == "delete_by_query" and receipt["chunks"] >= 1
    assert _ask(live, "globex", ["finance"])["authorized_chunks"] == []
    assert "HKD 450,000,000" in _texts(_ask(live, "acme", ["finance"]))
