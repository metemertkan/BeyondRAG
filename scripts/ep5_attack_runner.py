#!/usr/bin/env python3
"""Episode 5: run the same cross-tenant attacks against one checkout of the code.

``ep5_benchmark.py`` starts this file once per configuration, with the working directory set
to a throwaway ``git worktree``. It imports whatever ``src`` it finds there, so the same
attack list runs against the code before Episode 5 and the code after it.

Three configurations:

``main``     The code before Episode 5, as it is. Callers send ``user_roles`` in the request
             body and a ``tenant`` form field on upload, and the API believes them.
``gateway``  The same code behind an imagined API gateway that authenticates the caller and
             overwrites ``user_roles`` and ``tenant`` with the truth, and only lets admins
             write. This is the usual first answer to "the API trusts the request body".
             The gateway is simulated here, in ``Transport``.
``branch``   The Episode 5 code with ``RAG_AUTH_MODE=required``. Identity comes from a signed
             token. The attacker still sends the same lies in the body, the form and headers.

Every attack starts from an empty index and an empty workspace, loads the victim's documents
through the API, checks that the victim can read them (the control), attacks, and reports a
breach only when a canary string from another tenant's document shows up, or when the
victim's own data changed.

It deletes files under ``./workspace``, so it refuses to run outside a worktree that the
benchmark marked as a sandbox.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
CWD = Path.cwd()
SANDBOX_MARKER = CWD / ".ep5_attack_sandbox"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CWD))

import ep5_fixtures as fx  # noqa: E402

GOVERNANCE_TOKEN = "ep5-benchmark-governance-token"
EVIL_ORIGIN = "https://evil.example"
LIVENESS_KEYS = {"status", "elasticsearch_online", "auth_mode"}


@dataclass(frozen=True)
class Who:
    """A person, and what is true about them. What they *claim* is a separate argument."""

    name: str
    tenant: str
    roles: Tuple[str, ...] = ()


VICTIM_ADMIN = Who("gina@globex", fx.VICTIM, ("admin",))
VICTIM_FINANCE = Who("femke@globex", fx.VICTIM, ("finance",))
ATTACKER_NOBODY = Who("nico@acme", fx.ATTACKER, ())
ATTACKER_FINANCE = Who("fiona@acme", fx.ATTACKER, ("finance",))
ATTACKER_ADMIN = Who("adam@acme", fx.ATTACKER, ("admin",))
ANONYMOUS: Optional[Who] = None

Reply = Tuple[int, Dict[str, Any]]


def _json(response: Any) -> Dict[str, Any]:
    try:
        body = response.json()
    except Exception:
        return {"raw": response.text[:300]}
    return body if isinstance(body, dict) else {"body": body}


class Transport:
    """How a request reaches the API in each configuration."""

    def __init__(self, config: str, client: Any, mint: Optional[Callable[..., str]]) -> None:
        self.config = config
        self.client = client
        self._mint = mint

    # -- identity ---------------------------------------------------------------------
    def _headers(self, who: Optional[Who], headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        merged = dict(headers or {})
        if self.config == "branch" and who is not None and "Authorization" not in merged:
            merged["Authorization"] = "Bearer " + self._mint(who.tenant, list(who.roles), subject=who.name)
        return merged

    def _gateway_refuses(self, who: Optional[Who], *, write: bool = False) -> Optional[Reply]:
        if self.config != "gateway":
            return None
        if who is None:
            return 401, {"detail": "gateway: authentication required"}
        if write and "admin" not in who.roles:
            return 403, {"detail": "gateway: writes need the admin role"}
        return None

    # -- endpoints --------------------------------------------------------------------
    def chat(
        self,
        who: Optional[Who],
        question: str,
        *,
        lie: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Reply:
        refused = self._gateway_refuses(who)
        if refused:
            return refused
        body: Dict[str, Any] = {"question": question, "user_roles": list(who.roles) if who else []}
        body.update(lie or {})
        if self.config == "gateway":
            body["user_roles"] = list(who.roles)  # the gateway overwrites the claim
        response = self.client.post("/api/chat", json=body, headers=self._headers(who, headers), params=params)
        return response.status_code, _json(response)

    def upload(
        self,
        who: Optional[Who],
        name: str,
        text: str,
        tags: str = "",
        *,
        lie: Optional[Dict[str, str]] = None,
        **fields: str,
    ) -> Reply:
        refused = self._gateway_refuses(who, write=True)
        if refused:
            return refused
        data: Dict[str, str] = {"tags": tags, "tenant": who.tenant if who else "default", **fields}
        data.update(lie or {})
        if self.config == "gateway":
            data["tenant"] = who.tenant
        response = self.client.post(
            "/api/upload",
            files={"file": (name, text.encode("utf-8"), "text/markdown")},
            data=data,
            headers=self._headers(who, None),
        )
        return response.status_code, _json(response)

    def documents(self, who: Optional[Who]) -> Reply:
        refused = self._gateway_refuses(who)
        if refused:
            return refused
        response = self.client.get("/api/documents", headers=self._headers(who, None))
        return response.status_code, _json(response)

    def retag(self, who: Optional[Who], doc_id: str, tags: List[str]) -> Reply:
        refused = self._gateway_refuses(who, write=True)
        if refused:
            return refused
        response = self.client.put(
            f"/api/documents/{doc_id}/tags", json={"tags": tags}, headers=self._headers(who, None)
        )
        return response.status_code, _json(response)

    def delete(self, who: Optional[Who], doc_id: str) -> Reply:
        refused = self._gateway_refuses(who, write=True)
        if refused:
            return refused
        response = self.client.delete(f"/api/documents/{doc_id}", headers=self._headers(who, None))
        return response.status_code, _json(response)

    def health(self, headers: Optional[Dict[str, str]] = None) -> Any:
        """Health checks are probed by load balancers, so the gateway lets them through."""
        return self.client.get("/api/health", headers=headers or {})

    def operator(self, path: str, body: Dict[str, Any]) -> Reply:
        """An operator call with the governance token. ``tenant`` is sent in every configuration."""
        response = self.client.post(path, json=body, headers={"X-Governance-Token": GOVERNANCE_TOKEN})
        return response.status_code, _json(response)


# --- Reading responses -------------------------------------------------------------------


def visible_text(reply: Dict[str, Any]) -> str:
    """What the caller was actually shown: the answer and the chunks. Never the echoed question."""
    return json.dumps(
        {"answer": reply.get("answer"), "chunks": reply.get("authorized_chunks")}, ensure_ascii=False
    )


def sees(reply: Dict[str, Any], canary: str) -> bool:
    return canary in visible_text(reply)


def source_of(reply: Dict[str, Any], canary: str) -> str:
    for chunk in reply.get("authorized_chunks") or []:
        if canary in str(chunk.get("content", "")):
            return str(chunk.get("filename", "?"))
    return "the answer text"


def record(reply: Dict[str, Any], doc_id: str) -> Optional[Dict[str, Any]]:
    for document in reply.get("documents") or []:
        if document.get("doc_id") == doc_id:
            return document
    return None


def outcome(breached: bool, evidence: str, *, control_ok: bool = True, note: str = "") -> Dict[str, Any]:
    return {"breached": bool(breached), "evidence": evidence, "control_ok": bool(control_ok), "note": note}


# --- Shared setup ------------------------------------------------------------------------


def must_upload(t: Transport, who: Who, fixture: Tuple[str, ...], **fields: str) -> Dict[str, Any]:
    name, text = fixture[0], fixture[1]
    tags = fixture[2] if len(fixture) > 2 else ""
    status, reply = t.upload(who, name, text, tags, **fields)
    if status != 200:
        raise RuntimeError(f"setup upload of {name} for {who.tenant} failed: {status} {reply}")
    return reply["document"]


def seed(t: Transport) -> Dict[str, Dict[str, Any]]:
    """Both companies load their own documents, honestly, through the API."""
    docs = {
        "victim_treasury": must_upload(t, VICTIM_ADMIN, fx.GLOBEX_TREASURY),
        "victim_handbook": must_upload(t, VICTIM_ADMIN, fx.GLOBEX_HANDBOOK),
        "attacker_treasury": must_upload(t, ATTACKER_ADMIN, fx.ACME_TREASURY),
    }
    if not victim_reads(t):
        raise RuntimeError("control failed: the victim cannot read its own treasury document")
    return docs


def victim_reads(t: Transport, question: str = fx.Q_NIGHTJAR, canary: str = fx.SECRET_FIGURE) -> bool:
    status, reply = t.chat(VICTIM_FINANCE, question)
    return status == 200 and sees(reply, canary)


# --- Read attacks ------------------------------------------------------------------------


def read_restricted_same_role(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.chat(ATTACKER_FINANCE, fx.Q_NIGHTJAR)
    leaked = sees(reply, fx.SECRET_FIGURE)
    return outcome(
        leaked,
        f"HTTP {status}. EUR {fx.SECRET_FIGURE} returned from {source_of(reply, fx.SECRET_FIGURE)}."
        if leaked
        else f"HTTP {status}. {len(reply.get('authorized_chunks') or [])} chunks, none from globex.",
    )


def read_public_of_other_tenant(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.chat(ATTACKER_NOBODY, fx.Q_BADGE)
    leaked = sees(reply, fx.PUBLIC_CANARY)
    return outcome(
        leaked,
        f"HTTP {status}. Badge prefix {fx.PUBLIC_CANARY} returned from {source_of(reply, fx.PUBLIC_CANARY)}."
        if leaked
        else f"HTTP {status}. Globex's handbook was not returned.",
    )


def role_spoof_in_body(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.chat(ATTACKER_NOBODY, fx.Q_RESERVE, lie={"user_roles": ["super_admin"]})
    restricted = [c.get("filename") for c in reply.get("authorized_chunks") or [] if not c.get("is_public", True)]
    return outcome(
        bool(restricted),
        f"HTTP {status}. A user with no roles was given restricted chunks from {sorted(set(restricted))}."
        if restricted
        else f"HTTP {status}. Effective roles stayed {reply.get('effective_acl_filter')}.",
    )


def tenant_spoof_in_request(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.chat(
        ATTACKER_FINANCE,
        fx.Q_NIGHTJAR,
        lie={"tenant": fx.VICTIM},
        headers={"X-Tenant": fx.VICTIM, "X-Tenant-Id": fx.VICTIM},
        params={"tenant": fx.VICTIM},
    )
    leaked = sees(reply, fx.SECRET_FIGURE)
    return outcome(
        leaked,
        f"HTTP {status}. EUR {fx.SECRET_FIGURE} returned." if leaked else f"HTTP {status}. The claim was ignored.",
        note="Before Episode 5 there is no tenant to spoof: this leaks for the same reason as the first attack.",
    )


def masked_list_leak(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.chat(ATTACKER_NOBODY, fx.Q_NIGHTJAR)
    names = [str(m.get("filename")) for m in reply.get("masked_restricted_docs") or []]
    foreign = [name for name in names if name.startswith("globex_")]
    return outcome(
        bool(foreign),
        f"HTTP {status}. The 'blocked for you' list names {foreign}."
        if foreign
        else f"HTTP {status}. Blocked list: {names}.",
    )


def document_listing(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.documents(ATTACKER_FINANCE)
    foreign = [
        d.get("filename")
        for d in reply.get("documents") or []
        if d.get("tenant") == fx.VICTIM or str(d.get("filename", "")).startswith("globex_")
    ]
    return outcome(
        bool(foreign),
        f"HTTP {status}. The list includes {foreign}, with ids, tags and content hashes."
        if foreign
        else f"HTTP {status}. {len(reply.get('documents') or [])} documents, all acme's.",
    )


def tag_vocabulary_leak(t: Transport) -> Dict[str, Any]:
    seed(t)
    must_upload(t, VICTIM_ADMIN, fx.GLOBEX_DEAL_ROOM)
    _, listing = t.documents(ATTACKER_FINANCE)
    health = _json(t.health())
    where = []
    if fx.DEAL_TAG in json.dumps(listing.get("available_tags", [])):
        where.append("GET /api/documents (available_tags)")
    if fx.DEAL_TAG in json.dumps(health):
        where.append("GET /api/health")
    return outcome(
        bool(where),
        f"Globex's access tag '{fx.DEAL_TAG}' is shown by {' and '.join(where)}."
        if where
        else "The tag is not shown to acme.",
    )


class ConceptStub:
    """Stands in for the concept-extraction LLM. Off unless one attack switches it on."""

    enabled = False


def concept_expansion_leak(t: Transport) -> Dict[str, Any]:
    ConceptStub.enabled = True
    try:
        seed(t)
    finally:
        ConceptStub.enabled = False
    status, reply = t.chat(ATTACKER_NOBODY, fx.Q_NIGHTJAR_ZH)
    expanded = str(reply.get("expanded_query", ""))
    leaked = fx.SECRET_FIGURE in expanded or fx.SECRET_FIGURE in json.dumps(reply.get("matched_concepts", []))
    victim_status, victim_reply = t.chat(VICTIM_FINANCE, fx.Q_NIGHTJAR_ZH)
    control = victim_status == 200 and fx.SECRET_FIGURE in str(victim_reply.get("expanded_query", ""))
    return outcome(
        leaked,
        f"HTTP {status}. expanded_query came back as: {expanded!r}"
        if leaked
        else f"HTTP {status}. expanded_query: {expanded!r}",
        control_ok=control,
        note="The concept LLM is a stub that returns one fixed concept for the Nightjar document.",
    )


def unauthenticated_read(t: Transport) -> Dict[str, Any]:
    seed(t)
    status, reply = t.chat(ANONYMOUS, fx.Q_NIGHTJAR, lie={"user_roles": ["finance"]})
    leaked = sees(reply, fx.SECRET_FIGURE)
    return outcome(
        leaked,
        f"HTTP {status}. No credentials, EUR {fx.SECRET_FIGURE} returned."
        if leaked
        else f"HTTP {status}. {reply.get('detail', '')}",
    )


def health_disclosure(t: Transport) -> Dict[str, Any]:
    seed(t)
    response = t.health()
    extra = sorted(set(_json(response)) - LIVENESS_KEYS)
    return outcome(
        bool(extra),
        f"HTTP {response.status_code}. Without credentials: {extra}."
        if extra
        else f"HTTP {response.status_code}. Liveness only: {sorted(_json(response))}.",
    )


def cross_origin_grant(t: Transport) -> Dict[str, Any]:
    preflight = t.client.options(
        "/api/chat",
        headers={
            "Origin": EVIL_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    simple = t.health(headers={"Origin": EVIL_ORIGIN})
    grants = {
        "preflight": preflight.headers.get("access-control-allow-origin"),
        "simple": simple.headers.get("access-control-allow-origin"),
        "credentials": simple.headers.get("access-control-allow-credentials"),
    }
    granted = any(grants[key] in (EVIL_ORIGIN, "*") for key in ("preflight", "simple"))
    return outcome(
        granted,
        f"A page on {EVIL_ORIGIN} is allowed to call the API: {grants}."
        if granted
        else f"No cross-origin grant for {EVIL_ORIGIN}: {grants}.",
    )


# --- Write attacks -----------------------------------------------------------------------


def delete_foreign_document(t: Transport) -> Dict[str, Any]:
    docs = seed(t)
    status, _ = t.delete(ATTACKER_ADMIN, docs["victim_treasury"]["doc_id"])
    gone = not victim_reads(t)
    return outcome(
        gone,
        f"HTTP {status}. Globex's treasury policy is gone; Globex can no longer find its own budget."
        if gone
        else f"HTTP {status}. Globex's document is still there.",
    )


def retag_foreign_to_public(t: Transport) -> Dict[str, Any]:
    docs = seed(t)
    doc_id = docs["victim_treasury"]["doc_id"]
    status, _ = t.retag(ATTACKER_ADMIN, doc_id, [])
    _, reply = t.chat(ATTACKER_NOBODY, fx.Q_NIGHTJAR)
    _, listing = t.documents(VICTIM_ADMIN)
    now = record(listing, doc_id) or {}
    changed = now.get("tags") != ["finance"]
    leaked = sees(reply, fx.SECRET_FIGURE)
    return outcome(
        changed or leaked,
        f"HTTP {status}. Globex's restricted policy now has tags {now.get('tags')}"
        + (f" and an acme user with no roles reads EUR {fx.SECRET_FIGURE}." if leaked else ".")
        if changed or leaked
        else f"HTTP {status}. Tags are still {now.get('tags')}.",
    )


def poison_other_tenant_answers(t: Transport) -> Dict[str, Any]:
    seed(t)
    name, text, tags = fx.POISON
    status, _ = t.upload(ATTACKER_ADMIN, name, text, tags, lie={"tenant": fx.VICTIM})
    _, reply = t.chat(VICTIM_FINANCE, fx.Q_NIGHTJAR)
    poisoned = sees(reply, fx.POISON_CANARY)
    return outcome(
        poisoned,
        f"Upload HTTP {status}. Globex's finance user now gets acme's document ({name}) as a source: "
        f"'wire ... to account {fx.POISON_CANARY}'."
        if poisoned
        else f"Upload HTTP {status}. Globex's answer has only Globex's sources.",
        control_ok=status == 200,
    )


def supersede_by_source_key(t: Transport) -> Dict[str, Any]:
    docs = seed(t)
    victim = docs["victim_treasury"]
    name, text, tags = fx.SUPERSEDING_POISON
    status, _ = t.upload(
        ATTACKER_ADMIN, name, text, tags, lie={"tenant": fx.VICTIM}, source_key=str(victim.get("source_key", ""))
    )
    _, listing = t.documents(VICTIM_ADMIN)
    now = record(listing, victim["doc_id"]) or {}
    replaced = now.get("status") == "SUPERSEDED" or now.get("is_current") is False or not now
    still_reads = victim_reads(t)
    return outcome(
        replaced or not still_reads,
        f"Upload HTTP {status}. Globex's policy is now {now.get('status', 'missing')}; "
        f"the current version is acme's file."
        if replaced or not still_reads
        else f"Upload HTTP {status}. Globex's policy is still the current version.",
        control_ok=status == 200,
    )


def identical_file_two_tenants(t: Transport) -> Dict[str, Any]:
    seed(t)
    name, text = fx.SHARED_TEMPLATE
    mine = must_upload(t, VICTIM_ADMIN, (name, text, "finance"))
    status, theirs = t.upload(ATTACKER_ADMIN, name, text, "")
    _, listing = t.documents(VICTIM_ADMIN)
    now = record(listing, mine["doc_id"]) or {}
    changed = now.get("tags") != ["finance"] or now.get("status") != "INDEXED" or now.get("tenant") != fx.VICTIM
    reads = victim_reads(t, fx.Q_TEMPLATE, fx.TEMPLATE_CANARY)
    same_id = (theirs.get("document") or {}).get("doc_id") == mine["doc_id"]
    return outcome(
        changed or not reads or same_id,
        f"Upload HTTP {status}. Globex's copy changed: tags {now.get('tags')}, status {now.get('status')}, "
        f"tenant {now.get('tenant')}, same doc id {same_id}."
        if changed or not reads or same_id
        else f"Upload HTTP {status}. Two separate documents; Globex's copy is untouched.",
        control_ok=status == 200,
        note="Not an attack: two companies upload the same supplier template, with different access tags.",
    )


def existence_oracle(t: Transport) -> Dict[str, Any]:
    docs = seed(t)
    real = docs["victim_treasury"]["doc_id"]
    made_up = real[:-12] + "0" * 12
    real_status, real_reply = t.retag(ATTACKER_ADMIN, real, ["finance"])
    fake_status, fake_reply = t.retag(ATTACKER_ADMIN, made_up, ["finance"])
    # The reply may echo the id that was asked for. That tells the caller nothing.
    real_body = json.dumps(real_reply, sort_keys=True).replace(real, "<id>")
    fake_body = json.dumps(fake_reply, sort_keys=True).replace(made_up, "<id>")
    same = real_status == fake_status and real_body == fake_body
    return outcome(
        not same,
        f"Globex's real document id answers HTTP {real_status}, a made-up id answers HTTP {fake_status}: "
        "an attacker can test which ids exist."
        if not same
        else f"A real id and a made-up id both answer HTTP {real_status} with the same body.",
        note="Before Episode 5 the endpoint says 'updated' for any id, so there is nothing to probe. "
        "It does not need probing either: the document list hands out every id.",
    )


# --- Operator actions on behalf of acme ---------------------------------------------------


def _both_expenses(t: Transport) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed(t)
    victim = must_upload(t, VICTIM_ADMIN, fx.GLOBEX_EXPENSES)
    attacker = must_upload(t, ATTACKER_ADMIN, fx.ACME_EXPENSES)
    if not victim_reads(t, fx.Q_EXPENSES, fx.VICTIM_EXPENSES_CANARY):
        raise RuntimeError("control failed: globex cannot read its expenses policy")
    return victim, attacker


def _attacker_reads(t: Transport, question: str, canary: str) -> bool:
    status, reply = t.chat(ATTACKER_ADMIN, question)
    return status == 200 and any(
        canary in str(c.get("content", "")) for c in reply.get("authorized_chunks") or []
    )


def erasure_by_source_blast_radius(t: Transport) -> Dict[str, Any]:
    _, attacker = _both_expenses(t)
    status, reply = t.operator(
        "/api/erasure", {"tenant": fx.ATTACKER, "source_key": attacker["source_key"], "dry_run": False}
    )
    acme_erased = not _attacker_reads(t, fx.Q_EXPENSES, fx.ATTACKER_EXPENSES_CANARY)
    globex_intact = victim_reads(t, fx.Q_EXPENSES, fx.VICTIM_EXPENSES_CANARY)
    return outcome(
        not globex_intact,
        f"HTTP {status} {reply.get('status')}. Erasing acme's expenses_policy.md also removed Globex's file "
        "of the same name."
        if not globex_intact
        else f"HTTP {status} {reply.get('status')}. Acme's file is gone, Globex's file of the same name is intact.",
        control_ok=acme_erased,
    )


def subject_erasure_crosses_tenants(t: Transport) -> Dict[str, Any]:
    seed(t)
    must_upload(t, VICTIM_ADMIN, fx.GLOBEX_CUSTOMER, subject_ids=fx.SHARED_SUBJECT)
    must_upload(t, ATTACKER_ADMIN, fx.ACME_CUSTOMER, subject_ids=fx.SHARED_SUBJECT)
    if not victim_reads(t, fx.Q_CUSTOMER, fx.VICTIM_CUSTOMER_CANARY):
        raise RuntimeError("control failed: globex cannot read its customer case")
    status, reply = t.operator(
        "/api/erasure", {"tenant": fx.ATTACKER, "subject_id": fx.SHARED_SUBJECT, "dry_run": False}
    )
    acme_erased = not _attacker_reads(t, fx.Q_CUSTOMER, fx.ATTACKER_CUSTOMER_CANARY)
    globex_intact = victim_reads(t, fx.Q_CUSTOMER, fx.VICTIM_CUSTOMER_CANARY)
    return outcome(
        not globex_intact,
        f"HTTP {status} {reply.get('status')}. A request sent to acme deleted Globex's records about the "
        "same person. Globex is a separate data controller and was never asked."
        if not globex_intact
        else f"HTTP {status} {reply.get('status')}. Acme's record is gone, Globex's record is intact.",
        control_ok=acme_erased,
    )


def tombstone_blocks_other_tenant(t: Transport) -> Dict[str, Any]:
    seed(t)
    name, text = fx.SHARED_TEMPLATE
    theirs = must_upload(t, ATTACKER_ADMIN, (name, text, ""))
    erase_status, _ = t.operator(
        "/api/erasure", {"tenant": fx.ATTACKER, "source_key": theirs["source_key"], "dry_run": False}
    )
    status, reply = t.upload(VICTIM_ADMIN, name, text, "finance")
    blocked = status != 200
    return outcome(
        blocked,
        f"Erasure HTTP {erase_status}. Globex's upload of the same file is refused with HTTP {status}: "
        f"{str(reply.get('detail', ''))[:110]}"
        if blocked
        else f"Erasure HTTP {erase_status}. Globex uploads the same file: HTTP {status}.",
        control_ok=erase_status == 200,
        note="A refusal also tells Globex that somebody else held this exact file and erased it.",
    )


def sync_event_id_collision(t: Transport) -> Dict[str, Any]:
    victim, attacker = _both_expenses(t)
    first_status, first = t.operator(
        "/api/sync/events",
        {"tenant": fx.ATTACKER, "event_id": "evt-1001", "event_type": "archived", "source_key": attacker["source_key"]},
    )
    second_status, second = t.operator(
        "/api/sync/events",
        {"tenant": fx.VICTIM, "event_id": "evt-1001", "event_type": "deleted", "source_key": victim["source_key"]},
    )
    still_there = victim_reads(t, fx.Q_EXPENSES, fx.VICTIM_EXPENSES_CANARY)
    dropped = second.get("status") == "duplicate_ignored" or still_there
    return outcome(
        dropped,
        f"Globex's delete event came back '{second.get('status')}' (HTTP {second_status}) because acme's "
        "source system had already used event id evt-1001. The file Globex deleted is still searchable."
        if dropped
        else f"Globex's delete event was applied: '{second.get('status')}' (HTTP {second_status}).",
        control_ok=first_status == 200 and first.get("status") == "archived",
    )


def sync_event_names_foreign_source(t: Transport) -> Dict[str, Any]:
    victim, _ = _both_expenses(t)
    status, reply = t.operator(
        "/api/sync/events",
        {"tenant": fx.ATTACKER, "event_id": "evt-2002", "event_type": "deleted", "source_key": victim["source_key"]},
    )
    globex_intact = victim_reads(t, fx.Q_EXPENSES, fx.VICTIM_EXPENSES_CANARY)
    return outcome(
        not globex_intact,
        f"HTTP {status} {reply.get('status')}. An event from acme's connector deleted Globex's file."
        if not globex_intact
        else f"HTTP {status} {reply.get('status')}. Globex's file is intact.",
    )


ATTACKS: List[Tuple[str, str, str, str, Callable[[Transport], Dict[str, Any]]]] = [
    # id, category, who, title, function
    ("read_restricted_same_role", "read", "acme finance user", "Ask for another tenant's restricted figure", read_restricted_same_role),
    ("read_public_of_other_tenant", "read", "acme user, no roles", "Read another tenant's 'public' document", read_public_of_other_tenant),
    ("role_spoof_in_body", "read", "acme user, no roles", "Claim super_admin in the request body", role_spoof_in_body),
    ("tenant_spoof_in_request", "read", "acme finance user", "Claim tenant=globex in body, header and URL", tenant_spoof_in_request),
    ("masked_list_leak", "read", "acme user, no roles", "Read file names from the 'blocked for you' list", masked_list_leak),
    ("document_listing", "read", "acme finance user", "List the documents", document_listing),
    ("tag_vocabulary_leak", "read", "acme finance user", "Read the other tenant's access tags", tag_vocabulary_leak),
    ("concept_expansion_leak", "read", "acme user, no roles", "Trigger a concept learned from the other tenant's document", concept_expansion_leak),
    ("unauthenticated_read", "read", "nobody (no credentials)", "Ask with no credentials at all", unauthenticated_read),
    ("health_disclosure", "read", "nobody (no credentials)", "Read the health endpoint", health_disclosure),
    ("cross_origin_grant", "read", "a web page on another site", "Call the API from another origin", cross_origin_grant),
    ("delete_foreign_document", "write", "acme admin", "Delete another tenant's document by id", delete_foreign_document),
    ("retag_foreign_to_public", "write", "acme admin", "Retag another tenant's document as public", retag_foreign_to_public),
    ("poison_other_tenant_answers", "write", "acme admin", "Plant a document in another tenant's answers", poison_other_tenant_answers),
    ("supersede_by_source_key", "write", "acme admin", "Replace another tenant's document with a new 'version'", supersede_by_source_key),
    ("identical_file_two_tenants", "write", "acme admin", "Upload the same file another tenant holds", identical_file_two_tenants),
    ("existence_oracle", "write", "acme admin", "Test whether a document id exists", existence_oracle),
    ("erasure_by_source_blast_radius", "operator", "operator, for acme", "Erase acme's file that shares a name with Globex's", erasure_by_source_blast_radius),
    ("subject_erasure_crosses_tenants", "operator", "operator, for acme", "Erase a person for acme only", subject_erasure_crosses_tenants),
    ("tombstone_blocks_other_tenant", "operator", "operator, for acme", "Erase acme's file, then Globex uploads the same file", tombstone_blocks_other_tenant),
    ("sync_event_id_collision", "operator", "two source systems", "Two tenants' connectors reuse one event id", sync_event_id_collision),
    ("sync_event_names_foreign_source", "operator", "acme's connector", "Acme's connector names Globex's source", sync_event_names_foreign_source),
]


# --- Token forgery (only meaningful where tokens exist) ----------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign(header: Dict[str, Any], claims: Dict[str, Any], key: bytes) -> str:
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}"
    digest = {"HS256": hashlib.sha256, "HS384": hashlib.sha384, "HS512": hashlib.sha512}[header["alg"]]
    return f"{signing_input}.{_b64(hmac.new(key, signing_input.encode(), digest).digest())}"


def _claims(tenant: Any = fx.VICTIM, **overrides: Any) -> Dict[str, Any]:
    now = int(time.time())
    claims = {"sub": "mallory", "tenant": tenant, "roles": ["admin"], "aud": "beyondrag", "iat": now, "exp": now + 600}
    claims.update(overrides)
    return {key: value for key, value in claims.items() if value is not None}


def forged_tokens(t: Transport, secret: bytes) -> List[Dict[str, Any]]:
    seed(t)
    real = t._mint(fx.ATTACKER, ["finance"], subject="fiona@acme")
    head, _, signature = real.split(".")
    hs256 = {"alg": "HS256", "typ": "JWT"}
    cases: List[Tuple[str, str, Optional[str]]] = [
        ("no_token", "No Authorization header", None),
        ("not_a_token", "Random text as the token", "let-me-in"),
        (
            "alg_none",
            "Unsigned token with alg: none",
            f"{_b64(json.dumps({'alg': 'none', 'typ': 'JWT'}).encode())}.{_b64(json.dumps(_claims()).encode())}.",
        ),
        ("wrong_key", "Signed with a key the attacker made up", _sign(hs256, _claims(), b"attacker-chosen-key-attacker-chosen-key")),
        (
            "tampered_tenant",
            "A real acme token with the tenant edited to globex, signature kept",
            f"{head}.{_b64(json.dumps(_claims(sub='fiona@acme', roles=['finance'])).encode())}.{signature}",
        ),
        ("expired", "A real globex token, replayed after it expired", _sign(hs256, _claims(exp=int(time.time()) - 60), secret)),
        ("other_algorithm", "Correct key, HS512 instead of the pinned HS256", _sign({"alg": "HS512", "typ": "JWT"}, _claims(), secret)),
        ("other_audience", "A token issued for a different service, same key", _sign(hs256, _claims(aud="billing-api"), secret)),
        ("no_tenant_claim", "Correctly signed, but no tenant claim", _sign(hs256, _claims(tenant=None), secret)),
        ("wildcard_tenant", "Correctly signed, tenant '*'", _sign(hs256, _claims(tenant="*"), secret)),
        ("two_tenants", "Correctly signed, tenant 'acme,globex'", _sign(hs256, _claims(tenant="acme,globex"), secret)),
        ("no_expiry", "Correctly signed, no expiry", _sign(hs256, _claims(exp=None), secret)),
    ]
    results = []
    for case_id, title, token in cases:
        headers = {} if token is None else {"Authorization": f"Bearer {token}"}
        response = t.client.post("/api/chat", json={"question": fx.Q_NIGHTJAR}, headers=headers)
        reply = _json(response)
        results.append(
            {
                "id": case_id,
                "title": title,
                "status": response.status_code,
                "accepted": response.status_code == 200,
                "leaked": sees(reply, fx.SECRET_FIGURE),
                "detail": str(reply.get("detail", ""))[:120],
            }
        )
    control_status, control = t.chat(VICTIM_FINANCE, fx.Q_NIGHTJAR)
    results.append(
        {
            "id": "control_valid_token",
            "title": "Control: a valid globex finance token",
            "status": control_status,
            "accepted": control_status == 200,
            "leaked": False,
            "detail": "reads its own figure" if sees(control, fx.SECRET_FIGURE) else "did NOT read its own figure",
        }
    )
    return results


# --- Process plumbing --------------------------------------------------------------------


def _install_concept_stub() -> None:
    """Replace the concept-extraction HTTP call. Ingestion would otherwise wait on a local LLM."""
    import src.rag.cross_lingual as cross_lingual

    real_httpx = cross_lingual.httpx

    class _Response:
        def __init__(self, content: Dict[str, Any]) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"message": {"content": json.dumps(self._content, ensure_ascii=False)}}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def post(self, url: str, json: Optional[Dict[str, Any]] = None, **kwargs: Any) -> _Response:  # noqa: A002
            prompt = " ".join(str(m.get("content", "")) for m in (json or {}).get("messages", []))
            if ConceptStub.enabled and "Nightjar acquisition budget is EUR" in prompt:
                return _Response(fx.STUB_NIGHTJAR_CONCEPT)
            return _Response({})

    class _Httpx:
        Client = _Client

        def __getattr__(self, name: str) -> Any:
            return getattr(real_httpx, name)

    cross_lingual.httpx = _Httpx()


class Sandbox:
    """Fresh state for every attack: empty workspace, empty index, new service."""

    KEEP = {".gitkeep", "language_packs.json", "dynamic_concepts.json"}

    def __init__(self, config: str) -> None:
        self.config = config
        self.workspace = CWD / "workspace"
        self._concepts = (self.workspace / "dynamic_concepts.json").read_bytes()

    def fresh(self) -> None:
        import src.rag.cross_lingual as cross_lingual
        import src.rag.service as service

        for entry in self.workspace.iterdir():
            if entry.name in self.KEEP:
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        (self.workspace / "dynamic_concepts.json").write_bytes(self._concepts)
        service._KB_SERVICE_INSTANCE = None
        cross_lingual._REGISTRY = None
        kb = service.get_kb_service()  # before Episode 5 this also resets the index and loads samples
        if self.config == "branch":
            kb._index_manager.reset_index()

    def restart(self) -> None:
        """What a process restart does: the singletons go, the disk and the index stay."""
        import src.rag.cross_lingual as cross_lingual
        import src.rag.service as service

        service._KB_SERVICE_INSTANCE = None
        cross_lingual._REGISTRY = None
        service.get_kb_service()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, choices=["main", "gateway", "branch"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="", help="Comma-separated attack ids, for debugging.")
    args = parser.parse_args()

    if not SANDBOX_MARKER.exists():
        sys.exit(f"{CWD} is not marked as an attack sandbox. Run this through ep5_benchmark.py.")

    os.environ["RAG_GOVERNANCE_TOKEN"] = GOVERNANCE_TOKEN
    os.environ.setdefault("USE_LOCAL_ONLY", "true")

    import src

    if not str(Path(src.__file__).resolve()).startswith(str(CWD)):
        sys.exit(f"imported src from {src.__file__}, not from the sandbox {CWD}")

    from fastapi.testclient import TestClient

    from src.web.app import app

    _install_concept_stub()

    mint = None
    secret = os.environ.get("RAG_AUTH_SECRET", "").encode("utf-8")
    if args.config == "branch":
        from src.rag.auth import auth_mode, mint_token

        if auth_mode() != "required":
            sys.exit("the branch configuration must run with RAG_AUTH_MODE=required")
        mint = mint_token

    transport = Transport(args.config, TestClient(app, raise_server_exceptions=False), mint)
    sandbox = Sandbox(args.config)
    only = {item.strip() for item in args.only.split(",") if item.strip()}

    attacks = []
    for attack_id, category, attacker, title, run in ATTACKS:
        if only and attack_id not in only:
            continue
        started = time.time()
        try:
            sandbox.fresh()
            result = run(transport)
        except Exception as exc:  # an attack that cannot be set up is reported, never counted
            result = {"breached": None, "evidence": f"not run: {type(exc).__name__}: {exc}", "control_ok": False, "note": ""}
        result.update(id=attack_id, category=category, attacker=attacker, title=title, seconds=round(time.time() - started, 2))
        attacks.append(result)
        flag = {True: "BREACH", False: "held  ", None: "ERROR "}[result["breached"]]
        print(f"[{args.config:7}] {flag} {attack_id:34} {result['evidence'][:150]}", flush=True)

    forgery: List[Dict[str, Any]] = []
    restart: Dict[str, Any] = {}
    if not only:
        if args.config == "branch":
            sandbox.fresh()
            forgery = forged_tokens(transport, secret)
            for row in forgery:
                print(f"[forgery] {row['id']:22} HTTP {row['status']} accepted={row['accepted']} {row['detail'][:90]}", flush=True)
        sandbox.fresh()
        seed(transport)
        sandbox.restart()
        kept = victim_reads(transport)
        restart = {
            "data_kept_after_restart": kept,
            "evidence": "After a restart Globex still reads its own budget."
            if kept
            else "After a restart the index was reset and the samples reloaded: Globex's documents are gone.",
        }
        print(f"[{args.config:7}] restart: {restart['evidence']}", flush=True)

    Path(args.out).write_text(
        json.dumps(
            {
                "config": args.config,
                "src": str(Path(src.__file__).resolve().parent),
                "attacks": attacks,
                "forgery": forgery,
                "restart": restart,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
