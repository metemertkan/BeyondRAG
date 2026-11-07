"""Deterministic test corpora for the Episode 3 benchmark. All names and figures are invented."""

from __future__ import annotations

import random
from typing import Any, Dict, List

SEED = 42

_SERVICES = ["payment gateway", "billing service", "inventory sync", "report builder", "notification hub", "search index", "file uploader", "session manager"]
_CAUSES = ["a stale cache entry", "an expired certificate", "a full disk", "a dropped connection", "a malformed payload", "a schema mismatch", "a rate limit", "a clock drift"]
_ACTIONS = ["restart the worker", "rotate the credential", "clear the queue", "roll back the release", "increase the limit", "re-run the migration"]
_PRODUCT_LINES = ["ACME", "ZENTA", "KORVO", "NIMBUS"]
_PRODUCT_NOUNS = ["router", "sensor", "gateway", "controller", "scanner", "battery pack"]


def hard_exact_match_corpus() -> Dict[str, Any]:
    """The conditions where exact match breaks: long, near-identical pages that share code parts.

    - Sequential error codes, so their embeddings are close to each other.
    - One code in a long page, so it barely affects the page's embedding.
    - Cross-references that repeat the parts of another SKU ("ACME" and "X200" in the same page
      without "ACME-X200"), so an analyzer that splits on hyphens can't tell them apart.
    """
    rng = random.Random(SEED + 1)
    docs: List[Dict[str, str]] = []
    queries: List[Dict[str, str]] = []

    def page(body: str) -> str:
        before = " ".join(rng.sample(_FILLER, 4))
        after = " ".join(rng.sample(_FILLER, 4))
        return f"{before} {body} {after}"

    for family in rng.sample(range(100, 999), 12):
        for digit in range(10):
            code = f"ERR-{family}{digit}"
            doc_id = f"err-{code.lower()}"
            docs.append({"id": doc_id, "text": page(f"Error {code} is raised by the {rng.choice(_SERVICES)}. To resolve it, {rng.choice(_ACTIONS)} and check the logs.")})
            if digit in (3, 7):
                queries.append({"query": code, "target": doc_id, "kind": "bare error code"})
                queries.append({"query": f"What should I do about error {code}?", "target": doc_id, "kind": "error code in a question"})

    models = rng.sample(range(100, 999), 30)
    for line_index, line in enumerate(_PRODUCT_LINES):
        other = _PRODUCT_LINES[(line_index + 1) % len(_PRODUCT_LINES)]
        for position, model in enumerate(models):
            sku = f"{line}-X{model}"
            # Same model number on another line, and another model on this line: all the parts of
            # "sku" appear in this neighbour's page, but never the whole code.
            decoy_model = models[(position + 1) % len(models)]
            doc_id = f"sku-{sku.lower()}"
            docs.append(
                {
                    "id": doc_id,
                    "text": page(
                        f"The {sku} {rng.choice(_PRODUCT_NOUNS)} costs {rng.randint(50, 900)} dollars and weighs "
                        f"{rng.randint(1, 12)} kilograms. It pairs with the {other}-X{decoy_model} adapter and the "
                        f"{line}-X{(model + 1) % 1000} bracket."
                    ),
                }
            )
            if position % 3 == 0:
                queries.append({"query": sku, "target": doc_id, "kind": "bare SKU"})
                queries.append({"query": f"How much does the {sku} cost?", "target": doc_id, "kind": "SKU in a question"})
    return {"docs": docs, "queries": queries}


def exact_match_corpus() -> Dict[str, Any]:
    """Codes that differ by a digit or suffix, in near-identical wording, plus generic manuals."""
    rng = random.Random(SEED)
    docs: List[Dict[str, str]] = []
    queries: List[Dict[str, str]] = []
    numbers = rng.sample(range(1000, 9999), 60)
    for number in numbers:
        family = [f"ERR-{number}", f"ERR-{str(number)[0]}{str(number)[2]}{str(number)[1]}{str(number)[3]}"]
        for code in dict.fromkeys(family):
            doc_id = f"err-{code.lower()}"
            if any(d["id"] == doc_id for d in docs):
                continue
            service, cause, action = rng.choice(_SERVICES), rng.choice(_CAUSES), rng.choice(_ACTIONS)
            docs.append({"id": doc_id, "text": f"Error {code} is raised by the {service} after {cause}. To resolve it, {action} and check the logs."})
        queries.append({"query": f"{family[0]}", "target": f"err-{family[0].lower()}", "kind": "bare error code"})
        queries.append({"query": f"What should I do about error {family[0]}?", "target": f"err-{family[0].lower()}", "kind": "error code in a question"})
    for line in _PRODUCT_LINES:
        for model in rng.sample(range(100, 999), 12):
            noun = rng.choice(_PRODUCT_NOUNS)
            price, weight = rng.randint(50, 900), rng.randint(1, 12)
            for sku in (f"{line}-X{model}", f"{line}-X{model}-PRO"):
                doc_id = f"sku-{sku.lower()}"
                docs.append({"id": doc_id, "text": f"The {sku} {noun} costs {price} dollars and weighs {weight} kilograms. Warranty and returns follow the standard policy."})
                price += 40
            queries.append({"query": f"{line}-X{model}", "target": f"sku-{line.lower()}-x{model}", "kind": "bare SKU"})
            queries.append({"query": f"How much does the {line}-X{model}-PRO cost?", "target": f"sku-{line.lower()}-x{model}-pro", "kind": "SKU in a question"})
    for index in range(40):
        service, action = rng.choice(_SERVICES), rng.choice(_ACTIONS)
        docs.append({"id": f"manual-{index}", "text": f"General troubleshooting guide for the {service}. Most error codes clear once you {action}. Product codes are printed on the label. See {rng.randint(1000, 9999)} prior incidents."})
    return {"docs": docs, "queries": queries}


# Written by hand: the user's words share little vocabulary with the document they need.
RUNBOOKS: List[Dict[str, str]] = [
    {"id": "auth-timeout", "text": "Authentication Service Timeout Exceptions. Token validation calls to the identity provider exceed the 3 second deadline under load, and the client receives HTTP 504.", "query": "app keeps crashing on login"},
    {"id": "tls-expiry", "text": "TLS Certificate Rotation Failure. The load balancer serves an expired leaf certificate after the renewal job fails, and clients reject the handshake.", "query": "browser says the site is not secure"},
    {"id": "oom-kill", "text": "Container Memory Limit Exceeded. The kernel out-of-memory killer terminates the worker when resident set size passes the cgroup limit.", "query": "the worker just dies randomly with no error"},
    {"id": "deadlock", "text": "Database Lock Contention and Deadlock Resolution. Concurrent transactions acquire row locks in opposite order, and the engine aborts one as the victim.", "query": "saving an order sometimes fails if two people edit it"},
    {"id": "dns-ttl", "text": "DNS Propagation Delay After Failover. Resolvers cache the old record until the time-to-live expires, so traffic keeps reaching the drained region.", "query": "some users still hit the old site after we switched"},
    {"id": "n-plus-one", "text": "Unbounded ORM Query Fan-out. Lazy loading issues one additional select per parent row, which multiplies database round trips on list endpoints.", "query": "the customers page got really slow when we added more customers"},
    {"id": "clock-skew", "text": "NTP Clock Skew and Signature Validation. A host clock drifting beyond the allowed window makes request signatures appear expired.", "query": "api calls say my signature is too old but I just made it"},
    {"id": "disk-inode", "text": "Inode Exhaustion on Log Volumes. Millions of small rotated files consume all inodes while free block capacity remains available.", "query": "it says no space left but the disk shows half empty"},
    {"id": "cors", "text": "Cross-Origin Resource Sharing Preflight Rejection. The gateway omits the allow-origin header on OPTIONS requests, so browsers block the response.", "query": "the frontend can't call the backend but curl works fine"},
    {"id": "retry-storm", "text": "Retry Amplification During Partial Outage. Clients without exponential backoff multiply load on a degraded dependency, which prolongs the incident.", "query": "things got worse the longer the outage lasted"},
    {"id": "index-bloat", "text": "Stale Query Planner Statistics. Outdated table statistics make the optimizer choose a sequential scan over an available index.", "query": "the same report was fast yesterday and takes minutes today"},
    {"id": "sso-saml", "text": "SAML Assertion Audience Mismatch. The identity provider issues assertions for a different entity identifier, so the service provider rejects them.", "query": "SSO stopped working after we renamed the app"},
    {"id": "webhook-sig", "text": "Webhook HMAC Verification Failure. Body normalization by a proxy changes the raw payload bytes, which invalidates the computed digest.", "query": "payment notifications get rejected since we added the new proxy"},
    {"id": "gc-pause", "text": "Stop-the-World Garbage Collection Pauses. Old generation heap pressure triggers long collector pauses that exceed the health check deadline.", "query": "the service freezes for a few seconds every so often"},
    {"id": "conn-pool", "text": "Database Connection Pool Exhaustion. Long transactions hold connections until new requests queue and hit the acquisition timeout.", "query": "everything hangs at peak hours then recovers on its own"},
    {"id": "cache-stampede", "text": "Cache Stampede on Key Expiry. Many concurrent misses for one popular key recompute the same value and overload the origin.", "query": "the database spikes exactly every hour on the hour"},
    {"id": "timezone", "text": "UTC Offset Handling in Scheduled Jobs. Daylight saving transitions shift the local schedule, so jobs run twice or are skipped.", "query": "the nightly job ran twice last sunday"},
    {"id": "utf8", "text": "Character Encoding Mismatch in CSV Import. Latin-1 input decoded as UTF-8 produces replacement characters in customer names.", "query": "customer names show up with weird symbols after upload"},
    {"id": "feature-flag", "text": "Stale Feature Flag Evaluation. The SDK serves its cached ruleset when the flag service is unreachable, so targeting changes do not apply.", "query": "I turned the option off but some people still see it"},
    {"id": "pagination", "text": "Offset Pagination Drift Under Concurrent Writes. Inserts between page requests shift rows, which causes duplicates and omissions.", "query": "the list shows the same item twice when scrolling"},
    {"id": "kafka-lag", "text": "Consumer Group Rebalance Lag. Frequent rebalances pause partition consumption, and offset lag grows until processing falls behind.", "query": "events show up way late during deploys"},
    {"id": "s3-consistency", "text": "Presigned URL Expiry for Object Storage. Download links signed with a short lifetime return 403 once the validity window ends.", "query": "the download link worked earlier and now says forbidden"},
    {"id": "idempotency", "text": "Missing Idempotency Keys on Payment Retries. A client retry after a timeout creates a second charge because the server can't detect the duplicate.", "query": "a customer got charged twice"},
    {"id": "log-sampling", "text": "Trace Sampling Drops Rare Failures. Head-based sampling discards most spans before the outcome is known, so rare errors never reach the backend.", "query": "the bug happens but there's nothing in our monitoring"},
]


def mismatch_corpus() -> Dict[str, Any]:
    return {
        "docs": [{"id": item["id"], "text": item["text"]} for item in RUNBOOKS],
        "queries": [{"query": item["query"], "target": item["id"]} for item in RUNBOOKS],
    }


_FILLER = [
    "This section applies to all regional entities and is reviewed by the governance committee every year.",
    "Exceptions require written approval and must be recorded in the central register before the work begins.",
    "Teams should keep supporting records for seven years, in line with the group retention schedule.",
    "Questions about how this applies to a specific case should go to the policy owner named in the appendix.",
    "Staff complete the related training module when they join, and again whenever the policy changes materially.",
    "Local regulation takes precedence where it is stricter, and the regional lead documents any such difference.",
    "Internal audit samples compliance with this section as part of the annual assurance plan for each region.",
    "Changes to this section take effect thirty days after publication unless the notice states another date.",
]

_SECTIONS = [
    ("travel approval", "How much can a manager approve for one business trip?", "A line manager may approve a single trip of up to HKD 18,000", "18,000"),
    ("laptop replacement", "How often are staff laptops replaced?", "Standard laptops are replaced every 42 months", "42 months"),
    ("vendor onboarding", "How many quotes are needed before onboarding a new vendor?", "Three competing quotes are mandatory for new vendors", "Three competing quotes"),
    ("data retention", "How long is customer chat history kept?", "Customer chat history is deleted after 26 months", "26 months"),
    ("incident response", "How quickly must a severe security incident be reported?", "Severe security incidents must reach the security office within 55 minutes", "55 minutes"),
    ("expense claims", "What is the deadline for submitting an expense claim?", "Claims must be filed within 21 days of the purchase", "21 days"),
    ("remote work", "How many days per year may staff work from abroad?", "Staff may work from another country for at most 34 days a year", "34 days"),
    ("contract signing", "Who has to sign contracts above the regional limit?", "Contracts above the regional limit need the signature of the deputy treasurer", "deputy treasurer"),
    ("access reviews", "How often are privileged access rights reviewed?", "Privileged access rights are reviewed every 75 days", "75 days"),
    ("training budget", "What is the annual training budget per person?", "Each employee has an annual training budget of HKD 12,500", "12,500"),
    ("gift limits", "What is the limit on gifts from suppliers?", "Gifts from suppliers may not exceed HKD 640 in value", "640"),
    ("backup testing", "How often are backup restores tested?", "Backup restores are tested every 45 days", "45 days"),
]


def parent_child_corpus() -> Dict[str, Any]:
    """The heading and question words open a section. The answer sits well away from them."""
    rng = random.Random(SEED)
    docs: List[Dict[str, str]] = []
    queries: List[Dict[str, str]] = []
    for index, (topic, question, fact, answer) in enumerate(_SECTIONS):
        def filler(count: int) -> str:
            return " ".join(rng.choice(_FILLER) for _ in range(count))

        intro = f"Section {index + 1}: {topic}. This section sets the rules for {topic} across the group, and answers the common question: {question}"
        body = f"{intro} {filler(5)} {filler(5)} {fact}. {filler(4)}"
        docs.append({"id": f"section-{index + 1}", "text": body})
        queries.append({"query": question, "target": f"section-{index + 1}", "answer": answer})
    return {"docs": docs, "queries": queries}


def multi_topic_parent_child_corpus() -> Dict[str, Any]:
    """Each page covers several unrelated topics, as long policy pages do.

    A large chunk holding all of them gets a blended embedding, which is what small-to-big fixes.
    """
    rng = random.Random(SEED + 2)
    sections = list(enumerate(_SECTIONS))
    rng.shuffle(sections)
    docs: List[Dict[str, str]] = []
    queries: List[Dict[str, str]] = []
    for page, start in enumerate(range(0, len(sections), 4)):
        parts: List[str] = []
        for _, (topic, question, fact, answer) in sections[start:start + 4]:
            filler = " ".join(rng.choice(_FILLER) for _ in range(3))
            parts.append(f"{topic.capitalize()}. {fact}. {filler}")
            queries.append({"query": question, "target": f"page-{page + 1}", "answer": answer})
        docs.append({"id": f"page-{page + 1}", "text": "\n\n".join(parts)})
    # Single-topic pages on nearby subjects compete with the blended embeddings.
    for index, (topic, _, _, _) in enumerate(_SECTIONS):
        docs.append({"id": f"overview-{index + 1}", "text": f"Overview of {topic}. This page introduces how {topic} is organized across the group and who owns it. Specific limits are published in the relevant policy pages."})
    return {"docs": docs, "queries": queries}


_LAUNCHES = [("Q1", "Orion"), ("Q2", "Vega"), ("Q3", "Helios"), ("Q4", "Lyra")]
_REGIONS = {
    "Europe": ["EUR 4,200,000", "EUR 1,150,000", "EUR 2,760,000", "EUR 980,000"],
    "Asia": ["USD 3,100,000", "USD 2,450,000", "USD 1,880,000", "USD 640,000"],
}


def multi_hop_corpus() -> Dict[str, Any]:
    docs: List[Dict[str, str]] = []
    queries: List[Dict[str, Any]] = []
    for quarter, product in _LAUNCHES:
        docs.append({"id": f"launch-{quarter}", "text": f"{quarter} launch review. The product we launched in {quarter} was {product}, after an eight month development cycle and a staged rollout."})
    for region, figures in _REGIONS.items():
        for (quarter, product), figure in zip(_LAUNCHES, figures):
            docs.append({"id": f"revenue-{product}-{region}", "text": f"Regional revenue report. {product} revenue in {region} was {figure} this financial year, with growth driven by enterprise renewals."})
            queries.append(
                {
                    "query": f"What was the revenue in {region} from the product we launched in {quarter}?",
                    "targets": [f"launch-{quarter}", f"revenue-{product}-{region}"],
                    "answer": figure,
                }
            )
    return {"docs": docs, "queries": queries}


_DISTRACTORS = [
    "The cafeteria on level two opens at eight and closes after the lunch service.",
    "Visitors must sign in at reception and wear a badge at all times in the building.",
    "The quarterly town hall takes place in the main auditorium and is streamed to all regions.",
    "Parking permits are renewed every January through the facilities portal.",
    "The printer on level five is reserved for the design team during business hours.",
    "New hires receive their equipment during the first morning of orientation.",
    "The fire drill is scheduled twice a year and takes about twenty minutes.",
    "Meeting rooms can be booked up to two weeks ahead using the calendar tool.",
    "The wellness allowance can be used for gym memberships and approved classes.",
    "Lost property is kept at the front desk for one month before it is donated.",
    "The office closes early on the last working day before a public holiday.",
    "Bicycle storage is in the basement and requires a separate access card.",
    "The internal newsletter is published on the first Monday of each month.",
    "Coffee machines are serviced every Thursday morning before the office opens.",
    "Recycling bins are collected from each floor on Tuesdays and Fridays.",
]

ATTENTION_FACTS = [
    ("The maximum hotel rate for Singapore is SGD 372 per night.", "What is the maximum hotel rate for Singapore?", "372"),
    ("Project Kestrel has a codename of BLUE-HARBOR.", "What is the codename of Project Kestrel?", "BLUE-HARBOR"),
    ("The server room door code changes every 19 days.", "How often does the server room door code change?", "19"),
    ("The audit committee chair is Priya Raman.", "Who chairs the audit committee?", "Priya Raman"),
    ("The minimum password length for admins is 23 characters.", "What is the minimum password length for admins?", "23"),
    ("Invoices from Tessera Ltd are paid within 47 days.", "How quickly are invoices from Tessera Ltd paid?", "47"),
]


def attention_contexts() -> List[Dict[str, Any]]:
    """Fifteen chunks with one needle. Distractors are shuffled per fact, not per position."""
    rng = random.Random(SEED)
    cases = []
    for fact, question, answer in ATTENTION_FACTS:
        distractors = _DISTRACTORS[:]
        rng.shuffle(distractors)
        cases.append({"fact": fact, "question": question, "answer": answer, "distractors": distractors[:14]})
    return cases


def hard_attention_contexts() -> List[Dict[str, Any]]:
    """Long chunks, with distractors that state the same kind of fact about another subject."""
    rng = random.Random(SEED + 3)
    templates = [
        ("hotel", "The maximum hotel rate for {} is SGD {} per night.", "What is the maximum hotel rate for Singapore?", ["Tokyo", "Seoul", "Bangkok", "Jakarta", "Manila", "Hanoi", "Taipei", "Mumbai", "Sydney", "Dubai", "London", "Paris", "Berlin", "Toronto"], "Singapore", 372, (150, 600)),
        ("password", "The minimum password length for {} is {} characters.", "What is the minimum password length for admins?", ["contractors", "interns", "auditors", "developers", "analysts", "vendors", "managers", "kiosks", "printers", "testers", "partners", "trainers", "guests", "operators"], "admins", 23, (8, 40)),
        ("invoice", "Invoices from {} are paid within {} days.", "How quickly are invoices from Tessera Ltd paid?", ["Corvid Ltd", "Halden Ltd", "Marlow Ltd", "Pinion Ltd", "Sable Ltd", "Trident Ltd", "Umbra Ltd", "Vesper Ltd", "Wexford Ltd", "Yarrow Ltd", "Zephyr Ltd", "Argent Ltd", "Bastion Ltd", "Cinder Ltd"], "Tessera Ltd", 47, (10, 90)),
        ("door", "The {} door code changes every {} days.", "How often does the server room door code change?", ["loading dock", "archive room", "mail room", "lab", "boardroom", "rooftop", "garage", "records room", "print room", "kitchen", "lobby", "storage", "gym", "workshop"], "server room", 19, (5, 60)),
    ]
    cases = []
    for _, template, question, others, subject, value, (low, high) in templates:
        def padded(sentence: str) -> str:
            return " ".join(rng.sample(_FILLER, 4)) + " " + sentence + " " + " ".join(rng.sample(_FILLER, 4))

        used = {value}
        distractors = []
        for other in others:
            number = rng.randint(low, high)
            while number in used:
                number = rng.randint(low, high)
            used.add(number)
            distractors.append(padded(template.format(other, number)))
        cases.append({"fact": padded(template.format(subject, value)), "question": question, "answer": str(value), "distractors": distractors})
    return cases
