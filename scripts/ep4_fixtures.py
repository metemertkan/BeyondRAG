"""Test sets for the Episode 4 benchmark. Questions and claims refer to the Retrieval Lab corpus
(config/ep3_demo_docs.json); everything in it is invented."""

from __future__ import annotations

from typing import Any, Dict, List

# One fact per row, stated by a source. Each perturbation breaks it in a different way.
# "other" is a second source that is about something else, for wrong-citation and distractor tests.
CLAIM_FACTS: List[Dict[str, str]] = [
    {"source": "Helios revenue in Europe was EUR 2,760,000 this financial year, with growth driven by enterprise renewals.",
     "other": "Helios revenue in Asia was USD 1,880,000 this financial year, with growth driven by enterprise renewals.",
     "faithful": "Helios earned EUR 2,760,000 in Europe this financial year.",
     "number_swap": "Helios earned EUR 2,670,000 in Europe this financial year.",
     "entity_swap": "Vega earned EUR 2,760,000 in Europe this financial year.",
     "negation": "Helios revenue in Europe was not driven by enterprise renewals.",
     "addition": "Helios earned EUR 2,760,000 in Europe, mostly from the German market."},
    {"source": "The list price is 289 dollars per unit for the AR-7 model, sold under the Aurora name, with volume discounts from 50 units.",
     "other": "The Aurora router launch event drew 400 partners.",
     "faithful": "The AR-7 model lists at 289 dollars per unit.",
     "number_swap": "The AR-7 model lists at 298 dollars per unit.",
     "entity_swap": "The AR-9 model lists at 289 dollars per unit.",
     "negation": "The AR-7 model has no volume discounts.",
     "addition": "The AR-7 model lists at 289 dollars per unit and includes a three year warranty."},
    {"source": "A line manager may approve a single trip of up to HKD 18,000.",
     "other": "Standard laptops are replaced every 42 months.",
     "faithful": "Line managers can approve one business trip of up to HKD 18,000.",
     "number_swap": "Line managers can approve one business trip of up to HKD 81,000.",
     "entity_swap": "Department heads can approve one business trip of up to HKD 18,000.",
     "negation": "Line managers cannot approve business trips.",
     "addition": "Line managers can approve one business trip of up to HKD 18,000, or HKD 25,000 with director sign-off."},
    {"source": "Customer chat history is deleted after 26 months.",
     "other": "Three competing quotes are mandatory for new vendors.",
     "faithful": "Chat history with customers is kept for 26 months, then deleted.",
     "number_swap": "Chat history with customers is kept for 36 months, then deleted.",
     "entity_swap": "Customer email history is deleted after 26 months.",
     "negation": "Customer chat history is never deleted.",
     "addition": "Customer chat history is deleted after 26 months, unless a legal hold applies."},
    {"source": "Token validation calls to the identity provider exceed the 3 second deadline under load, and the client receives HTTP 504.",
     "other": "The load balancer serves an expired leaf certificate after the renewal job fails.",
     "faithful": "Under load, token validation exceeds its 3 second deadline and clients get HTTP 504.",
     "number_swap": "Under load, token validation exceeds its 3 second deadline and clients get HTTP 503.",
     "entity_swap": "Under load, database queries exceed a 3 second deadline and clients get HTTP 504.",
     "negation": "Token validation never exceeds its deadline under load.",
     "addition": "Under load, token validation exceeds its 3 second deadline, which started after the March upgrade."},
    {"source": "Privileged access rights are reviewed every 75 days.",
     "other": "Backup restores are tested every 45 days.",
     "faithful": "Privileged access is reviewed every 75 days.",
     "number_swap": "Privileged access is reviewed every 57 days.",
     "entity_swap": "Standard access is reviewed every 75 days.",
     "negation": "Privileged access rights are not reviewed.",
     "addition": "Privileged access is reviewed every 75 days by the internal audit team."},
    {"source": "The product we launched in Q3 was Helios, after an eight month development cycle and a staged rollout.",
     "other": "The product we launched in Q4 was Lyra, after an eight month development cycle and a staged rollout.",
     "faithful": "Helios was the Q3 launch, after eight months of development.",
     "number_swap": "Helios was the Q3 launch, after eighteen months of development.",
     "entity_swap": "Helios was the Q2 launch, after eight months of development.",
     "negation": "Helios was launched without a staged rollout.",
     "addition": "Helios was the Q3 launch and sold out in its first week."},
    {"source": "Gifts from suppliers may not exceed HKD 640 in value.",
     "other": "Each employee has an annual training budget of HKD 12,500.",
     "faithful": "Supplier gifts are capped at HKD 640.",
     "number_swap": "Supplier gifts are capped at HKD 460.",
     "entity_swap": "Client gifts are capped at HKD 640.",
     "negation": "There is no limit on gifts from suppliers.",
     "addition": "Supplier gifts are capped at HKD 640 and must be declared within 5 days."},
]

PERTURBATIONS: List[str] = ["faithful", "number_swap", "entity_swap", "negation", "addition", "wrong_citation"]


def claim_cases() -> List[Dict[str, Any]]:
    """Each case is one sentence citing S1. S1 is the fact's source and S2 is the other source."""
    cases: List[Dict[str, Any]] = []
    for index, fact in enumerate(CLAIM_FACTS):
        sources = {"S1": fact["source"], "S2": fact["other"]}
        for kind in PERTURBATIONS:
            if kind == "wrong_citation":
                # A correct sentence that cites the wrong source.
                cases.append({"fact": index, "kind": kind, "sentence": fact["faithful"], "cites": ["S2"], "sources": sources, "should_flag": True})
            else:
                cases.append({"fact": index, "kind": kind, "sentence": fact[kind], "cites": ["S1"], "sources": sources, "should_flag": kind != "faithful"})
    return cases


# Questions over the lab corpus. Answerable ones list values the answer must contain.
# Unanswerable ones sit close to something the corpus does cover, which is where models guess.
QUESTIONS: List[Dict[str, Any]] = [
    {"q": "What was Helios revenue in Europe?", "expect": ["2,760,000"]},
    {"q": "What was Orion revenue in Asia?", "expect": ["3,100,000"]},
    {"q": "What is the list price of the AR-7 model?", "expect": ["289"]},
    {"q": "How much can a line manager approve for one business trip?", "expect": ["18,000"]},
    {"q": "How many quotes are needed before onboarding a new vendor?", "expect": ["three", "3"]},
    {"q": "How long is customer chat history kept?", "expect": ["26 months"]},
    {"q": "How often are staff laptops replaced?", "expect": ["42 months"]},
    {"q": "Which product did we launch in Q3?", "expect": ["Helios"]},
    {"q": "What HTTP status does the client receive when token validation times out?", "expect": ["504"]},
    {"q": "How many partners attended the Aurora router launch event?", "expect": ["400"]},
    {"q": "In how many languages is Aurora router support available?", "expect": ["eleven", "11"]},
    {"q": "What should I do about error ERR-1393?", "expect": ["rotate the credential"]},
    {"q": "What was Helios revenue in Africa?", "expect": None},
    {"q": "What is the list price of the AR-9 model?", "expect": None},
    {"q": "How much can a department head approve for one business trip?", "expect": None},
    {"q": "How long are customer emails kept?", "expect": None},
    {"q": "How often are contractor laptops replaced?", "expect": None},
    {"q": "Which product did we launch in Q1 of next year?", "expect": None},
    {"q": "What HTTP status does the client receive when the database times out?", "expect": None},
    {"q": "How many partners attended the Helios launch event?", "expect": None},
    {"q": "What warranty comes with the AR-7 model?", "expect": None},
    {"q": "What should I do about error ERR-1400?", "expect": None},
    {"q": "What was total company revenue across all products this year?", "expect": None},
    {"q": "Who approves exceptions to the vendor onboarding rules?", "expect": None},
]
