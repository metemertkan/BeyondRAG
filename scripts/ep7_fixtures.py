"""Questions for scripts/ep7_benchmark.py.

Each base question has an expected fact (a string the right answer contains), two
paraphrases with the same answer, and near-misses: questions that read almost the same but
have a different answer, or none in the documents (``expected`` None). A cache that serves
the base answer to a near-miss has served a wrong answer, unless that answer happens to
contain the near-miss's fact too; the benchmark checks the actual text either way.

The demo documents are in ``src/rag/seeding.py``. Three files from ``eval/corpus`` are added
to Acme (remote work, travel in Chinese, the 2026 password policy).
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

ROLES = ["finance", "operation"]
EXTRA_ACME_DOCS = ["remote_work_policy.md", "travel_policy_zh.md", "it_password_policy_2026.md"]


class NearMiss(TypedDict):
    q: str
    expected: Optional[str]


class Case(TypedDict):
    id: str
    tenant: str
    q: str
    expected: str
    paraphrases: List[str]
    near: List[NearMiss]


CASES: List[Case] = [
    # --- Acme ---------------------------------------------------------------------------
    {"id": "core_hours", "tenant": "acme", "q": "What are the core office hours at the Hong Kong headquarters?", "expected": "09:30",
     "paraphrases": ["When do core hours start at the HK headquarters?", "What is the core working schedule at 2 IFC?"],
     "near": [{"q": "When do core hours end at the Hong Kong headquarters?", "expected": "18:00"}]},
    {"id": "typhoon", "tenant": "acme", "q": "What happens to office attendance when Typhoon Signal No. 8 is hoisted?", "expected": "remote",
     "paraphrases": ["Do we work from home during a T8 typhoon signal?", "What is the office attendance rule under Typhoon Signal 8?"],
     "near": [{"q": "What happens to office attendance when Typhoon Signal No. 3 is hoisted?", "expected": None}]},
    {"id": "helpdesk", "tenant": "acme", "q": "Which extension do I dial for IT hardware support?", "expected": "8800",
     "paraphrases": ["What is the phone extension of the IT service desk?", "How do I call the help desk for hardware problems?"],
     "near": [{"q": "Which extension do I dial for payroll support?", "expected": None}]},
    {"id": "cash_reserve", "tenant": "acme", "q": "What minimum cash reserve must Group Treasury keep?", "expected": "450",
     "paraphrases": ["How much cash must Treasury hold at minimum intraday?", "What is the minimum unencumbered cash buffer for Group Treasury?"],
     "near": [{"q": "What maximum cash reserve may Group Treasury keep?", "expected": None},
              {"q": "What minimum cash reserve must Group Treasury keep in USD?", "expected": None}]},
    {"id": "fx_signoff", "tenant": "acme", "q": "Which FX forward contracts need sign-off from the CFO?", "expected": "10,000,000",
     "paraphrases": ["Above what size does an FX forward require CFO approval?", "When does a foreign exchange forward need the CFO to sign off?"],
     "near": [{"q": "Which FX forward contracts need sign-off from the CEO?", "expected": None}]},
    {"id": "cny_cap", "tenant": "acme", "q": "What is the cap on unhedged CNY exposure?", "expected": "85,000,000",
     "paraphrases": ["How much unhedged CNY exposure is allowed?", "What is the limit for CNY exposure that is not hedged?"],
     "near": [{"q": "What is the cap on unhedged USD exposure?", "expected": None}]},
    {"id": "covenant", "tenant": "acme", "q": "When must the Q4 audited balance sheet and covenant certificate be submitted?", "expected": "5th",
     "paraphrases": ["What is the deadline for the Q4 covenant compliance certificate?", "By when do finance analysts submit the Q4 audited balance sheet?"],
     # Same answer, "the 5th business day of each quarter": a guard that rejects this is too strict.
     "near": [{"q": "When must the Q3 audited balance sheet and covenant certificate be submitted?", "expected": "5th"}]},
    {"id": "dispatch_rate", "tenant": "acme", "q": "What on-time dispatch rate must operations supervisors maintain at Terminal 9?", "expected": "99.4",
     "paraphrases": ["What is the Kwai Tsing Terminal 9 on-time loading target?", "Which on-time gate-out rate is required at Terminal 9?"],
     "near": [{"q": "What on-time dispatch rate must operations supervisors maintain at Terminal 8?", "expected": None}]},
    {"id": "dwell", "tenant": "acme", "q": "How long can a container dwell before a Tier-2 escalation?", "expected": "36",
     "paraphrases": ["After how many hours of dwell time is a container escalated to Tier 2?", "What dwell time triggers the automatic Tier-2 escalation?"],
     "near": [{"q": "How long can a container dwell before a Tier-1 escalation?", "expected": None}]},
    {"id": "reefer_range", "tenant": "acme", "q": "What temperature range must reefer containers at HKIA keep?", "expected": "8.0",
     "paraphrases": ["Between which temperatures must reefers at the airport cargo terminal stay?", "What is the required reefer temperature band at HKIA?"],
     "near": [{"q": "What is the minimum temperature reefer containers at HKIA must keep?", "expected": "2.0"}]},
    {"id": "excursion", "tenant": "acme", "q": "Within how many minutes must a reefer excursion alarm be inspected?", "expected": "15",
     "paraphrases": ["How fast must staff physically inspect a reefer excursion alarm?", "What is the response time for a cold-chain excursion alarm?"],
     "near": [{"q": "Within how many hours must a reefer excursion alarm be reported to the client?", "expected": None}]},
    {"id": "rocars", "tenant": "acme", "q": "How many hours before crossing the border must ROCARS clearance be done?", "expected": "4",
     "paraphrases": ["When must dual-plate trucks complete ROCARS customs clearance?", "How early is ROCARS clearance required before a border crossing?"],
     "near": [{"q": "How many hours after crossing the border must ROCARS clearance be done?", "expected": None}]},
    {"id": "amr_capex", "tenant": "acme", "q": "Who can initiate CapEx requisitions for warehouse robots between HKD 500,000 and HKD 2,500,000?", "expected": "Operations Lead",
     "paraphrases": ["Who may start a capital expenditure request for AMRs in the HKD 500k to 2.5M range?", "Which roles can raise warehouse automation CapEx between HKD 500,000 and 2,500,000?"],
     "near": [{"q": "Who can initiate CapEx requisitions for warehouse robots above HKD 2,500,000?", "expected": None}]},
    {"id": "expedite", "tenant": "acme", "q": "Up to what amount are expedited air-charter surcharges pre-authorized?", "expected": "350,000",
     "paraphrases": ["What is the pre-authorized limit for emergency air-charter expedite fees?", "How much expedite surcharge is pre-approved under the contingency budget?"],
     "near": [{"q": "Above what amount do expedited air-charter surcharges need CFO approval?", "expected": None}]},
    {"id": "remote_days", "tenant": "acme", "q": "How many days per week may staff work remotely?", "expected": "3",
     "paraphrases": ["What is the weekly limit on remote working days?", "How many remote days a week are allowed?"],
     "near": [{"q": "How many days per week may staff work remotely without manager approval?", "expected": None}]},
    {"id": "abroad_days", "tenant": "acme", "q": "How many days per year may staff work remotely from outside Hong Kong?", "expected": "20",
     "paraphrases": ["What is the annual cap on working remotely abroad?", "How many working days a year can I work from outside HK?"],
     "near": [{"q": "How many days per year may staff work remotely from inside Hong Kong?", "expected": None}]},
    {"id": "collab_hours", "tenant": "acme", "q": "What are the core collaboration hours for remote staff?", "expected": "10:00",
     "paraphrases": ["When must remote workers be available for collaboration?", "Which hours are core collaboration time under the remote work policy?"],
     "near": [{"q": "What are the core collaboration hours for remote staff in London?", "expected": None}]},
    {"id": "hotel_cap", "tenant": "acme", "q": "What is the hotel cap per night on business trips?", "expected": "1,800",
     "paraphrases": ["How much can I spend per night on accommodation when travelling for work?", "What is the nightly lodging limit for business travel?"],
     "near": [{"q": "What is the meal cap per day on business trips?", "expected": None}]},
    {"id": "claim_deadline", "tenant": "acme", "q": "Within how many days after a trip must a travel claim be submitted?", "expected": "14",
     "paraphrases": ["What is the deadline for submitting travel expense claims?", "How long do I have to file my travel reimbursement after the trip?"],
     "near": [{"q": "Within how many days before a trip must a travel request be submitted?", "expected": None}]},
    {"id": "password", "tenant": "acme", "q": "How often must employees rotate their network password?", "expected": "180",
     "paraphrases": ["What is the password rotation period?", "How many days between required network password changes?"],
     "near": [{"q": "How often did the 2023 policy require employees to rotate their network password?", "expected": None}]},
    {"id": "mfa", "tenant": "acme", "q": "Is multi-factor authentication mandatory for staff?", "expected": "mandatory",
     "paraphrases": ["Do all employees have to use MFA?", "Is phishing-resistant multi-factor authentication required?"],
     "near": [{"q": "Is multi-factor authentication not required for contractors?", "expected": None}]},
    # --- Globex -------------------------------------------------------------------------
    {"id": "g_core_hours", "tenant": "globex", "q": "What are the core office hours at the Rotterdam head office?", "expected": "09:00",
     "paraphrases": ["When do core hours start at the Waalhaven office?", "What is the core working schedule in Rotterdam?"],
     "near": [{"q": "When do core hours end at the Rotterdam head office?", "expected": "17:30"}]},
    {"id": "g_cash_reserve", "tenant": "globex", "q": "What minimum cash reserve must the Globex Treasury team keep?", "expected": "38",
     "paraphrases": ["How much cash must Globex Treasury hold at minimum intraday?", "What is Globex's minimum unencumbered cash buffer?"],
     "near": [{"q": "What maximum cash reserve may the Globex Treasury team keep?", "expected": None}]},
    {"id": "g_fx", "tenant": "globex", "q": "Which FX forward contracts need CFO sign-off at Globex?", "expected": "4,000,000",
     "paraphrases": ["Above what size does a Globex FX forward need the CFO?", "When must the CFO approve an FX forward contract at Globex?"],
     "near": [{"q": "Which FX forward contracts need board sign-off at Globex?", "expected": None}]},
    {"id": "g_usd_cap", "tenant": "globex", "q": "What is the cap on unhedged USD exposure at Globex?", "expected": "9,500,000",
     "paraphrases": ["How much unhedged USD exposure may Globex carry?", "What is Globex's limit on USD exposure that isn't hedged?"],
     "near": [{"q": "What is the cap on unhedged CNY exposure at Globex?", "expected": None}]},
    {"id": "g_dwell", "tenant": "globex", "q": "How long can a container dwell at Maasvlakte before escalation?", "expected": "48",
     "paraphrases": ["After how many hours of dwell time is a Maasvlakte container escalated?", "What dwell time triggers escalation to the Head of Operations?"],
     "near": [{"q": "How long can a container dwell at Maasvlakte before it is auctioned?", "expected": None}]},
    {"id": "g_dispatch", "tenant": "globex", "q": "What on-time dispatch rate is required at the Maasvlakte terminal?", "expected": "97.5",
     "paraphrases": ["What is the Maasvlakte on-time loading target?", "Which on-time gate-out rate must Globex supervisors maintain?"],
     "near": [{"q": "What on-time dispatch rate is required at the Rotterdam airport terminal?", "expected": None}]},
    {"id": "g_reefer", "tenant": "globex", "q": "What temperature range must reefers at Schiphol Cargo keep?", "expected": "6.0",
     "paraphrases": ["Between which temperatures must Globex reefer containers stay at Schiphol?", "What is the reefer temperature band at Schiphol Cargo?"],
     "near": [{"q": "What is the minimum temperature reefers at Schiphol Cargo must keep?", "expected": "2.0"}]},
    {"id": "g_excursion", "tenant": "globex", "q": "Within how many minutes must a Globex excursion alarm be inspected?", "expected": "20",
     "paraphrases": ["How quickly must Globex staff inspect a reefer excursion alarm?", "What is Globex's response time for cold-chain excursion alarms?"],
     "near": [{"q": "Within how many hours must a Globex excursion alarm be reported to the insurer?", "expected": None}]},
    {"id": "g_helpdesk", "tenant": "globex", "q": "Which extension do Globex employees dial for hardware support?", "expected": "4100",
     "paraphrases": ["What is the Globex IT service desk extension?", "How do I call Globex IT for hardware issues?"],
     "near": [{"q": "Which extension do Globex employees dial for HR support?", "expected": None}]},
]


def all_questions() -> List[Dict[str, object]]:
    """Every unique question with its case, kind and expected fact."""
    rows: List[Dict[str, object]] = []
    for case in CASES:
        rows.append({"case": case["id"], "tenant": case["tenant"], "kind": "base", "q": case["q"], "expected": case["expected"]})
        for text in case["paraphrases"]:
            rows.append({"case": case["id"], "tenant": case["tenant"], "kind": "paraphrase", "q": text, "expected": case["expected"]})
        for near in case["near"]:
            rows.append({"case": case["id"], "tenant": case["tenant"], "kind": "near_miss", "q": near["q"], "expected": near["expected"]})
    return rows
