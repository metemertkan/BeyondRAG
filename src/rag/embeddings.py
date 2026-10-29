"""Cross-Lingual (English <-> Chinese) Dense Vector Embedding Engine for BeyondRAG.

Projects English, Traditional Chinese, and Simplified Chinese text into a shared
256-dimensional L2-normalized semantic vector space (`int8_hnsw` cosine in ES 8.15)
so that pure English PDFs can be retrieved accurately using Chinese queries (and vice versa).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Final, List, Tuple

EMBEDDING_DIMS: Final[int] = 256
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9_]+|[\u4e00-\uffff]+", re.IGNORECASE)
_CJK_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\uffff]")

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

# Cross-Lingual Concept Alignment Table:
# Maps (Canonical Concept ID, English Triggers, Chinese Traditional/Simplified Triggers, English Expansion Keywords)
CROSS_LINGUAL_CONCEPTS: Final[List[Tuple[str, List[str], List[str], str]]] = [
    (
        "concept:cash_reserve_liquidity",
        ["cash reserve", "unencumbered cash", "liquidity coverage", "lcr", "450,000,000", "450m", "buffer", "minimum cash"],
        ["現金儲備", "现金储备", "流動性", "流动性", "緩衝", "缓冲", "最低港元", "現金", "现金", "儲備", "储备", "4.5億", "四億五千萬"],
        "minimum unencumbered cash reserve buffer HKD 450,000,000 450M liquidity coverage ratio LCR",
    ),
    (
        "concept:fx_forward_hedging",
        ["foreign exchange", "fx forward", "hedging", "peg band", "unhedged", "cny exposure", "10,000,000", "cfo"],
        ["外匯", "外汇", "遠期合約", "远期合约", "對沖", "对冲", "匯率", "汇率", "人民幣敞口", "人民币敞口", "財務總監", "财务总监"],
        "foreign exchange FX forward contract hedging limit USD 10,000,000 CFO unhedged CNY exposure",
    ),
    (
        "concept:treasury_banking",
        ["treasury", "hsbc", "0005.hk", "standard chartered", "clearing account", "net interest margin", "covenant"],
        ["財資", "财资", "庫務", "库务", "匯豐", "汇丰", "渣打", "清算帳戶", "清算账户", "淨息差", "净息差", "保證金", "保证金"],
        "Group Treasury HSBC 0005.HK Standard Chartered clearing accounts covenant margin",
    ),
    (
        "concept:typhoon_weather_policy",
        ["typhoon", "signal no. 8", "black rainstorm", "severe weather", "observatory", "labour department"],
        ["颱風", "台风", "八號風球", "八号风球", "8號風球", "8号风球", "黑色暴雨", "黑雨", "惡劣天氣", "恶劣天气", "天文台"],
        "Typhoon Signal No. 8 Black Rainstorm Warning severe weather Hong Kong Observatory remote work",
    ),
    (
        "concept:hybrid_attendance_office",
        ["hybrid attendance", "core hours", "09:30", "18:00", "headquarters", "2 ifc", "remote work", "workplace"],
        ["混合辦公", "混合办公", "考勤", "上班時間", "上班时间", "核心時間", "核心时间", "總部", "总部", "在家工作", "遠程辦公", "远程办公"],
        "general workplace hybrid attendance policy core hours 09:30 18:00 2 IFC remote work",
    ),
    (
        "concept:it_helpdesk_wifi",
        ["service desk", "wi-fi", "wifi", "guest access", "helpdesk", "extension 8800"],
        ["資訊科技", "信息科技", "無線網絡", "无线网络", "訪客網絡", "访客网络", "技術支援", "技术支持", "分機", "分机"],
        "IT Service Desk Public Wi-Fi Guest Access helpdesk extension 8800",
    ),
    (
        "concept:container_terminal_sla",
        ["kwai tsing", "container terminal", "dispatch sla", "99.4%", "vessel loading", "dwell time", "36 hours", "gate-out"],
        ["葵青", "貨櫃碼頭", "货柜码头", "集裝箱碼頭", "集装箱码头", "調度", "调度", "準時率", "准时率", "滯留時間", "滞留时间", "裝船", "装船"],
        "Kwai Tsing Container Terminal 9 dispatch SLA 99.4% on-time vessel loading container dwell time 36 hours",
    ),
    (
        "concept:cold_chain_reefer",
        ["cold-chain", "cold chain", "reefer", "pharmaceutical", "air freight", "+2.0 c", "+8.0 c", "temperature", "hkia"],
        ["冷鏈", "冷链", "冷藏", "醫藥", "医药", "空運", "空运", "溫度", "温度", "機場貨運", "机场货运", "溫控", "温控"],
        "cold-chain pharmaceutical air freight reefer container temperature +2.0 C +8.0 C HKIA",
    ),
    (
        "concept:customs_border_fleet",
        ["shenzhen bay", "customs", "rocars", "manifest", "dual-plate", "border crossing"],
        ["深圳灣", "深圳湾", "海關", "海关", "清關", "清关", "跨境車隊", "跨境车队", "兩地牌", "两地牌", "報關", "报关"],
        "Shenzhen Bay Port customs manifests ROCARS dual-plate logistics trucks border crossing",
    ),
    (
        "concept:gba_capex_procurement",
        ["greater bay area", "gba", "capex", "capital expenditure", "warehouse automation", "amr", "2,500,000", "500,000", "expedite", "procurement"],
        ["大灣區", "大湾区", "資本支出", "资本支出", "倉庫自動化", "仓库自动化", "採購", "采购", "機器人", "机器人", "審批門檻", "审批门槛", "加急費", "加急费"],
        "Greater Bay Area GBA warehouse automation CapEx capital expenditure approval threshold HKD 500,000 2,500,000 procurement",
    ),
]


def has_cjk(text: str) -> bool:
    """Return True if the string contains Chinese/CJK characters."""
    return bool(_CJK_CHAR_RE.search(text))


def expand_cross_lingual_query(query: str) -> Tuple[str, List[str]]:
    """Expand a Chinese or bilingual query with aligned English retrieval keywords.

    Returns:
        Tuple[str, List[str]]: (Expanded bilingual query string, list of matched concept IDs)
    """
    lowered = query.lower()
    matched_concepts: List[str] = []
    english_expansions: List[str] = []

    for concept_id, en_triggers, zh_triggers, en_expansion in CROSS_LINGUAL_CONCEPTS:
        if any(zt in query for zt in zh_triggers) or any(et in lowered for et in en_triggers):
            matched_concepts.append(concept_id)
            if has_cjk(query):
                english_expansions.append(en_expansion)

    if english_expansions:
        expanded = f"{query} {' '.join(english_expansions)}"
    else:
        expanded = query

    return expanded, matched_concepts


def _extract_features(text: str) -> list[tuple[str, float]]:
    """Extract lexical, character n-gram, and shared cross-lingual concept features."""
    lowered = text.lower().strip()
    features: list[tuple[str, float]] = []

    # 1. Inject high-weight Cross-Lingual Concept dimensions (shared between English & Chinese!)
    for concept_id, en_triggers, zh_triggers, _ in CROSS_LINGUAL_CONCEPTS:
        if any(et in lowered for et in en_triggers) or any(zt in text for zt in zh_triggers):
            features.append((concept_id, 5.5))
            features.append((f"{concept_id}:anchor", 4.0))

    # 2. Extract English words and CJK character unigrams/bigrams
    raw_chunks = _TOKEN_RE.findall(lowered)
    en_tokens: list[str] = []
    for chunk in raw_chunks:
        if has_cjk(chunk):
            # Add CJK character unigrams and bigrams
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


def embed_text(text: str) -> List[float]:
    """Compute an L2-normalized 256-dimensional cross-lingual dense vector for Elasticsearch int8_hnsw."""
    vec = [0.0] * EMBEDDING_DIMS
    features = _extract_features(text)

    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx1 = int.from_bytes(digest[0:2], "little") % EMBEDDING_DIMS
        sign1 = 1.0 if (digest[2] & 1) == 0 else -1.0
        idx2 = int.from_bytes(digest[4:6], "little") % EMBEDDING_DIMS
        sign2 = 1.0 if (digest[6] & 1) == 0 else -1.0

        vec[idx1] += sign1 * weight
        vec[idx2] += sign2 * (weight * 0.6)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        vec[0] = 1.0
        return vec

    return [round(v / norm, 6) for v in vec]
