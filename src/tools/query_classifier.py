"""Deterministic query intent classifier.

Replaces the gpt-4o router + tool-selector LLM calls with a fast, multi-layered
local classifier. This removed 2 LLM round-trips (~3–6 seconds) for standard
regulations.

Design principles
-----------------
- Waterfall Logic — layers run sequentially until a high-confidence match is found.
- Hybrid Architecture — Uses deterministic regex for speed (Layer 1-3) and 
  semantic embeddings/GPT-4o-mini for fallback (Layer 4-5).
- Word-boundary keyword matching — prevents false positives from substrings.
- Confidence scoring — every result carries a 0.0–1.0 confidence value.
- Async Cache — identical queries are free after the first classification.
- Telemetry — provides visibility into which layer triggered the result.

Note on performance:
Layers 1-3 are extremely fast (< 1ms). Layers 4 and 5 involve network I/O
(OpenAI API) and contribute ~200-500ms of latency.
"""

from __future__ import annotations

import re
import unicodedata
import json
import os
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict


# ── Intent Enum ───────────────────────────────────────────────────────────────


class QueryIntent(str, Enum):
    """Classified intent for an incoming user query."""

    CLAUSE_LOOKUP = "clause_lookup"  # Specific clause reference
    REGULATION_SEARCH = "regulation_search"  # General regulatory question
    OUT_OF_SCOPE = "out_of_scope"  # Non-regulatory query


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Matches explicit clause references, e.g.:
#   FAR 52.236-2  |  DFARS 252.204-7012  |  EM 385 1-1  |  EM385 05.A
#   48 CFR 52.212-4  |  HSAR 3052.209-70  |  OSHA 1926.502  |  DEAR 970.5204-2
#   VAAR 852.219-10  |  NFS 1852.223-70   |  HHSAR 352.270-1
_CLAUSE_PATTERN = re.compile(
    r"""
    \b
    (
        48\s*CFR            |   # Title 48 of the Code of Federal Regulations
        FAR                 |   # Federal Acquisition Regulation
        DFARS               |   # Defense FAR Supplement
        DFARSPGI            |   # DFARS Procedures, Guidance, and Information
        HSAR                |   # Homeland Security AR Supplement
        DEAR                |   # Dept of Energy AR Supplement
        VAAR                |   # Veterans Affairs AR Supplement
        NFS                 |   # NASA FAR Supplement
        HHSAR               |   # Health and Human Services AR Supplement
        AFARS               |   # Army FAR Supplement
        NMCARS              |   # Navy Marine Corps AR Supplement
        DLAD                |   # Defense Logistics Agency Directive
        OSHA                |   # Occupational Safety and Health regulations
        EM[\s\-]*385            # Army Corps Safety Manual
    )
    [\s\-,]*
    (?:section|part|clause)?
    [\s\-,]*
    (\d+(?:[\.\-\(\/][\d\-A-Za-z\(\)\.\/]+)?)   # Clause/section number (fixed)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches a regulatory source name anywhere in the query (without a clause number)
_SOURCE_PATTERN = re.compile(
    r"""
    \b
    (
        48\s*CFR    |
        FAR         |
        DFARS       |
        DFARSPGI    |
        HSAR        |
        DEAR        |
        VAAR        |
        NFS         |
        HHSAR       |
        AFARS       |
        NMCARS      |
        DLAD        |
        OSHA        |
        EM[\s\-]*385
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pre-compiled word-boundary patterns for each keyword (avoids rebuilding per call)
# Using word boundaries prevents "part" matching "apartment", "cost" matching "forecast", etc.
_REGULATION_KEYWORDS: tuple[str, ...] = (
    # Core acquisition / contracting
    "acquisition",
    "award",
    "bid",
    "bidder",
    "bidding",
    "clause",
    "competition",
    "competitive",
    "compliance",
    "contracting",
    "contractor",
    "contract",
    "cost",
    "cost-reimbursement",
    "cost-plus",
    "debarment",
    "default",
    "delivery",
    # Domain-specific procurement terms
    "earnest",
    "evaluation",
    "federal",
    "grant",
    "guaranty",
    "indemnification",
    "invoice",
    "invoicing",
    "labor",
    "labour",
    "liquidated damages",
    "modification",
    "negotiation",
    "offeror",
    "option",
    "penalty",
    "performance bond",
    "procurement",
    "proposal",
    "provision",
    "purchase order",
    "regulation",
    "requirement",
    "rfp",
    "rfq",
    "rfi",
    "sealed bid",
    "section",
    "solicitation",
    "subcontract",
    "subcontractor",
    "surety",
    "suspension",
    # Construction / safety specific
    "asbestos",
    "confined space",
    "construction",
    "fall protection",
    "hazardous",
    "lockout",
    "noise exposure",
    "ozone",
    "ppe",
    "radiation",
    "respiratory",
    "rigging",
    "safety",
    "scaffold",
    "silica",
    "trench",
    # Small business / diversity
    "8a",
    "dbe",
    "disadvantaged",
    "hubzone",
    "sba",
    "sdvosb",
    "small business",
    "set-aside",
    "set-asides",
    "veteran-owned",
    "vosb",
    "wosb",
    # Government structure
    "appropriation",
    "authorization",
    "clin",
    "defense",
    "department",
    "executive order",
    "far part",
    "fiscal",
    "government",
    "part",
    "prime contractor",
    "program",
    "project",
    "standard",
    "subpart",
    "title",
    # FAR-specific topic anchors
    "buy american",
    "domestic preference",
    "country of origin",
    "trade agreements act",
    # Construction / contract admin long-tail (reduce over-refusal)
    "rea",
    "request for equitable adjustment",
    "equitable adjustment",
    "change order",
    "change orders",
    "cpars",
    "past performance",
    "debrief",
    "debriefing",
    "daily rate",
    "field office overhead",
    "overhead",
    "certify",
    "certification",
    "rea certification",
    "novation",
    "novation agreement",
    "withhold",
    "withholding",
    "withholdings",
    "termination for default",
    "termination for convenience",
    "protest",
    "bid protest",
    "gao protest",
    "agency protest",
    "product substitution",
    "product variance",
    "substitution",
    "variance",
    "mobilization",
    "mobilisation",
    "post award",
    "post-award",
    "compensable delay",
    "concurrent delay",
    "excusable delay",
    "delay letter",
    "serial letter",
    "structure the serial letter",
    "ko letter",
    "differing site conditions",
    "differing site condition",
    "site conditions",
    "duct bank",
    "gas line",
    "oil tank",
    "unexploded ordnance",
    "uxo",
    "wildfire",
    "fire at site",
    "fire overnight",
    "stop-work",
    "qc checklist",
    "quality checklist",
    "qc plan",
    "qc program",
    "qcm",
    "quality control manager",
    "conduit",
    "masonry",
    "demolition",
    "phase inspection",
    "inspection report",
    "submittal",
    "submittals",
    "rfi",
    "rfis",
    "hot work",
    "hot work permit",
    "redline",
    "redline drawing",
    "as-built",
    "as built",
    "superintendent",
    "drawing",
    "drawings",
    "update drawing",
    "limited rights",
    "government purpose rights",
    "unlimited rights",
    "data rights",
    "scope",
    "solicitation",
    "scope of work",
    "scope alignment",
    "dispute",
    "disputes",
    "appeal",
    "disagreement",
    "bonding",
    "performance bond",
    "payment bond",
    "contracting officer representative",
    "cor",
    "contracting officer",
    "daily report",
    "daily reports",
    "prime",
    "prime contractor",
    "subcontractor payment",
    "paying",
    "siop",
    "ppi",
    "pre-purchase inspection",
    "pre purchase inspection",
    "insurance",
    "insurance requirements",
    "small business set-aside",
    "set aside",
    "set-aside",
    "off site storage",
    "off-site storage",
    "offsite storage",
    "stored materials",
    "materials stored offsite",
    "bill of materials",
    "insurance certificate",
    "bond certificate",
    "storage location",
)


def _build_keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compile a keyword pattern with strict boundaries and plural support."""
    parts = keyword.strip().split()
    token_patterns: list[str] = []
    for token in parts:
        if re.fullmatch(r"[A-Za-z]+", token):
            token_patterns.append(rf"{re.escape(token)}(?:s|es)?")
        else:
            token_patterns.append(re.escape(token))
    return re.compile(r"\b" + r"\s+".join(token_patterns) + r"\b", re.IGNORECASE)


_KW_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kw, _build_keyword_pattern(kw)) for kw in _REGULATION_KEYWORDS
)


_DOMAIN_HINT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # FAR Part 25 topic hint when acronym is omitted.
    (re.compile(r"\bbuy\s+american\b", re.IGNORECASE), "FAR"),
)


# ── Procedural / contract-CO flags (single source of truth for pipeline) ───────

_PROCEDURAL_TRIGGERS: tuple[str, ...] = (
    "what do i do",
    "what do we do",
    "what should i do",
    "what can i do",
    "next steps",
    "how do i",
    "how do you",
    "how should i",
    "can i get reimbursed",
    "get reimbursed",
    "reimbursed for",
    "what documents do i need",
    "what documents must",
    "what should i verify",
    "write a letter",
    "draft a letter",
    "draft me",
    "write me an",
    "notify the",
    "notify the ko",
    "notify the contracting",
    "appeal",
    "how do you appeal",
    "first 10 steps",
    "first 5 steps",
    "provide the first",
    "encountered",
    "discovered",
    "ran into",
    "uncovered",
    "we have discovered",
    "fire overnight",
    "fire at the site",
    "fire at site",
    "what does the government need",
    "what does the govt need",
    "request for equitable",
    "submit an rea",
    "submit a rea",
    "structure the serial letter",
    "how should i structure",
    "what do i do next",
    "what do we do if",
    "disagreement",
    "do not agree",
    "dispute",
)

_CONTRACT_CO_TRIGGERS: tuple[str, ...] = (
    "how often",
    "how frequently",
    "when do i submit",
    "when do i send",
    "daily report",
    "daily reports",
    "frequency of report",
    "reporting frequency",
    "schedule of report",
    "when are reports due",
    "how many times",
)

_COMPARISON_PATTERN = re.compile(
    r"\bvs\.?\b|versus\b|difference(?:s)?\s+between\b|compar(?:e|ison)\s+(?:of|between)\b"
    r"|how\s+(?:does?|do)\s+.{2,40}\s+differ\b|distinguish\s+between\b",
    re.IGNORECASE,
)

_CONSTRUCTION_LIFECYCLE_TRIGGERS: tuple[str, ...] = (
    "commissioning",
    "pre-commissioning",
    "recommissioning",
    "punchlist",
    "punch list",
    "punch-list",
    "substantial completion",
    "beneficial occupancy",
    "final inspection",
    "pre-final inspection",
    "functional testing",
    "performance testing",
    "system testing",
    "acceptance testing",
    "operational testing",
    "turnover",
    "closeout",
    "close-out",
    "close out",
    "warranty period",
    "warranty inspection",
    "as-built",
    "as built",
    "red-line",
    "redline",
    "attic stock",
    "spare parts turnover",
    "o&m manual",
    "operations and maintenance manual",
    "building automation",
    "bas commissioning",
    "fire alarm testing",
    "elevator testing",
    "balancing",
    "tab report",
    "testing adjusting balancing",
)

_DOCUMENT_REQUEST_TRIGGERS: tuple[str, ...] = (
    "write me a",
    "write me an",
    "draft me a",
    "draft me an",
    "draft a ",
    "write a ",
    "generate a ",
    "generate me a ",
    "create a ",
    "prepare a ",
    "serial letter",
    "write the ko",
    "write a letter",
    "draft a letter",
    "draft an rea",
    "write an rea",
    "request for equitable",
    "rea for ",
    "rfi ",
    "write an rfi",
    "draft an rfi",
    "generate a form",
    "generate a checklist",
    "stop-work order",
    "inspection report",
    "phase inspection",
    "also include",
    "add clause",
    "revise the letter",
    "update the letter",
    "amend the letter",
    "include this clause",
)

# Checklist/form phrasing: these are document requests but go to synthesis (checklist content), not letter_drafter.
_CHECKLIST_REQUEST_PHRASES: tuple[str, ...] = (
    "generate a checklist",
    "create a checklist",
    "prepare a checklist",
    "draft a checklist",
    "write a checklist",
    "inspection checklist",
)
_FORM_REQUEST_PHRASES: tuple[str, ...] = (
    "generate a form",
    "create a form",
    "prepare a form",
    "draft a form",
    "write a form",
)

_LETTER_AMENDMENT_PATTERN = re.compile(
    r"\b(also include|add clause|amend|revise|update|include this clause)\b.*\b(letter|rea|rfi)\b",
    re.IGNORECASE,
)


def get_document_request_type(query: str | None) -> Optional[str]:
    """Classify document request into letter | checklist | form. Only letter goes to letter_drafter."""
    if not query or not query.strip():
        return None
    q = query.strip().lower()
    if any(p in q for p in _CHECKLIST_REQUEST_PHRASES):
        return "checklist"
    if any(p in q for p in _FORM_REQUEST_PHRASES):
        return "form"
    if _LETTER_AMENDMENT_PATTERN.search(q):
        return "letter"
    if any(t in q for t in _DOCUMENT_REQUEST_TRIGGERS):
        return "letter"
    return None


def is_procedural_query(query: str | None) -> bool:
    """True if the query asks for steps / what to do. Single source of truth for pipeline."""
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    return any(t in q for t in _PROCEDURAL_TRIGGERS)


def is_contract_co_query(query: str | None) -> bool:
    """True if the query is about frequency/schedule often specified in contract/CO. Single source of truth."""
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    return any(t in q for t in _CONTRACT_CO_TRIGGERS)


def is_document_request_query(query: str | None) -> bool:
    """True if the query asks to draft/generate a document (letter only). Checklist/form use synthesis, not letter_drafter."""
    return get_document_request_type(query) == "letter"


def is_comparison_query(query: str | None) -> bool:
    """True if the query asks to compare two concepts (e.g. 'REA vs change order')."""
    if not query or not query.strip():
        return False
    return bool(_COMPARISON_PATTERN.search(query.strip()))


def is_construction_lifecycle_query(query: str | None) -> bool:
    """True if the query is about construction lifecycle phases (commissioning, punchlist, closeout, etc.)."""
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    return any(t in q for t in _CONSTRUCTION_LIFECYCLE_TRIGGERS)


_SCHEDULE_RISK_RE = re.compile(
    r"\b(schedule|delay|time\s+extension|liquidated\s+damages?)\b", re.IGNORECASE
)
_RISK_RE = re.compile(
    r"\b(risk|entitlement|compensable|excusable|concurrent|impact|mitigation)\b",
    re.IGNORECASE,
)


def is_schedule_risk_query(query: str | None) -> bool:
    """True if the query involves schedule/delay risk analysis."""
    if not query or not query.strip():
        return False
    q = query.strip()
    return bool(_SCHEDULE_RISK_RE.search(q) and _RISK_RE.search(q))


# ── Source normalisation ───────────────────────────────────────────────────────

_SOURCE_NORM: dict[str, str] = {
    "FAR": "FAR",
    "DFARS": "DFARS",
    "DFARSPGI": "DFARSPGI",
    "48 CFR": "48 CFR",
    "48CFR": "48 CFR",
    "EM385": "EM385",
    "EM 385": "EM385",
    "HSAR": "HSAR",
    "DEAR": "DEAR",
    "VAAR": "VAAR",
    "NFS": "NFS",
    "HHSAR": "HHSAR",
    "AFARS": "AFARS",
    "NMCARS": "NMCARS",
    "DLAD": "DLAD",
    "OSHA": "OSHA",
}


def _normalise_source(raw: str) -> str:
    """Normalise matched source string to canonical label."""
    # Special handle for EM-385 variations
    key = raw.strip().upper()
    if "EM" in key and "385" in key:
        return "EM385"
    key = re.sub(r"\s+", " ", key)
    return _SOURCE_NORM.get(key, key)


def _normalize_query(query: str) -> str:
    """Unicode-normalize, collapse whitespace, strip leading/trailing space."""
    # NFC normalization handles smart quotes, ligatures, etc.
    text = unicodedata.normalize("NFC", query)
    return re.sub(r"\s+", " ", text).strip()


def _infer_regulation_hint(query: str) -> Optional[str]:
    """Infer likely regulation type from known domain-topic anchors."""
    for pattern, regulation in _DOMAIN_HINT_PATTERNS:
        if pattern.search(query):
            return regulation
    return None


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassificationResult:
    """Immutable result returned by classify_query()."""

    intent: QueryIntent
    confidence: float = 0.0  # 0.0–1.0; see module docstring
    clause_reference: Optional[str] = None  # e.g. "FAR 52.236-2"
    clause_source: Optional[str] = None  # e.g. "FAR"
    clause_number: Optional[str] = None  # e.g. "52.236-2"
    regulation_type: Optional[str] = None  # coerced source label
    matched_keywords: list[str] = field(default_factory=list)
    # Pipeline flags: set for in-scope queries so retrieval/synthesis use one source of truth
    is_procedural: bool = False  # steps / what to do
    is_contract_co: bool = False  # frequency/schedule → contract/CO
    is_document_request: bool = (
        False  # True only for letter-type; checklist/form use synthesis
    )
    document_request_type: Optional[str] = (
        None  # "letter" | "checklist" | "form" | None
    )
    is_comparison: bool = False  # "REA vs change order", "type 1 vs type 2"
    is_construction_lifecycle: bool = (
        False  # commissioning, punchlist, closeout, testing
    )
    is_schedule_risk: bool = False  # schedule/delay risk analysis

    @property
    def is_clause_lookup(self) -> bool:
        return self.intent == QueryIntent.CLAUSE_LOOKUP

    @property
    def is_regulation_search(self) -> bool:
        return self.intent == QueryIntent.REGULATION_SEARCH

    @property
    def is_out_of_scope(self) -> bool:
        return self.intent == QueryIntent.OUT_OF_SCOPE


# ── Main classifier ────────────────────────────────────────────────────────────

# ── Global Cache for Centroids ───────────────────────────────────────────────

_CENTROIDS: Optional[dict[str, np.ndarray]] = None


def _load_centroids() -> Optional[dict[str, np.ndarray]]:
    """Lazy-load semantic centroids from disk."""
    global _CENTROIDS
    if _CENTROIDS is not None:
        return _CENTROIDS

    path = os.path.join(os.path.dirname(__file__), "intent_centroids.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
            _CENTROIDS = {k: np.array(v) for k, v in data.items()}
        return _CENTROIDS
    except Exception:
        return None


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# ── Out-of-scope product/feature patterns (before keyword match) ───────────────
# Queries about the product/system, not FAR/DFARS/EM385 — refuse early.
_OUT_OF_SCOPE_SYSTEM_PATTERNS: tuple[str, ...] = (
    "document generator",
    "process guidance agent",
    "why can't i access",
    "is the document generator",
    "when will the document generator",
    "why is the document generator",
    "access the process guidance",
    "recommend a lawyer",
    "recommend an attorney",
    "recommend a government contracts attorney",
    "find me a lawyer",
    "find me an attorney",
    "find an attorney",
    "export to word",
    "export to pdf",
    "export to excel",
    "export the response",
    "save as word",
    "save as pdf",
)


def _is_system_or_product_query(query: str) -> bool:
    """True only for product/feature questions with no regulatory signal."""
    if not query or not query.strip():
        return False
    q = query.strip()
    q_lower = q.lower()
    has_product_signal = any(p in q_lower for p in _OUT_OF_SCOPE_SYSTEM_PATTERNS)
    if not has_product_signal:
        return False

    # Do not force out-of-scope when the query clearly asks about regulations.
    has_clause = bool(_CLAUSE_PATTERN.search(q))
    has_source = bool(_SOURCE_PATTERN.search(q))
    has_keyword = any(pattern.search(q) for _, pattern in _KW_PATTERNS)
    return not (has_clause or has_source or has_keyword)


# ── Waterfall Layers ─────────────────────────────────────────────────────────

_ASYNC_CACHE: Dict[str, ClassificationResult] = {}


async def _get_semantic_intent(query: str) -> Tuple[QueryIntent, float]:
    """Layer 4: Semantic similarity check via embeddings."""
    from src.tools.llm_tools import get_embedding

    centroids = _load_centroids()
    if not centroids:
        return QueryIntent.OUT_OF_SCOPE, 0.0

    try:
        query_emb = np.array(get_embedding(query))
        reg_sim = _cosine_similarity(query_emb, centroids["REGULATION_SEARCH"])
        oos_sim = _cosine_similarity(query_emb, centroids["OUT_OF_SCOPE"])

        # If it's clearly regulatory or at least more regulatory than random
        if reg_sim > oos_sim and reg_sim > 0.82:
            return QueryIntent.REGULATION_SEARCH, reg_sim
        return QueryIntent.OUT_OF_SCOPE, oos_sim
    except Exception:
        return QueryIntent.OUT_OF_SCOPE, 0.0


async def _extract_clause_llm(query: str) -> Optional[Tuple[str, str]]:
    """Layer 5: Micro-LLM fallback for messy extraction."""
    try:
        from src.tools.llm_tools import client

        prompt = f"""Extract the regulation source (FAR/DFARS/EM385/etc.) and the specific section/clause number from this query.
Query: "{query}"

Respond ONLY with a JSON object: {{"source": "...", "number": "..."}} or null if no regulation is mentioned."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=60,
            temperature=0,
        )
        res_text = response.choices[0].message.content
        data = json.loads(res_text or "null")
        if data and data.get("source") and data.get("number"):
            return str(data["source"]).upper(), str(data["number"])
    except Exception:
        pass
    return None


async def classify_query(query: Optional[str]) -> ClassificationResult:
    """Classify a user query with a robust multi-layered (HIC) waterfall.

    Waterfall Logic:
      1. Layer 1: Enhanced Regex Clause Extract (1.0)
      2. Layer 2: Deterministic Source Match (0.8)
      3. Layer 3: Deterministic Keyword Match (0.6)
      4. Layer 4: Semantic Intent Embedding (0.82+)
      5. Layer 5: Micro-LLM Fallback (0.9 - gated)
    """
    if not query or not query.strip():
        return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE, confidence=0.0)

    normalised = _normalize_query(query)
    proc = is_procedural_query(normalised)
    co = is_contract_co_query(normalised)
    doc_type = get_document_request_type(normalised)
    doc = (
        doc_type == "letter"
    )  # only letter-type goes to letter_drafter; checklist/form use synthesis
    comp = is_comparison_query(normalised)
    const = is_construction_lifecycle_query(normalised)
    sched_risk = is_schedule_risk_query(normalised)

    # ── 0. Cache Check ───────────────────────────────────────────────────────────
    if normalised in _ASYNC_CACHE:
        return _ASYNC_CACHE[normalised]

    # ── 1. Layer 1: Exact clause reference (1.0) ───────────────────────────────────
    match = _CLAUSE_PATTERN.search(normalised)
    if match:
        raw_source = match.group(1)
        clause_num = match.group(2).rstrip("-")
        source = _normalise_source(raw_source)
        ref = f"{source} {clause_num}"
        res = ClassificationResult(
            intent=QueryIntent.CLAUSE_LOOKUP,
            confidence=1.0,
            clause_reference=ref,
            clause_source=source,
            clause_number=clause_num,
            regulation_type=source,
            is_procedural=proc,
            is_contract_co=co,
            is_document_request=doc,
            document_request_type=doc_type,
            is_comparison=comp,
            is_construction_lifecycle=const,
            is_schedule_risk=sched_risk,
        )
        _ASYNC_CACHE[normalised] = res
        return res

    # ── 1.5. Out-of-scope: product/system queries (not regulations) ──────────────
    if _is_system_or_product_query(normalised):
        res = ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE, confidence=0.0)
        _ASYNC_CACHE[normalised] = res
        return res

    # ── 2. Layer 2: Deterministic Source Match (0.8) ───────────────────────────────
    source_match = _SOURCE_PATTERN.search(normalised)
    if source_match:
        source = _normalise_source(source_match.group(1))
        res = ClassificationResult(
            intent=QueryIntent.REGULATION_SEARCH,
            confidence=0.8,
            regulation_type=source,
            is_procedural=proc,
            is_contract_co=co,
            is_document_request=doc,
            document_request_type=doc_type,
            is_comparison=comp,
            is_construction_lifecycle=const,
            is_schedule_risk=sched_risk,
        )
        _ASYNC_CACHE[normalised] = res
        return res

    # ── 3. Layer 3: Deterministic Keyword Match (0.6) ────────────────────────────
    matched_kws = []
    for kw, pattern in _KW_PATTERNS:
        if pattern.search(normalised):
            matched_kws.append(kw)

    if matched_kws:
        res = ClassificationResult(
            intent=QueryIntent.REGULATION_SEARCH,
            confidence=0.6,
            matched_keywords=matched_kws,
            regulation_type=_infer_regulation_hint(normalised),
            is_procedural=proc,
            is_contract_co=co,
            is_document_request=doc,
            document_request_type=doc_type,
            is_comparison=comp,
            is_construction_lifecycle=const,
            is_schedule_risk=sched_risk,
        )
        _ASYNC_CACHE[normalised] = res
        return res

    # ── 3.5. Question frame + acquisition/construction hook (0.55) ─────────────────
    # Catches "What is X?", "How do I X?", "If I have X can I...?" when X is contract-related.
    # Robust: one layer for in-scope boundary; no per-query trigger lists.
    _question_start = re.compile(
        r"^(what|how|when|where|why|can|should|do i|do we|does the|did the|is there|are there|if\s+i\s|if\s+we\s)\b",
        re.IGNORECASE,
    )
    _acquisition_hook = re.compile(
        r"\b(contract|contractor|government|ko|far|dfars|em\s*385|clause|requirement|"
        r"schedule|payment|invoice|award|modification|order|letter|report|document|"
        r"delay|cost|price|bond|subcontract|prime|specification|specifications|drawing|"
        r"compensable|concurrent|vendor)\b",
        re.IGNORECASE,
    )
    if _question_start.search(normalised) and _acquisition_hook.search(normalised):
        res = ClassificationResult(
            intent=QueryIntent.REGULATION_SEARCH,
            confidence=0.55,
            regulation_type=_infer_regulation_hint(normalised),
            is_procedural=proc,
            is_contract_co=co,
            is_document_request=doc,
            document_request_type=doc_type,
            is_comparison=comp,
            is_construction_lifecycle=const,
            is_schedule_risk=sched_risk,
        )
        _ASYNC_CACHE[normalised] = res
        return res

    # ── 4. Layer 4: Semantic Intent (0.82+) ──────────────────────────────────────
    sem_intent, sem_conf = await _get_semantic_intent(normalised)
    if sem_intent == QueryIntent.REGULATION_SEARCH:
        res = ClassificationResult(
            intent=QueryIntent.REGULATION_SEARCH,
            confidence=sem_conf,
            is_procedural=proc,
            is_contract_co=co,
            is_document_request=doc,
            document_request_type=doc_type,
            is_comparison=comp,
            is_construction_lifecycle=const,
            is_schedule_risk=sched_risk,
        )
        _ASYNC_CACHE[normalised] = res
        return res

    # ── 5. Layer 5: Micro-LLM Fallback (0.9 - Gated) ────────────────────────────
    # Heuristic Gate: Only trigger LLM if there is a "regulatory signal"
    # (mentions of clause/section/part or numbers after source-like strings)
    # to avoid latency on obvious OOS queries.
    regulatory_signals = {
        "clause",
        "section",
        "part",
        "subpart",
        "article",
        "provision",
    }
    has_signal = any(s in normalised.lower() for s in regulatory_signals)

    # Refined has_digit: Look for a number with a period/dash/paren (regulation-like)
    # or a number in a query that ALREADY matches a source name (handled by Layer 2).
    # This specifically addresses the "I need 3 examples" false positive.
    has_likely_number = bool(re.search(r"\d+[\.\-\(]", normalised))

    if has_signal or has_likely_number:
        llm_match = await _extract_clause_llm(normalised)
        if llm_match:
            source, num = llm_match
            norm_source = _normalise_source(source)
            res = ClassificationResult(
                intent=QueryIntent.CLAUSE_LOOKUP,
                confidence=0.9,
                clause_reference=f"{norm_source} {num}",
                clause_source=norm_source,
                clause_number=num,
                regulation_type=norm_source,
                is_procedural=proc,
                is_contract_co=co,
                is_document_request=doc,
                document_request_type=doc_type,
                is_comparison=comp,
                is_construction_lifecycle=const,
                is_schedule_risk=sched_risk,
            )
            _ASYNC_CACHE[normalised] = res
            return res

    # ── 6. Out of scope ────────────────────────────────────────────────────────────
    res = ClassificationResult(
        intent=QueryIntent.OUT_OF_SCOPE,
        confidence=0.0,
        is_document_request=doc,
        document_request_type=doc_type,
        is_comparison=comp,
        is_construction_lifecycle=const,
        is_schedule_risk=sched_risk,
    )
    _ASYNC_CACHE[normalised] = res
    return res
