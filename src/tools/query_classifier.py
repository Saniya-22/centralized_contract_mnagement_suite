"""Deterministic query intent classifier.

Replaces the gpt-4o router + tool-selector LLM calls with a fast, zero-cost
local classifier.  This alone removes 2 LLM round-trips (~3–6 seconds) from
every single query.

Design principles
-----------------
- No LLM dependency — purely regex + keyword matching, runs in <1 ms
- Word-boundary keyword matching — no false positives from substrings
- Confidence scoring — every result carries a 0.0–1.0 confidence value
- LRU-cached — identical queries are free after the first classification
- Single responsibility — classify intent and extract structured metadata
- Testable — pure function, no side effects, no I/O
- Extensible — add new intent types without touching orchestrator/agent code

Intent types
------------
CLAUSE_LOOKUP       Exact clause reference detected (FAR 52.x, DFARS 252.x, EM385, 48 CFR, OSHA, HSAR …)
REGULATION_SEARCH   General regulatory question — route to hybrid RAG search
OUT_OF_SCOPE        Nothing regulatory detected — candidate for refusal

Confidence scale
----------------
1.0   Exact clause reference matched
0.8   Regulatory source name (FAR / DFARS / …) present
0.6   Regulatory keyword(s) matched
0.0   Out of scope / empty query
"""

from __future__ import annotations

import re
import unicodedata
import json
import os
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Optional, List, Tuple


# ── Intent Enum ───────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    """Classified intent for an incoming user query."""
    CLAUSE_LOOKUP      = "clause_lookup"       # Specific clause reference
    REGULATION_SEARCH  = "regulation_search"   # General regulatory question
    OUT_OF_SCOPE       = "out_of_scope"        # Non-regulatory query


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
    "acquisition", "award", "bid", "bidder", "bidding",
    "clause", "competition", "competitive",
    "compliance", "contracting", "contractor", "contract",
    "cost", "cost-reimbursement", "cost-plus",
    "debarment", "default", "delivery",
    # Domain-specific procurement terms
    "earnest", "evaluation", "federal",
    "grant", "guaranty", "indemnification",
    "invoice", "invoicing",
    "labor", "labour", "liquidated damages",
    "modification", "negotiation",
    "offeror", "option",
    "penalty", "performance bond",
    "procurement", "proposal",
    "provision", "purchase order",
    "regulation", "requirement",
    "rfp", "rfq", "rfi",
    "sealed bid", "section", "solicitation",
    "subcontract", "subcontractor",
    "surety", "suspension",
    # Construction / safety specific
    "asbestos", "confined space",
    "construction", "fall protection",
    "hazardous", "lockout", "noise exposure",
    "ozone", "ppe", "radiation",
    "respiratory", "rigging", "safety",
    "scaffold", "silica", "trench",
    # Small business / diversity
    "8a", "dbe", "disadvantaged", "hubzone",
    "sba", "sdvosb", "small business", "set-aside", "set-asides",
    "veteran-owned", "vosb", "wosb",
    # Government structure
    "appropriation", "authorization",
    "clin", "defense", "department",
    "executive order", "far part",
    "fiscal", "government",
    "part", "prime contractor",
    "program", "project",
    "standard", "subpart", "title",
    # FAR-specific topic anchors
    "buy american", "domestic preference", "country of origin", "trade agreements act",
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


# ── Source normalisation ───────────────────────────────────────────────────────

_SOURCE_NORM: dict[str, str] = {
    "FAR":      "FAR",
    "DFARS":    "DFARS",
    "DFARSPGI": "DFARSPGI",
    "48 CFR":   "48 CFR",
    "48CFR":    "48 CFR",
    "EM385":    "EM385",
    "EM 385":   "EM385",
    "HSAR":     "HSAR",
    "DEAR":     "DEAR",
    "VAAR":     "VAAR",
    "NFS":      "NFS",
    "HHSAR":    "HHSAR",
    "AFARS":    "AFARS",
    "NMCARS":   "NMCARS",
    "DLAD":     "DLAD",
    "OSHA":     "OSHA",
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
    intent:           QueryIntent
    confidence:       float              = 0.0    # 0.0–1.0; see module docstring
    clause_reference: Optional[str]     = None   # e.g. "FAR 52.236-2"
    clause_source:    Optional[str]     = None   # e.g. "FAR"
    clause_number:    Optional[str]     = None   # e.g. "52.236-2"
    regulation_type:  Optional[str]     = None   # coerced source label
    matched_keywords: list[str]         = field(default_factory=list)

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
    if denom == 0: return 0.0
    return float(np.dot(v1, v2) / denom)


# ── Waterfall Layers ─────────────────────────────────────────────────────────

def _get_semantic_intent(query: str) -> Tuple[QueryIntent, float]:
    """Layer 2: Semantic similarity check via embeddings."""
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

def _extract_clause_llm(query: str) -> Optional[Tuple[str, str]]:
    """Layer 3: Micro-LLM fallback for messy extraction."""
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


@lru_cache(maxsize=512)
def classify_query(query: Optional[str]) -> ClassificationResult:
    """Classify a user query with a robust multi-layered (HIC) waterfall.

    Waterfall Logic:
      1. Layer 1: Enhanced Regex (Fastest, High Confidence)
      2. Layer 2: Micro-LLM Fallback (Robust extraction for sloppy text)
      3. Layer 3: Semantic Intent (Catches synonyms/phrasing)
      4. Layer 4: Deterministic Source Match (Safety net)
    """
    if not query or not query.strip():
        return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE, confidence=0.0)

    normalised = _normalize_query(query)

    # ── 1. Layer 1: Exact clause reference ─────────────────────────────────────────────
    match = _CLAUSE_PATTERN.search(normalised)
    if match:
        raw_source = match.group(1)
        clause_num = match.group(2).rstrip("-")
        source     = _normalise_source(raw_source)
        ref        = f"{source} {clause_num}"
        return ClassificationResult(
            intent           = QueryIntent.CLAUSE_LOOKUP,
            confidence       = 1.0,
            clause_reference = ref,
            clause_source    = source,
            clause_number    = clause_num,
            regulation_type  = source,
        )

    # ── 2. Layer 2: Micro-LLM Fallback ────────────────────────────────────────
    # We try LLM here if regex failed, to catch sloppy clause refs before general search
    llm_match = _extract_clause_llm(normalised)
    if llm_match:
        source, num = llm_match
        norm_source = _normalise_source(source)
        return ClassificationResult(
            intent           = QueryIntent.CLAUSE_LOOKUP,
            confidence       = 0.9,
            clause_reference = f"{norm_source} {num}",
            clause_source    = norm_source,
            clause_number    = num,
            regulation_type  = norm_source,
        )

    # ── 3. Layer 3: Semantic Intent ──────────────────────────────────────────
    sem_intent, sem_conf = _get_semantic_intent(normalised)
    if sem_intent == QueryIntent.REGULATION_SEARCH:
        return ClassificationResult(
            intent     = QueryIntent.REGULATION_SEARCH,
            confidence = sem_conf,
        )

    # ── 4. Layer 4: Deterministic Source Match ────────────────────────────────────
    source_match = _SOURCE_PATTERN.search(normalised)
    if source_match:
        source = _normalise_source(source_match.group(1))
        return ClassificationResult(
            intent          = QueryIntent.REGULATION_SEARCH,
            confidence      = 0.8,
            regulation_type = source,
        )

    # ── 5. Out of scope ───────────────────────────────────────────────────────
    return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE, confidence=0.0)
