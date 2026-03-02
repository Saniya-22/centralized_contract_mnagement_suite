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
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Optional


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
        EM\s*385                # Army Corps Safety Manual
    )
    \s+
    (\d+[\.\-][\d\-A-Za-z]+)   # Clause/section number
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
        EM\s*385
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
    key = re.sub(r"\s+", " ", raw.strip().upper())
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

@lru_cache(maxsize=512)
def classify_query(query: Optional[str]) -> ClassificationResult:
    """Classify a user query with no LLM calls.

    Results are LRU-cached (up to 512 unique queries) so repeat calls
    within a session are essentially free.

    Precedence:
      1. Explicit clause reference  → CLAUSE_LOOKUP      (confidence 1.0)
      2. Regulatory source name     → REGULATION_SEARCH  (confidence 0.8)
      3. Regulatory keyword(s)      → REGULATION_SEARCH  (confidence 0.6)
      4. Fallback                   → OUT_OF_SCOPE       (confidence 0.0)

    Args:
        query: Raw user query string (None is safe — returns OUT_OF_SCOPE).

    Returns:
        ClassificationResult with intent, confidence, and extracted metadata.
    """
    if not query or not query.strip():
        return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE, confidence=0.0)

    normalised = _normalize_query(query)

    # ── 1. Exact clause reference ─────────────────────────────────────────────
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

    # ── 2. Regulatory source name present ────────────────────────────────────
    source_match = _SOURCE_PATTERN.search(normalised)
    if source_match:
        source = _normalise_source(source_match.group(1))
        return ClassificationResult(
            intent          = QueryIntent.REGULATION_SEARCH,
            confidence      = 0.8,
            regulation_type = source,
        )

    # ── 3. Regulatory keyword present (word-boundary safe) ───────────────────
    matched = [kw for kw, pat in _KW_PATTERNS if pat.search(normalised)]
    if matched:
        hinted_regulation = _infer_regulation_hint(normalised)
        return ClassificationResult(
            intent           = QueryIntent.REGULATION_SEARCH,
            confidence       = 0.6,
            regulation_type  = hinted_regulation,
            matched_keywords = matched,
        )

    # ── 4. Out of scope ───────────────────────────────────────────────────────
    return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE, confidence=0.0)
