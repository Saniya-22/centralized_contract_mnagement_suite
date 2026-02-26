"""Deterministic query intent classifier.

Replaces the gpt-4o router + tool-selector LLM calls with a fast, zero-cost
local classifier.  This alone removes 2 LLM round-trips (~3–6 seconds) from
every single query.

Design principles
-----------------
- No LLM dependency — purely regex + keyword matching, runs in <1 ms
- Single responsibility — classify intent and extract structured metadata
- Testable — pure function, no side effects, no I/O
- Extensible — add new intent types without touching orchestrator/agent code

Intent types
------------
CLAUSE_LOOKUP       Exact clause reference detected (FAR 52.x, DFARS 252.x, EM385)
REGULATION_SEARCH   General regulatory question — route to hybrid RAG search
OUT_OF_SCOPE        Nothing regulatory detected — candidate for refusal
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Intent Enum ───────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    """Classified intent for an incoming user query."""
    CLAUSE_LOOKUP      = "clause_lookup"       # Specific FAR/DFARS/EM385 clause
    REGULATION_SEARCH  = "regulation_search"   # General regulatory question
    OUT_OF_SCOPE       = "out_of_scope"        # Non-regulatory query


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Matches:  FAR 52.236-2  |  DFARS 252.204-7012  |  EM 385 1-1  |  EM385 05.A
_CLAUSE_PATTERN = re.compile(
    r"\b(FAR|DFARS|EM\s*385)\s+(\d+[\.\-][\d\-A-Za-z]+)",
    re.IGNORECASE,
)

# Matches "FAR", "DFARS", "EM 385", "EM385" anywhere in the query
_SOURCE_PATTERN = re.compile(
    r"\b(FAR|DFARS|EM\s*385)\b",
    re.IGNORECASE,
)

# Keywords that indicate a regulatory / procurement domain query
_REGULATION_KEYWORDS: frozenset[str] = frozenset({
    "acquisition", "clause", "compliance", "contract", "contracting",
    "contractor", "construction", "cost", "defense", "federal",
    "part", "procurement", "provision", "regulation", "requirement",
    "safety", "section", "solicitation", "standard", "subpart",
    "subcontract", "subcontractor", "title",
})


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassificationResult:
    """Immutable result returned by classify_query()."""
    intent: QueryIntent
    clause_reference: Optional[str] = None   # e.g. "FAR 52.236-2"
    clause_source: Optional[str]   = None    # e.g. "FAR"
    clause_number: Optional[str]   = None    # e.g. "52.236-2"
    regulation_type: Optional[str] = None    # coerced source label
    matched_keywords: list[str]     = field(default_factory=list)

    @property
    def is_clause_lookup(self) -> bool:
        return self.intent == QueryIntent.CLAUSE_LOOKUP

    @property
    def is_regulation_search(self) -> bool:
        return self.intent == QueryIntent.REGULATION_SEARCH

    @property
    def is_out_of_scope(self) -> bool:
        return self.intent == QueryIntent.OUT_OF_SCOPE


# ── Source normalisation ───────────────────────────────────────────────────────

_SOURCE_NORM: dict[str, str] = {
    "FAR":   "FAR",
    "DFARS": "DFARS",
    "EM385": "EM385",
    "EM 385": "EM385",
}

def _normalise_source(raw: str) -> str:
    """Normalise matched source string to canonical label."""
    key = re.sub(r"\s+", " ", raw.strip().upper())
    return _SOURCE_NORM.get(key, key)


# ── Main classifier ────────────────────────────────────────────────────────────

def classify_query(query: str) -> ClassificationResult:
    """Classify a user query with no LLM calls.

    Precedence:
      1. Explicit clause reference  → CLAUSE_LOOKUP
      2. Regulatory keyword/source  → REGULATION_SEARCH
      3. Fallback                   → OUT_OF_SCOPE

    Args:
        query: Raw user query string.

    Returns:
        ClassificationResult with intent and extracted metadata.
    """
    if not query or not query.strip():
        return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE)

    # ── 1. Exact clause reference ─────────────────────────────────────────────
    match = _CLAUSE_PATTERN.search(query)
    if match:
        raw_source = match.group(1)
        clause_num = match.group(2).rstrip("-")
        source     = _normalise_source(raw_source)
        ref        = f"{source} {clause_num}"
        return ClassificationResult(
            intent           = QueryIntent.CLAUSE_LOOKUP,
            clause_reference = ref,
            clause_source    = source,
            clause_number    = clause_num,
            regulation_type  = source,
        )

    # ── 2. Regulatory source name present ────────────────────────────────────
    source_match = _SOURCE_PATTERN.search(query)
    if source_match:
        source = _normalise_source(source_match.group(1))
        return ClassificationResult(
            intent          = QueryIntent.REGULATION_SEARCH,
            regulation_type = source,
        )

    # ── 3. Regulatory keyword present ────────────────────────────────────────
    query_lower = query.lower()
    matched = [kw for kw in _REGULATION_KEYWORDS if kw in query_lower]
    if matched:
        return ClassificationResult(
            intent           = QueryIntent.REGULATION_SEARCH,
            matched_keywords = matched,
        )

    # ── 4. Out of scope ───────────────────────────────────────────────────────
    return ClassificationResult(intent=QueryIntent.OUT_OF_SCOPE)
