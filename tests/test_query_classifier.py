"""Tests for the deterministic query classifier."""

import pytest
from src.tools.query_classifier import classify_query, QueryIntent


# ── Basic edge cases ──────────────────────────────────────────────────────────

def test_classify_empty_query():
    assert classify_query("").intent == QueryIntent.OUT_OF_SCOPE
    assert classify_query("   ").intent == QueryIntent.OUT_OF_SCOPE
    assert classify_query(None).intent == QueryIntent.OUT_OF_SCOPE

def test_out_of_scope_has_zero_confidence():
    result = classify_query("What is the weather today?")
    assert result.intent == QueryIntent.OUT_OF_SCOPE
    assert result.confidence == 0.0
    assert result.clause_reference is None
    assert result.regulation_type is None


# ── CLAUSE_LOOKUP — FAR / DFARS / EM385 ──────────────────────────────────────

def test_classify_clause_lookup_far():
    result = classify_query("What does FAR 52.236-2 say?")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "FAR 52.236-2"
    assert result.regulation_type == "FAR"

def test_classify_clause_lookup_dfars():
    result = classify_query("Tell me about DFARS 252.204-7012")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "DFARS 252.204-7012"
    assert result.regulation_type == "DFARS"

def test_classify_clause_lookup_em385():
    result = classify_query("Check EM 385 05.A for safety")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "EM385 05.A"
    assert result.regulation_type == "EM385"


# ── CLAUSE_LOOKUP — extended sources ─────────────────────────────────────────

def test_classify_clause_lookup_48cfr():
    result = classify_query("Explain 48 CFR 52.212-4 please")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "48 CFR 52.212-4"
    assert result.regulation_type == "48 CFR"

def test_classify_clause_lookup_osha():
    result = classify_query("What does OSHA 1926.502 cover?")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "OSHA 1926.502"
    assert result.regulation_type == "OSHA"

def test_classify_clause_lookup_hsar():
    result = classify_query("HSAR 3052.209-70 requirements")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "HSAR 3052.209-70"
    assert result.regulation_type == "HSAR"

def test_classify_clause_lookup_dear():
    result = classify_query("Summarise DEAR 970.5204-2")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.regulation_type == "DEAR"

def test_classify_clause_lookup_vaar():
    result = classify_query("Apply VAAR 852.219-10")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.regulation_type == "VAAR"


# ── REGULATION_SEARCH — source name only ─────────────────────────────────────

def test_classify_regulation_search_source_far():
    result = classify_query("I need information about FAR regulations")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.8
    assert result.regulation_type == "FAR"

def test_classify_regulation_search_source_osha():
    result = classify_query("What are OSHA rules for construction sites?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.8
    assert result.regulation_type == "OSHA"

def test_classify_regulation_search_source_hsar():
    result = classify_query("Is there a HSAR requirement for this?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.8
    assert result.regulation_type == "HSAR"


# ── REGULATION_SEARCH — keyword match ────────────────────────────────────────

def test_classify_regulation_search_keywords():
    result = classify_query("What are the safety requirements for construction?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6
    assert "safety" in result.matched_keywords
    assert "construction" in result.matched_keywords

def test_classify_regulation_search_procurement():
    result = classify_query("How does the procurement process work for small businesses?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6
    assert "procurement" in result.matched_keywords

def test_classify_regulation_search_solicitation():
    result = classify_query("How do I respond to a solicitation?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6


# ── No false positives from substring matching ────────────────────────────────

def test_no_false_positive_apartment():
    """'apartment' must not match keyword 'part'."""
    result = classify_query("I am looking for an apartment near downtown")
    assert result.intent == QueryIntent.OUT_OF_SCOPE

def test_no_false_positive_contractor_word():
    """Ensure 'contractor' matches but random words containing 'contract' do NOT."""
    result = classify_query("I need to contact someone")
    # 'contact' should not match 'contract' (different word)
    assert result.intent == QueryIntent.OUT_OF_SCOPE

def test_no_false_positive_standard_word():
    """'standard' must only match at word boundary."""
    result = classify_query("The substandard conditions were noted")
    # 'substandard' contains 'standard' but is a different word — should not match
    assert result.intent == QueryIntent.OUT_OF_SCOPE


# ── LRU cache: calling twice returns same object ──────────────────────────────

def test_lru_cache_same_result():
    r1 = classify_query("What does FAR 52.236-2 say?")
    r2 = classify_query("What does FAR 52.236-2 say?")
    assert r1 is r2  # exact same object from cache


# ── Confidence scale check ────────────────────────────────────────────────────

def test_confidence_clause_lookup_is_1():
    assert classify_query("FAR 52.212-4").confidence == 1.0

def test_confidence_source_match_is_0_8():
    assert classify_query("Tell me about DFARS").confidence == 0.8

def test_confidence_keyword_match_is_0_6():
    assert classify_query("What are the compliance requirements?").confidence == 0.6

def test_confidence_out_of_scope_is_0():
    assert classify_query("How do I bake a cake?").confidence == 0.0
