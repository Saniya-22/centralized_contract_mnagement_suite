"""Tests for the deterministic query classifier."""

import pytest
from src.tools.query_classifier import classify_query, QueryIntent


def test_classify_empty_query():
    """Test classification of empty or whitespace queries."""
    assert classify_query("").intent == QueryIntent.OUT_OF_SCOPE
    assert classify_query("   ").intent == QueryIntent.OUT_OF_SCOPE
    assert classify_query(None).intent == QueryIntent.OUT_OF_SCOPE


def test_classify_clause_lookup_far():
    """Test FAR clause lookup detection."""
    result = classify_query("What does FAR 52.236-2 say?")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.clause_reference == "FAR 52.236-2"
    assert result.regulation_type == "FAR"


def test_classify_clause_lookup_dfars():
    """Test DFARS clause lookup detection."""
    result = classify_query("Tell me about DFARS 252.204-7012")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.clause_reference == "DFARS 252.204-7012"
    assert result.regulation_type == "DFARS"


def test_classify_clause_lookup_em385():
    """Test EM 385 clause lookup detection."""
    result = classify_query("Check EM 385 05.A for safety")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.clause_reference == "EM385 05.A"
    assert result.regulation_type == "EM385"


def test_classify_regulation_search_source():
    """Test general regulation search by source name."""
    result = classify_query("I need information about FAR regulations")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.regulation_type == "FAR"


def test_classify_regulation_search_keywords():
    """Test general regulation search by keywords."""
    result = classify_query("What are the safety requirements for construction?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert "safety" in result.matched_keywords
    assert "construction" in result.matched_keywords


def test_classify_out_of_scope():
    """Test out-of-scope queries."""
    result = classify_query("What is the weather today?")
    assert result.intent == QueryIntent.OUT_OF_SCOPE
    assert result.clause_reference is None
    assert result.regulation_type is None
