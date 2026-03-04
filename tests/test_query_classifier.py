"""Tests for the deterministic query classifier."""

import pytest
from src.tools.query_classifier import classify_query, QueryIntent


# ── Basic edge cases ──────────────────────────────────────────────────────────

# ── Basic edge cases ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_empty_query():
    assert (await classify_query("")).intent == QueryIntent.OUT_OF_SCOPE
    assert (await classify_query("   ")).intent == QueryIntent.OUT_OF_SCOPE
    assert (await classify_query(None)).intent == QueryIntent.OUT_OF_SCOPE

@pytest.mark.asyncio
async def test_out_of_scope_has_zero_confidence():
    result = await classify_query("What is the weather today?")
    assert result.intent == QueryIntent.OUT_OF_SCOPE
    assert result.confidence == 0.0
    assert result.clause_reference is None
    assert result.regulation_type is None


# ── CLAUSE_LOOKUP — FAR / DFARS / EM385 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_clause_lookup_far():
    result = await classify_query("What does FAR 52.236-2 say?")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "FAR 52.236-2"
    assert result.regulation_type == "FAR"

@pytest.mark.asyncio
async def test_classify_clause_lookup_dfars():
    result = await classify_query("Tell me about DFARS 252.204-7012")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "DFARS 252.204-7012"
    assert result.regulation_type == "DFARS"

@pytest.mark.asyncio
async def test_classify_clause_lookup_em385():
    result = await classify_query("Check EM 385 05.A for safety")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "EM385 05.A"
    assert result.regulation_type == "EM385"


# ── CLAUSE_LOOKUP — extended sources ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_clause_lookup_48cfr():
    result = await classify_query("Explain 48 CFR 52.212-4 please")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "48 CFR 52.212-4"
    assert result.regulation_type == "48 CFR"

@pytest.mark.asyncio
async def test_classify_clause_lookup_osha():
    result = await classify_query("What does OSHA 1926.502 cover?")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "OSHA 1926.502"
    assert result.regulation_type == "OSHA"

@pytest.mark.asyncio
async def test_classify_clause_lookup_hsar():
    result = await classify_query("HSAR 3052.209-70 requirements")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.clause_reference == "HSAR 3052.209-70"
    assert result.regulation_type == "HSAR"

@pytest.mark.asyncio
async def test_classify_clause_lookup_dear():
    result = await classify_query("Summarise DEAR 970.5204-2")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.regulation_type == "DEAR"

@pytest.mark.asyncio
async def test_classify_clause_lookup_vaar():
    result = await classify_query("Apply VAAR 852.219-10")
    assert result.intent == QueryIntent.CLAUSE_LOOKUP
    assert result.confidence == 1.0
    assert result.regulation_type == "VAAR"


# ── REGULATION_SEARCH — source name only ─────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_regulation_search_source_far():
    result = await classify_query("I need information about FAR regulations")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.8
    assert result.regulation_type == "FAR"

@pytest.mark.asyncio
async def test_classify_regulation_search_source_osha():
    result = await classify_query("What are OSHA rules for construction sites?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.8
    assert result.regulation_type == "OSHA"

@pytest.mark.asyncio
async def test_classify_regulation_search_source_hsar():
    result = await classify_query("Is there a HSAR requirement for this?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.8
    assert result.regulation_type == "HSAR"


# ── REGULATION_SEARCH — keyword match ────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_regulation_search_keywords():
    result = await classify_query("What are the safety requirements for construction?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6
    assert "safety" in result.matched_keywords
    assert "construction" in result.matched_keywords

@pytest.mark.asyncio
async def test_classify_regulation_search_procurement():
    result = await classify_query("How does the procurement process work for small businesses?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6
    assert "procurement" in result.matched_keywords

@pytest.mark.asyncio
async def test_classify_regulation_search_solicitation():
    result = await classify_query("How do I respond to a solicitation?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6

@pytest.mark.asyncio
async def test_classify_regulation_search_buy_american_with_far_hint():
    result = await classify_query("What are the Buy American requirements for steel?")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert result.confidence == 0.6
    assert result.regulation_type == "FAR"
    assert "buy american" in result.matched_keywords


# ── No false positives from substring matching ────────────────────────────────

@pytest.mark.asyncio
async def test_no_false_positive_apartment():
    """'apartment' must not match keyword 'part'."""
    result = await classify_query("I am looking for an apartment near downtown")
    assert result.intent == QueryIntent.OUT_OF_SCOPE

@pytest.mark.asyncio
async def test_no_false_positive_contractor_word():
    """Ensure 'contractor' matches but random words containing 'contract' do NOT."""
    result = await classify_query("I need to contact someone")
    # 'contact' should not match 'contract' (different word)
    assert result.intent == QueryIntent.OUT_OF_SCOPE

@pytest.mark.asyncio
async def test_no_false_positive_standard_word():
    """'standard' must only match at word boundary."""
    result = await classify_query("The substandard conditions were noted")
    # 'substandard' contains 'standard' but is a different word — should not match
    assert result.intent == QueryIntent.OUT_OF_SCOPE


# ── LRU cache: calling twice returns same object ──────────────────────────────

@pytest.mark.asyncio
async def test_async_cache_same_result():
    r1 = await classify_query("What does FAR 52.236-2 say?")
    r2 = await classify_query("What does FAR 52.236-2 say?")
    assert r1 is r2  # exact same object from cache


# ── Confidence scale check ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confidence_clause_lookup_is_1():
    assert (await classify_query("FAR 52.212-4")).confidence == 1.0

@pytest.mark.asyncio
async def test_confidence_source_match_is_0_8():
    assert (await classify_query("Tell me about DFARS")).confidence == 0.8

@pytest.mark.asyncio
async def test_confidence_keyword_match_is_0_6():
    assert (await classify_query("What are the compliance requirements?")).confidence == 0.6

@pytest.mark.asyncio
async def test_confidence_out_of_scope_is_0():
    assert (await classify_query("How do I bake a cake?")).confidence == 0.0


# ── Regression Tests for Reported Bugs ────────────────────────────────────────

@pytest.mark.asyncio
async def test_layer5_gate_is_not_too_broad():
    """'I need 3 examples' should not trigger Layer 5 (GPT-4o-mini)."""
    # Since we can't easily mock the internal _extract_clause_llm without more complex overrides,
    # we check that it stays OUT_OF_SCOPE instead of becoming a CLAUSE_LOOKUP with low conf (0.9).
    result = await classify_query("I need 3 examples of procurement rules")
    assert result.intent == QueryIntent.OUT_OF_SCOPE
    assert result.confidence == 0.0

@pytest.mark.asyncio
async def test_keyword_plural_support():
    """Keywords should match plural versions (fixing the regex bug)."""
    result = await classify_query("Show me all active contracts")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert "contract" in result.matched_keywords
    
    result = await classify_query("Check for safety requirements")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert "requirement" in result.matched_keywords
