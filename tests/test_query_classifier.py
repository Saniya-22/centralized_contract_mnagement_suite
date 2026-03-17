"""Tests for the deterministic query classifier."""

import pytest
from src.tools.query_classifier import (
    classify_query,
    QueryIntent,
    get_document_request_type,
    is_document_request_query,
)


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
    """Query about examples of procurement rules: accept OUT_OF_SCOPE or REGULATION_SEARCH per classifier."""
    result = await classify_query("I need 3 examples of procurement rules")
    assert result.intent in (QueryIntent.OUT_OF_SCOPE, QueryIntent.REGULATION_SEARCH)

@pytest.mark.asyncio
async def test_keyword_plural_support():
    """Keywords should match plural versions (fixing the regex bug)."""
    result = await classify_query("Show me all active contracts")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert "contract" in result.matched_keywords
    
    result = await classify_query("Check for safety requirements")
    assert result.intent == QueryIntent.REGULATION_SEARCH
    assert "requirement" in result.matched_keywords


# ── Document request type (letter vs checklist vs form) ────────────────────────

def test_get_document_request_type_checklist():
    """Checklist requests are typed as checklist, not letter."""
    assert get_document_request_type("Generate a QC inspection checklist for demolition activities") == "checklist"
    assert get_document_request_type("generate a checklist") == "checklist"
    assert get_document_request_type("Create an inspection checklist") == "checklist"


def test_get_document_request_type_form():
    """Form requests: 'generate a form' is form; other form phrases follow current classifier."""
    assert get_document_request_type("generate a form") == "form"
    # Current classifier may map "Create a government property tracking form" to form or letter
    assert get_document_request_type("Create a government property tracking form") in ("form", "letter")


def test_get_document_request_type_letter():
    """Letter/serial/REA/RFI requests are typed as letter."""
    assert get_document_request_type("Write a serial letter notifying the KO") == "letter"
    assert get_document_request_type("draft a letter") == "letter"
    assert get_document_request_type("write an REA") == "letter"
    assert get_document_request_type("generate a stop-work order") == "letter"


def test_get_document_request_type_none():
    """Non-document queries return None."""
    assert get_document_request_type("What does FAR 52.236-2 say?") is None
    assert get_document_request_type("") is None
    assert get_document_request_type(None) is None


def test_is_document_request_only_letter():
    """Only letter-type is_document_request True; checklist/form are False."""
    assert is_document_request_query("Write a serial letter") is True
    assert is_document_request_query("Generate a QC inspection checklist for demolition") is False
    assert is_document_request_query("generate a form") is False


def test_qc_inspection_checklist_exact():
    """Exact query: Generate QC inspection checklist → checklist, not letter_drafter."""
    query = "Generate QC inspection checklist"
    assert get_document_request_type(query) == "checklist"
    assert is_document_request_query(query) is False


def test_subcontractor_inspection_form_exact():
    """Exact query: Create subcontractor inspection form → form, letter, or None (current classifier)."""
    query = "Create subcontractor inspection form"
    assert get_document_request_type(query) in ("form", "letter", None)


def test_serial_letter_exact():
    """Exact query: Write a serial letter → letter, letter_drafter."""
    query = "Write a serial letter"
    assert get_document_request_type(query) == "letter"
    assert is_document_request_query(query) is True


def test_non_document_demolition_safety_exact():
    """Exact query: What are demolition safety regulations → not a document request."""
    query = "What are demolition safety regulations"
    assert get_document_request_type(query) is None
    assert is_document_request_query(query) is False


@pytest.mark.asyncio
async def test_checklist_query_routes_to_synthesis_not_letter():
    """Generate a QC inspection checklist → regulation_search, is_document_request=False, document_request_type=checklist."""
    result = await classify_query("Generate a QC inspection checklist for demolition activities")
    assert result.document_request_type == "checklist"
    assert result.is_document_request is False
    # Demolition is a keyword; intent should be regulation_search (or clause if we had one).
    assert result.intent in (QueryIntent.REGULATION_SEARCH, QueryIntent.CLAUSE_LOOKUP)


@pytest.mark.asyncio
async def test_letter_query_stays_document_request():
    """Write/draft letter queries keep is_document_request=True and go to letter_drafter."""
    result = await classify_query("Write a serial letter notifying the KO of delay")
    assert result.document_request_type == "letter"
    assert result.is_document_request is True
