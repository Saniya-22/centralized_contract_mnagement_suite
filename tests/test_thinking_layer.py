import pytest
from src.tools.query_classifier import classify_query, QueryIntent


@pytest.mark.asyncio
async def test_safety_critical_uxo():
    """Verify that UXO query triggers is_safety_critical."""
    query = "We found UXO on the construction site."
    res = await classify_query(query)
    assert res.is_safety_critical is True
    assert res.intent == QueryIntent.REGULATION_SEARCH


@pytest.mark.asyncio
async def test_comparison_rea_vs_co():
    """Verify that REA vs CO triggers is_comparison."""
    query = "What is the difference between REA and Change Order?"
    res = await classify_query(query)
    assert res.is_comparison is True


@pytest.mark.asyncio
async def test_drafting_delay_letter():
    """Verify that delay letter query triggers is_document_request."""
    query = "Write a letter about weather delay."
    res = await classify_query(query)
    assert res.is_document_request is True
    assert res.document_request_type == "letter"


@pytest.mark.asyncio
async def test_analytical_commissioning():
    """Verify that commissioning query triggers is_construction_lifecycle."""
    query = "analyze commissioning requirements"
    res = await classify_query(query)
    assert res.is_construction_lifecycle is True


@pytest.mark.asyncio
async def test_safety_critical_fire():
    """Verify that fire query triggers is_safety_critical."""
    query = "There is a fire at the site overnight."
    res = await classify_query(query)
    assert res.is_safety_critical is True


@pytest.mark.asyncio
async def test_comparison_instead_of():
    """Verify that 'instead of' triggers is_comparison."""
    query = "Why should I submit an REA instead of a Change Order?"
    res = await classify_query(query)
    assert res.is_comparison is True
