"""Tests for reflection components (critique + healing + query expansion)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.reflection.critique import RetrievalCritique
from src.reflection.expansion import QueryExpansion
from src.reflection.manager import ReflectionManager


def test_critique_fails_when_no_documents():
    """Reflection critique should fail fast when nothing is retrieved."""
    critique = RetrievalCritique(threshold=0.7)

    result = critique.evaluate("FAR 52.219-8 requirements", [])

    assert result["passed"] is False
    assert result["score"] == 0.0
    assert "No documents" in result["reason"]


def test_critique_fails_on_regulation_mismatch():
    """Reflection critique should reject top docs from the wrong regulation."""
    critique = RetrievalCritique(threshold=0.7)
    docs = [
        {"score": 9.0, "regulation_type": "DFARS", "content": "Cyber rules"},
        {"score": 8.5, "regulation_type": "DFARS", "content": "More cyber rules"},
        {"score": 8.2, "regulation_type": "DFARS", "content": "Related content"},
    ]

    result = critique.evaluate("What does FAR 52.204-21 require?", docs)

    assert result["passed"] is False
    assert "mismatch" in result["reason"].lower()


def test_critique_borderline_score_can_pass_with_keyword_overlap():
    """Borderline scores can pass when semantic overlap is still high."""
    critique = RetrievalCritique(threshold=0.7)
    docs = [
        {
            "score": 0.5,
            "regulation_type": "FAR",
            "content": "Safety requirements for construction excavation operations and site planning.",
        }
    ]

    result = critique.evaluate(
        "construction excavation safety requirements planning",
        docs,
    )

    assert result["passed"] is True
    assert result["score"] == 0.5


@pytest.mark.asyncio
async def test_heal_search_returns_flattened_results():
    """Self-healing should run expanded queries and flatten list outputs."""
    manager = ReflectionManager(threshold=0.7)
    manager.expansion.expand = AsyncMock(return_value=["q1", "q2"])

    async def fake_search(query):
        if query == "q1":
            return [{"content": "doc1"}, {"content": "doc2"}]
        return [{"content": "doc3"}]

    result = await manager.heal_search("base query", "Low retrieval confidence.", fake_search)

    assert len(result) == 3
    assert {d["content"] for d in result} == {"doc1", "doc2", "doc3"}


@pytest.mark.asyncio
async def test_heal_search_returns_empty_when_no_expansions():
    """If expansion yields nothing, healing should short-circuit."""
    manager = ReflectionManager(threshold=0.7)
    manager.expansion.expand = AsyncMock(return_value=[])

    async def fake_search(_query):
        return [{"content": "should-not-run"}]

    result = await manager.heal_search("base query", "No documents retrieved.", fake_search)

    assert result == []


@pytest.mark.asyncio
@patch("src.reflection.expansion.ChatOpenAI")
async def test_query_expansion_returns_two_lines(mock_chat_openai):
    """Query expansion should parse LLM output into at most two queries."""
    mock_llm = mock_chat_openai.return_value
    mock_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content=(
                "FAR 52.219-8 subcontracting plan requirements\n"
                "FAR small business set-aside clause guidance\n"
                "extra line ignored"
            )
        )
    )

    expansion = QueryExpansion()
    result = await expansion.expand("FAR 52.219-8", "Low retrieval confidence.")

    assert len(result) == 2
    assert "subcontracting" in result[0]


@pytest.mark.asyncio
@patch("src.reflection.expansion.ChatOpenAI")
async def test_query_expansion_returns_empty_on_llm_error(mock_chat_openai):
    """Query expansion should fail safely when the LLM call errors."""
    mock_llm = mock_chat_openai.return_value
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))

    expansion = QueryExpansion()
    result = await expansion.expand("FAR 52.219-8", "Low retrieval confidence.")

    assert result == []
