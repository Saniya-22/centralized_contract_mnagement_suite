"""Tests for the GovGigOrchestrator."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from types import SimpleNamespace
from src.agents.orchestrator import GovGigOrchestrator
from src.tools.query_classifier import QueryIntent
from src.config import settings
from langchain_core.messages import AIMessage

@pytest.fixture
def orchestrator():
    """Create orchestrator instance with mocked LLM and agents."""
    with patch('src.agents.orchestrator.ChatOpenAI'), \
         patch('src.agents.orchestrator.DataRetrievalAgent'):
        return GovGigOrchestrator()


def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization."""
    assert orchestrator.synthesizer_llm is not None
    assert orchestrator.data_retrieval is not None
    assert orchestrator.graph is not None
    assert orchestrator.app is not None


def test_route_query(orchestrator):
    """Test the routing node logic."""
    state = {"query": "What is FAR 52.236-2?"}
    
    with patch('src.agents.orchestrator.classify_query') as mock_classify:
        mock_classify.return_value = Mock(
            intent=QueryIntent.CLAUSE_LOOKUP,
            confidence=0.95,
            clause_reference="FAR 52.236-2",
            regulation_type="FAR"
        )
        
        delta = orchestrator._route_query(state)
        
        assert delta["next_agent"] == "data_retrieval"
        assert "FAR 52.236-2" in delta["detected_clause_ref"]
        assert delta["query_intent"] == QueryIntent.CLAUSE_LOOKUP.value


def test_determine_next_agent(orchestrator):
    """Test conditional transition logic."""
    # Test data_retrieval path
    state = {"next_agent": "data_retrieval"}
    assert orchestrator._determine_next_agent(state) == "data_retrieval"

    # Test end path (out of scope)
    state = {"next_agent": "end"}
    assert orchestrator._determine_next_agent(state) == "end"

    # Test unknown agent path (fallback to synthesizer)
    state = {"next_agent": "non_existent"}
    assert orchestrator._determine_next_agent(state) == "synthesizer"


def test_after_retrieval_routing(orchestrator):
    """Route to letter_drafter when is_document_request; else synthesizer."""
    state = {"is_document_request": True}
    assert orchestrator._after_retrieval_routing(state) == "letter_drafter"

    state = {"is_document_request": False}
    assert orchestrator._after_retrieval_routing(state) == "synthesizer"

    state = {}
    assert orchestrator._after_retrieval_routing(state) == "synthesizer"


@patch('src.agents.orchestrator.get_synthesizer_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_synthesize_response_success(mock_format, mock_prompt, orchestrator):
    """Test successful response synthesis."""
    mock_format.return_value = "Formatted Docs"
    mock_prompt.return_value = "System Prompt"
    
    state = {
        "query": "test query",
        "retrieved_documents": [
            {
                "content": "Contractor must notify the contracting officer within 5 days [FAR 52.236-2].",
                "score": 0.8
            },
            {
                "content": "Notification timeline is five days from discovery [FAR 52.236-2].",
                "score": 0.78
            }
        ]
    }
    
    # Mock LLM response
    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = "Contractor must notify the contracting officer within 5 days [FAR 52.236-2]."
    orchestrator.synthesizer_llm.invoke.return_value = mock_response
    
    result = orchestrator._synthesize_response(state)
    
    assert result["generated_response"] == mock_response.content
    assert result["confidence_score"] == pytest.approx(0.79)
    assert result["quality_metrics"]["low_confidence"] is False
    assert result["low_confidence"] is False
    assert "Synthesizer: Generated" in result["agent_path"][-1]


def test_synthesize_response_no_docs(orchestrator):
    """Test synthesis fallback when no documents are found."""
    state = {
        "query": "test query",
        "retrieved_documents": []
    }

    result = orchestrator._synthesize_response(state)

    assert "sufficient high-confidence evidence" in result["generated_response"]
    assert "Synthesizer: No documents" in result["agent_path"][-1]


def test_draft_letter_no_docs(orchestrator):
    """Letter drafter returns fallback when no documents."""
    state = {
        "query": "Write a serial letter notifying the KO of delay",
        "retrieved_documents": [],
    }
    result = orchestrator._draft_letter(state)
    assert "sufficient high-confidence evidence" in result["generated_response"]
    assert result["low_confidence"] is True
    assert any("LetterDrafter" in p for p in result["agent_path"])


@patch('src.agents.orchestrator.get_letter_drafter_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_draft_letter_success(mock_format, mock_prompt, orchestrator):
    """Letter drafter produces full draft when documents are present."""
    mock_format.return_value = "Formatted Docs"
    mock_prompt.return_value = "System Prompt"

    state = {
        "query": "Write a serial letter notifying the KO of wildfire impact",
        "retrieved_documents": [
            {"content": "FAR 52.242-5 permits excusable delay [FAR 52.242-5].", "score": 0.8},
            {"content": "Notify Contracting Officer in writing [FAR 52.236-2].", "score": 0.75},
        ],
    }

    draft_body = (
        "Date: [Insert Date]\n\nTo: Contracting Officer\nFrom: Contractor\n"
        "Subject: Notice of Wildfire Impact\n\nParagraph citing FAR 52.242-5 and 52.236-2.\n\n"
        "Draft for reference only; tailor to your situation and consult your contract and legal/CO as needed."
    )
    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = draft_body
    orchestrator.letter_drafter_llm.invoke.return_value = mock_response

    result = orchestrator._draft_letter(state)

    assert result["generated_response"] == draft_body
    assert "quality_metrics" in result
    assert any("LetterDrafter" in p for p in result["agent_path"])
    assert "LetterDrafter: Generated" in result["agent_path"][-1]


@pytest.mark.asyncio
async def test_run_async_document_request_routes_to_letter_drafter(orchestrator):
    """E2E: Document-request query goes through letter_drafter and returns a draft."""
    with patch('src.agents.orchestrator.classify_query', new_callable=AsyncMock) as mock_classify:
        mock_classify.return_value = Mock(
            intent=QueryIntent.REGULATION_SEARCH,
            confidence=0.9,
            clause_reference=None,
            regulation_type="FAR",
            is_procedural=False,
            is_contract_co=False,
            is_document_request=True,
        )
        orchestrator.data_retrieval.run = AsyncMock(return_value={
            "retrieved_documents": [
                {"content": "FAR 52.242-5 excusable delay [FAR 52.242-5].", "score": 0.8},
                {"content": "Notify KO in writing [FAR 52.236-2].", "score": 0.75},
            ],
            "regulation_types_used": ["FAR"],
            "agent_path": ["DataRetrievalAgent: completed"],
        })
        draft_text = (
            "To: Contracting Officer\nFrom: Contractor\nSubject: Wildfire Impact\n\n"
            "Body with FAR 52.242-5 and 52.236-2.\n\n"
            "Draft for reference only; tailor to your situation and consult your contract and legal/CO as needed."
        )
        orchestrator.letter_drafter_llm.invoke.return_value = MagicMock(
            spec=AIMessage, content=draft_text
        )

        result = await orchestrator.run_async(
            "Write a serial letter notifying the KO of the wildfire impact",
            context={"thread_id": "e2e-test", "person_id": "user1"},
        )

        assert "LetterDrafter" in " ".join(result["agent_path"])
        assert "To:" in result["response"] or "Contracting Officer" in result["response"]
        assert "Draft for reference only" in result["response"]
        assert result["response"] == draft_text


@patch('src.agents.orchestrator.get_synthesizer_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_clause_lookup_allows_single_high_conf_doc(mock_format, mock_prompt, orchestrator):
    """Clause lookup path should synthesize directly from exact match evidence."""
    mock_format.return_value = "Formatted Clause Doc"
    mock_prompt.return_value = "System Prompt"

    state = {
        "query": "What does DFARS 252.204-7012 require?",
        "query_intent": QueryIntent.CLAUSE_LOOKUP.value,
        "detected_clause_ref": "DFARS 252.204-7012",
        "retrieved_documents": [{
            "content": "DFARS 252.204-7012 requires cyber incident reporting to DoD [DFARS 252.204-7012].",
            "score": 1.0
        }],
    }

    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = "DFARS 252.204-7012 requires cyber incident reporting to DoD [DFARS 252.204-7012]."
    orchestrator.synthesizer_llm.invoke.return_value = mock_response

    result = orchestrator._synthesize_response(state)

    assert result["generated_response"] == mock_response.content
    assert result["low_confidence"] is False
    assert "low-confidence label applied" not in " ".join(result["agent_path"])


@patch('src.agents.orchestrator.get_synthesizer_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_low_confidence_label_applied_when_weak_support(mock_format, mock_prompt, orchestrator):
    """Weakly grounded and uncited answers should be labeled as low confidence."""
    mock_format.return_value = "Formatted Docs"
    mock_prompt.return_value = "System Prompt"

    state = {
        "query": "test query",
        "retrieved_documents": [{"content": "Short unrelated chunk.", "score": 0.05}],
    }

    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = "You should do extra reporting within two days."
    orchestrator.synthesizer_llm.invoke.return_value = mock_response

    result = orchestrator._synthesize_response(state)

    assert result["low_confidence"] is True
    assert result["generated_response"].startswith("Low confidence notice:")
    assert "low-confidence label applied" in " ".join(result["agent_path"])


@patch('src.agents.orchestrator.get_synthesizer_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_synthesize_response_uses_current_run_documents_only(mock_format, mock_prompt, orchestrator):
    """Synthesis should ignore checkpointed docs from prior turns."""
    mock_format.return_value = "Formatted Docs"
    mock_prompt.return_value = "System Prompt"

    state = {
        "query": "test query",
        "run_offsets": {"retrieved_documents": 1},
        "retrieved_documents": [
            {"content": "Old unrelated checkpoint doc.", "score": 0.95},
            {"content": "Current cited doc [FAR 52.236-2].", "score": 0.2},
        ],
    }

    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = "Current cited doc [FAR 52.236-2]."
    orchestrator.synthesizer_llm.invoke.return_value = mock_response

    result = orchestrator._synthesize_response(state)

    sent_docs = mock_format.call_args.args[0]
    assert len(sent_docs) == 1
    assert sent_docs[0]["content"] == "Current cited doc [FAR 52.236-2]."
    assert result["confidence_score"] == pytest.approx(0.2)


def test_run_sync(orchestrator):
    """Test synchronous execution wrapper."""
    mock_output = {
        "generated_response": "Test Response",
        "retrieved_documents": [],
        "confidence_score": 0.9,
        "quality_metrics": {"quality_score": 0.82, "low_confidence": False},
        "low_confidence": False,
        "agent_path": ["Path"],
        "regulation_types_used": ["FAR"],
        "errors": []
    }
    
    # Mock the compiled graph's invoke method
    orchestrator.app.invoke = MagicMock(return_value=mock_output)
    
    result = orchestrator.run_sync("test query")
    
    assert result["response"] == "Test Response"
    assert result["confidence"] == 0.9
    assert result["quality_metrics"]["quality_score"] == 0.82
    assert result["low_confidence"] is False
    assert result["regulation_types"] == ["FAR"]


def test_run_sync_slices_accumulated_lists_with_checkpointer(orchestrator):
    """run_sync should return only current-turn deltas when checkpoint state exists."""
    orchestrator.checkpointer = object()
    orchestrator.app.get_state = MagicMock(return_value=SimpleNamespace(values={
        "agent_path": ["old-path"],
        "errors": ["old-error"],
        "thought_process": ["old-thought"],
        "retrieved_documents": [{"content": "old-doc"}],
        "regulation_types_used": ["OLD"],
    }))
    orchestrator.app.invoke = MagicMock(return_value={
        "generated_response": "Test Response",
        "retrieved_documents": [{"content": "old-doc"}, {"content": "new-doc"}],
        "confidence_score": 0.9,
        "quality_metrics": {"quality_score": 0.82, "low_confidence": False},
        "low_confidence": False,
        "agent_path": ["old-path", "new-path"],
        "thought_process": ["old-thought", "new-thought"],
        "regulation_types_used": ["OLD", "NEW"],
        "errors": ["old-error", "new-error"],
        "cot_enabled": True,
    })

    result = orchestrator.run_sync("test query", {"thread_id": "t1", "cot": True})

    assert result["documents"] == [{"content": "new-doc"}]
    assert result["agent_path"] == ["new-path"]
    assert result["thought_process"] == ["new-thought"]
    assert result["regulation_types"] == ["NEW"]
    assert result["errors"] == ["new-error"]


@patch('src.agents.orchestrator.get_synthesizer_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_sovereign_soft_block_adds_safety_notice(mock_format, mock_prompt, orchestrator):
    """Soft block mode should label the response instead of replacing it."""
    mock_format.return_value = "Formatted Docs"
    mock_prompt.return_value = "System Prompt"

    state = {
        "query": "test query",
        "retrieved_documents": [
            {"content": "Evidence [FAR 52.236-2].", "score": 0.8},
            {"content": "More evidence [FAR 52.236-2].", "score": 0.75},
        ],
    }

    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = "Answer with citation [FAR 52.236-2]."
    orchestrator.synthesizer_llm.invoke.return_value = mock_response
    orchestrator.sovereign_guard.evaluate_response = MagicMock(return_value={
        "provider": "sovereign_ai",
        "action": "block",
        "should_block": True,
        "reason": "Prompt injection pattern detected.",
    })

    with patch.object(settings, "SOVEREIGN_GUARD_BLOCK_MODE", "soft"):
        result = orchestrator._synthesize_response(state)

    assert "Safety review notice:" in result["generated_response"]
    assert "Prompt injection pattern detected." in result["generated_response"]
    assert result["quality_metrics"]["sovereign_guard"]["action"] == "block"
    assert result["low_confidence"] is True
    assert "soft mode" in " ".join(result["agent_path"])


@patch('src.agents.orchestrator.get_synthesizer_prompt')
@patch('src.agents.orchestrator.format_documents')
def test_sovereign_hard_block_replaces_response(mock_format, mock_prompt, orchestrator):
    """Hard block mode should replace generated content with safe fallback."""
    mock_format.return_value = "Formatted Docs"
    mock_prompt.return_value = "System Prompt"

    state = {
        "query": "test query",
        "retrieved_documents": [
            {"content": "Evidence [FAR 52.236-2].", "score": 0.8},
            {"content": "More evidence [FAR 52.236-2].", "score": 0.75},
        ],
    }

    mock_response = MagicMock(spec=AIMessage)
    mock_response.content = "Answer with citation [FAR 52.236-2]."
    orchestrator.synthesizer_llm.invoke.return_value = mock_response
    orchestrator.sovereign_guard.evaluate_response = MagicMock(return_value={
        "provider": "sovereign_ai",
        "action": "block",
        "should_block": True,
        "reason": "Policy violation",
    })

    with patch.object(settings, "SOVEREIGN_GUARD_BLOCK_MODE", "hard"):
        result = orchestrator._synthesize_response(state)

    assert result["generated_response"] == orchestrator._safe_blocked_message()
    assert result["quality_metrics"]["sovereign_guard"]["action"] == "block"
    assert result["low_confidence"] is True
    assert "hard mode" in " ".join(result["agent_path"])
