"""Tests for the GovGigOrchestrator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.orchestrator import GovGigOrchestrator
from src.tools.query_classifier import QueryIntent
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
    
    # Test unknown agent path (fallback to synthesizer)
    state = {"next_agent": "non_existent"}
    assert orchestrator._determine_next_agent(state) == "synthesizer"


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
