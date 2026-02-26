"""Tests for data retrieval agent"""

import pytest
from unittest.mock import Mock, patch
from src.agents.data_retrieval import DataRetrievalAgent
from src.state.graph_state import GovGigState


@pytest.fixture
def data_retrieval_agent():
    """Create data retrieval agent instance"""
    return DataRetrievalAgent()


@pytest.fixture
def sample_state():
    """Create sample state"""
    return GovGigState(
        messages=[],
        query="What are the requirements for small business set-asides?",
        person_id=None,
        current_date="Thursday, February 26, 2026",
        chat_history=[],
        cot_enabled=False,
        retrieved_documents=[],
        tool_calls=[],
        thought_process=[],
        agent_path=[],
        regulation_types_used=[],
        errors=[]
    )


def test_agent_initialization(data_retrieval_agent):
    """Test agent initialization"""
    assert data_retrieval_agent is not None
    assert data_retrieval_agent.name == "DataRetrievalAgent"
    assert hasattr(data_retrieval_agent, 'llm_with_tools')


def test_get_system_prompt(data_retrieval_agent, sample_state):
    """Test system prompt generation"""
    prompt = data_retrieval_agent.get_system_prompt(sample_state)
    
    assert "Data Retrieval Agent" in prompt
    assert "FAR" in prompt
    assert "DFARS" in prompt
    assert "EM385" in prompt


@patch('src.agents.data_retrieval.DataRetrievalAgent._create_messages')
@patch('src.agents.base.ChatOpenAI')
def test_agent_run(mock_llm_class, mock_create_messages, data_retrieval_agent, sample_state):
    """Test agent execution"""
    # Mock LLM instance and its bind_tools method
    mock_llm_instance = mock_llm_class.return_value
    mock_bound_llm = Mock()
    mock_llm_instance.bind_tools.return_value = mock_bound_llm
    
    # Re-initialize agent to use mocked bind_tools
    agent = DataRetrievalAgent()
    agent.llm_with_tools = mock_bound_llm
    
    # Mock LLM response
    mock_response = Mock()
    mock_response.tool_calls = []
    mock_response.content = "Test response"
    mock_bound_llm.invoke.return_value = mock_response
    
    # Run agent
    result = agent.run(sample_state)
    
    assert 'agent_path' in result
    assert len(result['agent_path']) > 0
