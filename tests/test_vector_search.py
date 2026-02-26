"""Tests for vector search functionality"""

import pytest
from unittest.mock import Mock, patch
from src.tools.vector_search import VectorSearchTool


@pytest.fixture
def vector_search_tool():
    """Create vector search tool instance"""
    return VectorSearchTool()


def test_vector_search_tool_initialization(vector_search_tool):
    """Test tool initialization"""
    assert vector_search_tool is not None
    assert hasattr(vector_search_tool, 'search_regulations')


def test_search_regulations_schema():
    """Test search regulations tool schema"""
    tool = VectorSearchTool.search_regulations
    
    assert tool.name == "search_regulations"
    assert tool.description is not None
    assert 'query' in str(tool.args_schema.model_json_schema())


@patch('src.tools.vector_search.rerank')
@patch('src.tools.vector_search.get_embedding')
@patch('src.tools.vector_search.VectorQueries.hybrid_search')
def test_search_regulations_execution(mock_hybrid_search, mock_get_embedding, mock_rerank):
    """Test search execution"""
    # Mock embedding
    mock_get_embedding.return_value = [0.1] * 1536
    
    # Mock search results
    mock_hybrid_search.return_value = [
        {
            "content": "Test content",
            "source_file": "test.pdf",
            "metadata": {"regulation_type": "FAR", "section": "1.1"},
            "chunk_index": 1,
            "final_score": 0.95
        }
    ]
    
    # Mock rerank to return the same results with a fixed rerank_score
    mock_rerank.return_value = [
        {
            "content": "Test content",
            "source_file": "test.pdf",
            "metadata": {"regulation_type": "FAR", "section": "1.1"},
            "chunk_index": 1,
            "rerank_score": 0.95
        }
    ]
    
    # Execute search
    result = VectorSearchTool.search_regulations.invoke({
        "query": "test query",
        "k": 5
    })
    
    assert len(result) == 1
    assert result[0]["content"] == "Test content"
    assert result[0]["regulation_type"] == "FAR"
    assert result[0]["score"] == 0.95
