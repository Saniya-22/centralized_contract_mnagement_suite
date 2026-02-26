"""Tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "database" in data
    assert "orchestrator" in data


def test_query_endpoint_validation(client):
    """Test query endpoint validation"""
    # Missing query
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422
    
    # Empty query
    response = client.post("/api/v1/query", json={"query": ""})
    assert response.status_code == 422


def test_query_endpoint_success(client):
    """Test successful query processing"""
    with patch('src.api.main.orchestrator') as mock_orch:
        mock_orch.run_sync.return_value = {
            "response": "Success response",
            "documents": [{"content": "Doc 1"}],
            "confidence": 0.95,
            "agent_path": ["Path"],
            "regulation_types": ["FAR"],
            "errors": []
        }
        
        response = client.post(
            "/api/v1/query",
            json={"query": "test query", "cot": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Success response"
        assert len(data["documents"]) == 1
        assert data["confidence"] == 0.95


def test_clause_lookup_success(client):
    """Test successful clause lookup"""
    with patch('src.api.main.VectorQueries.get_clause_by_reference') as mock_lookup:
        mock_lookup.return_value = {
            "found": True,
            "clause": {"content": "Clause content"},
            "context": "Clause context"
        }
        
        response = client.get("/api/v1/clause/FAR%2052.236-2")
        
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["clause"]["content"] == "Clause content"


def test_clause_lookup_not_found(client):
    """Test clause lookup for non-existent clause"""
    with patch('src.api.main.VectorQueries.get_clause_by_reference') as mock_lookup:
        mock_lookup.return_value = {
            "found": False
        }
        
        response = client.get("/api/v1/clause/NONEXISTENT")
        
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is False
