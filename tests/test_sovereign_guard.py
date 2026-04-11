"""Unit tests for the SovereignGuard service."""

import pytest
from unittest.mock import patch, MagicMock
from src.services.sovereign_guard import SovereignGuard


@pytest.fixture
def guard():
    """Create a SovereignGuard instance."""
    return SovereignGuard()


def test_build_detect_urls_default(guard):
    """Test URL building with default settings."""
    guard.base_url = "http://test-guard"
    guard.detect_path = "/detect"
    urls = guard._build_detect_urls()
    assert "http://test-guard/detect" in urls
    assert "http://test-guard/api/detect" in urls
    assert len(urls) == 2


def test_build_detect_urls_custom(guard):
    """Test URL building with custom path."""
    guard.base_url = "http://test-guard"
    guard.detect_path = "/custom/path"
    urls = guard._build_detect_urls()
    assert "http://test-guard/custom/path" in urls
    assert len(urls) == 1


def test_parse_verdict_allow(guard):
    """Test parsing an 'allow' verdict."""
    payload = {"action": "allow", "confidence": 0.95, "explanation": "Safe content"}
    verdict = guard._parse_verdict(payload)
    assert verdict["action"] == "allow"
    assert verdict["should_block"] is False
    assert verdict["confidence"] == 0.95
    assert verdict["reason"] == "Safe content"


def test_parse_verdict_block(guard):
    """Test parsing a 'block' verdict."""
    payload = {"action": "block", "should_block": True, "reason": "Policy violation"}
    verdict = guard._parse_verdict(payload)
    assert verdict["action"] == "block"
    assert verdict["should_block"] is True
    assert verdict["reason"] == "Policy violation"


def test_parse_verdict_alternate_keys(guard):
    """Test parsing payload with alternate 'blocked' key."""
    payload = {"blocked": True, "explanation": "Blocked by filter"}
    verdict = guard._parse_verdict(payload)
    assert verdict["should_block"] is True
    assert verdict["reason"] == "Blocked by filter"


@patch("src.services.sovereign_guard.httpx.Client")
def test_evaluate_response_success(mock_client_class, guard):
    """Test evaluate_response with a successful mock HTTP call."""
    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "action": "allow",
        "confidence": 0.98,
        "reason": "Clean",
    }
    mock_client.post.return_value = mock_response

    with patch.object(guard, "enabled", True):
        verdict = guard.evaluate_response("Test response", "test query")
        assert verdict["action"] == "allow"
        assert verdict["should_block"] is False
        assert verdict["confidence"] == 0.98


@patch("src.services.sovereign_guard.httpx.Client")
def test_evaluate_response_http_error_fail_open(mock_client_class, guard):
    """Test evaluate_response with HTTP error and fail-open enabled."""
    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_client.post.return_value = mock_response

    with patch.object(guard, "enabled", True):
        with patch.object(guard, "fail_open", True):
            verdict = guard.evaluate_response("Test response", "test query")
            assert verdict["action"] == "allow"
            assert verdict["should_block"] is False
            assert "HTTP 500" in verdict["reason"]


@patch("src.services.sovereign_guard.httpx.Client")
def test_evaluate_response_http_error_fail_closed(mock_client_class, guard):
    """Test evaluate_response with HTTP error and fail-open disabled."""
    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_client.post.return_value = mock_response

    with patch.object(guard, "enabled", True):
        with patch.object(guard, "fail_open", False):
            verdict = guard.evaluate_response("Test response", "test query")
            assert verdict["action"] == "block"
            assert verdict["should_block"] is True
            assert "HTTP 500" in verdict["reason"]


def test_evaluate_response_disabled(guard):
    """Test evaluate_response when the service is disabled."""
    with patch.object(guard, "enabled", False):
        verdict = guard.evaluate_response("Test response", "test query")
        assert verdict is None
