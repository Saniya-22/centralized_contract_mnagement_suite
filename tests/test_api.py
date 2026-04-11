"""API test suite for FastAPI endpoints."""

import uuid
import pytest
from unittest.mock import patch, AsyncMock

# client and client_no_auth fixtures from tests/conftest.py


# ─── Root and health (no auth) ───


@pytest.mark.api
def test_root_endpoint(client):
    """Root returns welcome and version."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data


@pytest.mark.api
def test_health_check(client):
    """Health returns status, version, database, orchestrator."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "database" in data
    assert "orchestrator" in data


@pytest.mark.api
def test_docs_and_redoc(client):
    """OpenAPI docs and ReDoc are served."""
    r = client.get("/docs")
    assert r.status_code == 200
    r = client.get("/redoc")
    assert r.status_code == 200


# ─── Query endpoint ───


@pytest.mark.api
def test_query_endpoint_validation(client):
    """Query rejects missing or empty body."""
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422
    response = client.post("/api/v1/query", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.api
def test_query_endpoint_success(client):
    """Query returns 200 and response when orchestrator returns result."""
    with patch("src.api.main.orchestrator") as mock_orch:
        mock_orch.run_async = AsyncMock(
            return_value={
                "response": "Success response",
                "documents": [{"content": "Doc 1"}],
                "confidence": 0.95,
                "agent_path": ["Path"],
                "regulation_types": ["FAR"],
                "errors": [],
            }
        )
        response = client.post(
            "/api/v1/query",
            json={"query": "test query", "cot": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Success response"
        assert len(data["documents"]) == 1
        assert data["confidence"] == 0.95


@pytest.mark.api
def test_query_requires_auth(client_no_auth):
    """Query without Authorization returns 403."""
    response = client_no_auth.post(
        "/api/v1/query",
        json={"query": "hello"},
    )
    assert response.status_code == 403


@pytest.mark.api
def test_query_invalid_token_returns_401(client_no_auth):
    """Query with invalid Bearer token returns 401."""
    response = client_no_auth.post(
        "/api/v1/query",
        json={"query": "hello"},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


@pytest.mark.api
def test_query_rate_limit_429(client):
    """Query returns 429 when rate limit exceeded."""
    with patch("src.api.main.rate_limiter") as mock_limiter:
        mock_limiter.check.return_value = False
        with patch("src.api.main.orchestrator") as mock_orch:
            mock_orch.run_async = AsyncMock(return_value={})
            response = client.post(
                "/api/v1/query",
                json={"query": "test"},
            )
            assert response.status_code == 429
            data = response.json()
            assert "detail" in data


@pytest.mark.api
def test_query_orchestrator_unavailable_503(client):
    """Query returns 503 when orchestrator is None."""
    with patch("src.api.main.orchestrator", None):
        response = client.post(
            "/api/v1/query",
            json={"query": "test"},
        )
        assert response.status_code == 503


# ─── Clause lookup ───


@pytest.mark.api
def test_clause_lookup_success(client):
    """Clause lookup returns 200 and clause when found."""
    with patch("src.api.main.VectorQueries.get_clause_by_reference") as mock_lookup:
        mock_lookup.return_value = {
            "found": True,
            "clause": {"content": "Clause content"},
            "context": "Clause context",
        }
        response = client.get("/api/v1/clause/FAR%2052.236-2")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["clause"]["content"] == "Clause content"


@pytest.mark.api
def test_clause_lookup_not_found(client):
    """Clause lookup returns 200 and found=False when not found."""
    with patch("src.api.main.VectorQueries.get_clause_by_reference") as mock_lookup:
        mock_lookup.return_value = {"found": False}
        response = client.get("/api/v1/clause/NONEXISTENT")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is False


@pytest.mark.api
def test_clause_requires_auth(client_no_auth):
    """Clause lookup without auth returns 403."""
    response = client_no_auth.get("/api/v1/clause/FAR%2052.236-2")
    assert response.status_code == 403


# ─── Auth: signup ───


@pytest.mark.api
def test_signup_success(client_no_auth):
    """Signup with new email returns 201 and user_id."""
    with patch("src.api.main.VectorQueries.get_user_by_email", return_value=None):
        with patch(
            "src.api.main.VectorQueries.create_user",
            return_value={"user_id": uuid.uuid4(), "id": 1},
        ):
            response = client_no_auth.post(
                "/api/v1/auth/signup",
                json={
                    "full_name": "Test User",
                    "email": "new@example.com",
                    "password": "password123",
                    "confirm_password": "password123",
                },
            )
            assert response.status_code == 201
            data = response.json()
            assert data.get("status") == "Signup successful"
            assert data.get("user_id") is not None


@pytest.mark.api
def test_signup_email_already_registered(client_no_auth):
    """Signup with existing email returns 409."""
    with patch("src.api.main.VectorQueries.get_user_by_email") as mock_get:
        mock_get.return_value = {"id": 1, "email": "existing@example.com"}
        response = client_no_auth.post(
            "/api/v1/auth/signup",
            json={
                "full_name": "Test User",
                "email": "existing@example.com",
                "password": "password123",
                "confirm_password": "password123",
            },
        )
        assert response.status_code == 409
        data = response.json()
        assert (
            "already registered" in data.get("detail", "").lower()
            or data.get("status") == "Signup unsuccessful"
        )


@pytest.mark.api
def test_signup_validation(client_no_auth):
    """Signup with invalid body returns 422 (e.g. short password, mismatch)."""
    # Passwords don't match
    response = client_no_auth.post(
        "/api/v1/auth/signup",
        json={
            "full_name": "Test",
            "email": "a@b.com",
            "password": "password123",
            "confirm_password": "different",
        },
    )
    assert response.status_code == 422
    # Too short password
    response = client_no_auth.post(
        "/api/v1/auth/signup",
        json={
            "full_name": "Test",
            "email": "a@b.com",
            "password": "short",
            "confirm_password": "short",
        },
    )
    assert response.status_code == 422


# ─── Auth: login ───


@pytest.mark.api
def test_login_success(client_no_auth):
    """Login with valid credentials returns 200 and access_token."""
    with patch("src.api.main.VectorQueries.get_user_by_email") as mock_get:
        mock_get.return_value = {
            "id": 1,
            "user_id": uuid.uuid4(),
            "email": "user@example.com",
            "hashed_password": "$2b$12$dummyhashedpassword",  # bcrypt hash of something
            "full_name": "User",
            "lock_until": None,
            "failed_login_attempts": 0,
        }
        with patch("src.api.main.bcrypt.checkpw", return_value=True):
            with patch("src.api.main.VectorQueries.update_login_success"):
                response = client_no_auth.post(
                    "/api/v1/auth/login",
                    json={"email": "user@example.com", "password": "secret123"},
                )
                assert response.status_code == 200
                data = response.json()
                assert data.get("access_token")
                assert data.get("status") == "Login successful"


@pytest.mark.api
def test_login_invalid_credentials(client_no_auth):
    """Login with wrong password returns 401."""
    with patch("src.api.main.VectorQueries.get_user_by_email") as mock_get:
        mock_get.return_value = {
            "id": 1,
            "user_id": uuid.uuid4(),
            "email": "user@example.com",
            "hashed_password": "hashed",
            "full_name": "User",
            "lock_until": None,
        }
        with patch("src.api.main.bcrypt.checkpw", return_value=False):
            with patch("src.api.main.VectorQueries.update_login_failure"):
                response = client_no_auth.post(
                    "/api/v1/auth/login",
                    json={"email": "user@example.com", "password": "wrong"},
                )
                assert response.status_code == 401


@pytest.mark.api
def test_login_user_not_found(client_no_auth):
    """Login with unknown email returns 401."""
    with patch("src.api.main.VectorQueries.get_user_by_email", return_value=None):
        response = client_no_auth.post(
            "/api/v1/auth/login",
            json={"email": "unknown@example.com", "password": "secret123"},
        )
        assert response.status_code == 401


# ─── Feedback ───


@pytest.mark.api
def test_feedback_success(client):
    """Feedback with valid body returns 201."""
    with patch("src.api.main.VectorQueries.insert_user_feedback", return_value=True):
        response = client.post(
            "/api/v1/feedback",
            json={"query_id": str(uuid.uuid4()), "response": "good"},
        )
        assert response.status_code == 201
        assert response.json().get("ok") is True


@pytest.mark.api
def test_feedback_requires_auth(client_no_auth):
    """Feedback without auth returns 403."""
    response = client_no_auth.post(
        "/api/v1/feedback",
        json={"query_id": str(uuid.uuid4()), "response": "good"},
    )
    assert response.status_code == 403


@pytest.mark.api
def test_feedback_validation(client):
    """Feedback with invalid response or query_id returns 422."""
    # Invalid response (not 'good' or 'bad')
    response = client.post(
        "/api/v1/feedback",
        json={"query_id": str(uuid.uuid4()), "response": "neutral"},
    )
    assert response.status_code == 422
    # Invalid query_id (not UUID)
    response = client.post(
        "/api/v1/feedback",
        json={"query_id": "not-a-uuid", "response": "good"},
    )
    assert response.status_code == 422


# ─── Chat: threads and history ───


@pytest.mark.api
def test_chat_threads_success(client):
    """GET /chat/threads returns 200 and threads list."""
    with patch("src.api.main.VectorQueries.list_chat_threads", return_value=[]):
        response = client.get("/api/v1/chat/threads")
        assert response.status_code == 200
        data = response.json()
        assert "threads" in data


@pytest.mark.api
def test_chat_threads_requires_auth(client_no_auth):
    """GET /chat/threads without auth returns 403."""
    response = client_no_auth.get("/api/v1/chat/threads")
    assert response.status_code == 403


@pytest.mark.api
def test_chat_history_success(client):
    """GET /chat/history returns 200 and history."""
    with patch("src.api.main.VectorQueries.get_chat_history", return_value=[]):
        response = client.get("/api/v1/chat/history?thread_id=" + str(uuid.uuid4()))
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "thread_id" in data


@pytest.mark.api
def test_chat_history_requires_auth(client_no_auth):
    """GET /chat/history without auth returns 403."""
    response = client_no_auth.get("/api/v1/chat/history?thread_id=" + str(uuid.uuid4()))
    assert response.status_code == 403


# ─── Analytics ───


@pytest.mark.api
def test_analytics_summary_success(client):
    """GET /analytics/summary returns 200."""
    with patch("src.api.main.VectorQueries.get_analytics_summary", return_value={}):
        response = client.get("/api/v1/analytics/summary?hours=24")
        assert response.status_code == 200


@pytest.mark.api
def test_analytics_summary_requires_auth(client_no_auth):
    """GET /analytics/summary without auth returns 403."""
    response = client_no_auth.get("/api/v1/analytics/summary?hours=24")
    assert response.status_code == 403
