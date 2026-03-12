"""Shared pytest fixtures for API and other tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.auth import get_current_user


@pytest.fixture
def client():
    """Authenticated test client (get_current_user overridden to return a test user)."""
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test_user_id"}
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


@pytest.fixture
def client_no_auth():
    """Test client with no auth override; protected routes return 403 (no token) or 401 (invalid token)."""
    old = app.dependency_overrides.pop(get_current_user, None)
    try:
        with TestClient(app) as c:
            yield c
    finally:
        if old is not None:
            app.dependency_overrides[get_current_user] = old
