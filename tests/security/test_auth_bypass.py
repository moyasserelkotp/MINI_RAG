# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
# pyrefly: ignore [missing-import]
from main import app
from unittest.mock import AsyncMock
# pyrefly: ignore [missing-import]
from helpers.config import get_settings

@pytest.fixture
def auth_client():
    settings = get_settings()
    settings.ENABLE_AUTH = True
    settings.DEBUG = True
    settings.API_KEYS = ["secret-api-key-123"]
    app.db_client = AsyncMock()
    return TestClient(app)

def test_auth_missing_header_rejected(auth_client):
    response = auth_client.get("/api/v1/projects/")
    assert response.status_code == 401
    assert "Invalid or missing API key" in response.json()["detail"]

def test_auth_invalid_key_rejected(auth_client):
    response = auth_client.get("/api/v1/projects/", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401

def test_auth_exempt_paths_allowed_without_key(auth_client):
    # Health checks and docs are exempt
    response = auth_client.get("/api/v1/health")
    assert response.status_code == 200
    # /docs is not tested here because its registration depends on DEBUG flag at import time
