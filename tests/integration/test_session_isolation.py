import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import AsyncMock
from helpers.config import get_settings

@pytest.fixture
def client():
    # Bypass auth for this test entirely by modifying the cached settings
    settings = get_settings()
    settings.ENABLE_AUTH = False
    app.db_client = AsyncMock()
    return TestClient(app)

@pytest.fixture
def mock_models(mocker):
    # Same standard mocking for project and sessions
    mock_project = AsyncMock()
    mock_project.get_project_by_id.return_value = {"project_id": "proj-1"}
    mocker.patch("routes.projects.ProjectModel.create_instance", return_value=mock_project)
    
    mock_session = AsyncMock()
    # The session belongs to proj-1 but we try to access it via proj-2
    mock_session.get_session_by_id.return_value = {"session_id": "sess-abc", "project_id": "proj-1"}
    mocker.patch("routes.sessions.SessionModel.create_instance", return_value=mock_session)
    
    return mock_session

def test_get_session_cross_project_fails(client, mock_models):
    response = client.get("/api/v1/sessions/proj-2/sess-abc", headers={"X-API-Key": "mock-api-key"})
    assert response.status_code == 404
    assert response.json()["detail"]["signal"] == "SESSION_NOT_FOUND_ERROR"

def test_delete_session_cross_project_fails(client, mock_models):
    response = client.delete("/api/v1/sessions/proj-2/sess-abc", headers={"X-API-Key": "mock-api-key"})
    assert response.status_code == 404
    assert response.json()["detail"]["signal"] == "SESSION_NOT_FOUND_ERROR"
