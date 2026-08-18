import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import AsyncMock
from helpers.config import get_settings

class FakeProject:
    def __init__(self, id="mock-id", project_id="proj-test"):
        self.id = id
        self.project_id = project_id

@pytest.fixture
def client():
    settings = get_settings()
    settings.ENABLE_AUTH = False
    app.db_client = AsyncMock()
    return TestClient(app)

def test_ssrf_file_scheme_blocked(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_or_create_one.return_value = FakeProject()
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_project_model)
    
    # URL with invalid/dangerous scheme file://
    # Pydantic schema validation or endpoint checks scheme
    response = client.post(
        "/api/v1/data/ingest/url/test-proj",
        json={"url": "file:///etc/passwd"},
        headers={"X-API-Key": "mock"}
    )
    assert response.status_code in (400, 422)

def test_ssrf_localhost_blocked(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_or_create_one.return_value = FakeProject()
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_project_model)
    
    # Localhost / loopback
    response = client.post(
        "/api/v1/data/ingest/url/test-proj",
        json={"url": "http://127.0.0.1:8000/internal"},
        headers={"X-API-Key": "mock"}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["signal"] == "URL_BLOCKED"

def test_ssrf_private_network_blocked(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_or_create_one.return_value = FakeProject()
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_project_model)
    
    # RFC 1918 Private IP
    response = client.post(
        "/api/v1/data/ingest/url/test-proj",
        json={"url": "http://192.168.1.1/admin"},
        headers={"X-API-Key": "mock"}
    )
    assert response.status_code == 400
    assert response.json()["detail"]["signal"] == "URL_BLOCKED"
