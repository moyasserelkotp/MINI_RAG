import pytest
from fastapi.testclient import TestClient
from main import app
from models.ProjectModel import ProjectModel
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
def mock_project_model(mocker):
    # Mock ProjectModel.create_instance
    mock_create = mocker.patch("routes.nlp.ProjectModel.create_instance", new_callable=AsyncMock)
    mock_instance = AsyncMock()
    # get_project_by_id returns None for 404
    mock_instance.get_project_by_id.return_value = None
    mock_create.return_value = mock_instance
    
    # Mock in data router too
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_instance)
    
    return mock_instance

def test_nlp_answer_404(client, mock_project_model):
    response = client.post("/api/v1/nlp/answer/missing-project", json={"text": "hello"}, headers={"X-API-Key": "mock-api-key"})
    assert response.status_code == 404
    assert response.json()["detail"]["signal"] == "PROJECT_NOT_FOUND_ERROR"

def test_nlp_search_404(client, mock_project_model):
    response = client.post("/api/v1/nlp/index/search/missing-project", json={"text": "hello"}, headers={"X-API-Key": "mock-api-key"})
    assert response.status_code == 404

def test_nlp_delete_index_404(client, mock_project_model):
    response = client.delete("/api/v1/nlp/index/missing-project", headers={"X-API-Key": "mock-api-key"})
    assert response.status_code == 404

def test_data_list_assets_404(client, mock_project_model):
    response = client.get("/api/v1/data/assets/missing-project", headers={"X-API-Key": "mock-api-key"})
    assert response.status_code == 404


