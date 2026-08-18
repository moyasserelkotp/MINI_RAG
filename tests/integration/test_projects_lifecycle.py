import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import AsyncMock, MagicMock
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
    app.vectordb_client = MagicMock()
    app.generation_client = MagicMock()
    app.embedding_client = MagicMock()
    app.template_parser = MagicMock()
    return TestClient(app)

def test_create_project_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_or_create_one.return_value = FakeProject(id="123", project_id="new-proj")
    mocker.patch("routes.projects.ProjectModel.create_instance", return_value=mock_project_model)
    
    response = client.post("/api/v1/projects/", json={"project_id": "new-proj"}, headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "PROJECT_CREATED"
    assert response.json()["project"]["project_id"] == "new-proj"

def test_list_projects_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_all_projects.return_value = ([FakeProject(id="123", project_id="proj-1")], 1)
    mocker.patch("routes.projects.ProjectModel.create_instance", return_value=mock_project_model)
    
    response = client.get("/api/v1/projects/", headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "PROJECTS_RETRIEVED"
    assert len(response.json()["projects"]) == 1

def test_delete_project_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_by_id.return_value = FakeProject(id="123", project_id="del-proj")
    mocker.patch("routes.projects.ProjectModel.create_instance", return_value=mock_project_model)
    
    mock_chunk_model = AsyncMock()
    mocker.patch("routes.projects.ChunkModel.create_instance", return_value=mock_chunk_model)
    
    mock_asset_model = AsyncMock()
    mocker.patch("routes.projects.AssetModel.create_instance", return_value=mock_asset_model)
    
    mocker.patch("routes.projects.NLPController.reset_vector_db_collection", return_value=True)
    
    response = client.delete("/api/v1/projects/del-proj", headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "delete_project_success"
