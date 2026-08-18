import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import AsyncMock, MagicMock
from helpers.config import get_settings
from datetime import datetime

class FakeProject:
    def __init__(self, id="mock-id", project_id="proj-test"):
        self.id = id
        self.project_id = project_id

class FakeAsset:
    def __init__(self, id="asset-1", asset_name="doc.pdf", asset_size=1024, asset_type="file", asset_project_id="mock-id"):
        self.id = id
        self.asset_name = asset_name
        self.asset_size = asset_size
        self.asset_type = asset_type
        self.asset_project_id = asset_project_id
        self.asset_pushed_at = datetime.utcnow()

@pytest.fixture
def client():
    settings = get_settings()
    settings.ENABLE_AUTH = False
    app.db_client = AsyncMock()
    app.vectordb_client = MagicMock()
    return TestClient(app)

def test_list_assets_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_by_id.return_value = FakeProject(id="proj-1-id", project_id="proj-1")
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_project_model)
    
    mock_asset_model = AsyncMock()
    mock_asset_model.get_all_project_assets.return_value = [FakeAsset()]
    mocker.patch("routes.data.AssetModel.create_instance", return_value=mock_asset_model)
    
    response = client.get("/api/v1/data/assets/proj-1", headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "GET_ASSETS_SUCCESS"
    assert response.json()["total"] == 1
    assert response.json()["assets"][0]["asset_name"] == "doc.pdf"

def test_delete_asset_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_or_create_one.return_value = FakeProject(id="proj-1-id", project_id="proj-1")
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_project_model)
    
    mock_asset_model = AsyncMock()
    fake_asset = FakeAsset(id="asset-1", asset_name="doc.pdf", asset_project_id="proj-1-id")
    mock_asset_model.get_asset_by_id.return_value = fake_asset
    mock_asset_model.delete_asset_by_id.return_value = True
    mocker.patch("routes.data.AssetModel.create_instance", return_value=mock_asset_model)
    
    mock_chunk_model = AsyncMock()
    mock_chunk_model.delete_chunks_by_asset_id.return_value = 5
    mocker.patch("routes.data.ChunkModel.create_instance", return_value=mock_chunk_model)
    
    mocker.patch("routes.data.DataController.delete_file_by_name", return_value=True)
    
    response = client.delete("/api/v1/data/assets/proj-1/asset-1", headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "DELETE_ASSET_SUCCESS"

def test_delete_asset_forbidden_cross_project(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_or_create_one.return_value = FakeProject(id="proj-2-id", project_id="proj-2")
    mocker.patch("routes.data.ProjectModel.create_instance", return_value=mock_project_model)
    
    mock_asset_model = AsyncMock()
    fake_asset = FakeAsset(id="asset-1", asset_name="doc.pdf", asset_project_id="proj-1-id")
    mock_asset_model.get_asset_by_id.return_value = fake_asset
    mocker.patch("routes.data.AssetModel.create_instance", return_value=mock_asset_model)
    
    response = client.delete("/api/v1/data/assets/proj-2/asset-1", headers={"X-API-Key": "mock"})
    assert response.status_code == 403
    assert response.json()["detail"]["signal"] == "delete_asset_error"
