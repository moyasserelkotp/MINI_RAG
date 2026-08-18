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

def test_nlp_info_index_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_by_id.return_value = FakeProject(id="123", project_id="proj-1")
    mocker.patch("routes.nlp.ProjectModel.create_instance", return_value=mock_project_model)
    
    mocker.patch("routes.nlp.NLPController.get_vector_db_collection_info", return_value={"vectors_count": 42})
    
    response = client.get("/api/v1/nlp/index/info/proj-1", headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "vectordb_collection_retrieved"
    assert response.json()["collection_info"]["vectors_count"] == 42

def test_nlp_search_retrieve_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_by_id.return_value = FakeProject(id="123", project_id="proj-1")
    mocker.patch("routes.nlp.ProjectModel.create_instance", return_value=mock_project_model)
    
    class FakeHit:
        id = "hit-1"
        score = 0.95
        payload = {"text": "Tourism in Cairo is wonderful", "metadata": {"source": "doc1.pdf"}}
        
    mocker.patch("routes.nlp.NLPController.search_vector_db_collection", return_value=[FakeHit()])
    
    response = client.post("/api/v1/nlp/retrieve/proj-1", json={"text": "Cairo tourism"}, headers={"X-API-Key": "mock"})
    assert response.status_code == 200
    assert response.json()["signal"] == "vectordb_search_success"
    assert response.json()["total"] == 1
    assert response.json()["results"][0]["text"] == "Tourism in Cairo is wonderful"

def test_nlp_answer_rag_success(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_by_id.return_value = FakeProject(id="123", project_id="proj-1")
    mocker.patch("routes.nlp.ProjectModel.create_instance", return_value=mock_project_model)
    
    mocker.patch(
        "routes.nlp.NLPController.answer_rag_question",
        new_callable=AsyncMock,
        return_value=(
            "Cairo is the capital of Egypt and is famous for the Pyramids.",
            "Prompt",
            [],
            [{"source": "guide.pdf", "score": 0.9}],
            False
        )
    )
    
    response = client.post(
        "/api/v1/nlp/answer/proj-1",
        json={"text": "Tell me about Cairo", "session_id": "sess-1"},
        headers={"X-API-Key": "mock"}
    )
    assert response.status_code == 200
    assert response.json()["signal"] == "rag_answer_success"
    assert "Pyramids" in response.json()["answer"]
    assert response.json()["session_id"] == "sess-1"

def test_nlp_answer_rag_no_docs(client, mocker):
    mock_project_model = AsyncMock()
    mock_project_model.get_project_by_id.return_value = FakeProject(id="123", project_id="proj-1")
    mocker.patch("routes.nlp.ProjectModel.create_instance", return_value=mock_project_model)
    
    mocker.patch(
        "routes.nlp.NLPController.answer_rag_question",
        new_callable=AsyncMock,
        return_value=(None, None, None, [], False)
    )
    
    response = client.post(
        "/api/v1/nlp/answer/proj-1",
        json={"text": "Unknown topic"},
        headers={"X-API-Key": "mock"}
    )
    assert response.status_code == 200
    assert response.json()["signal"] == "rag_answer_success"
    assert "No relevant documents" in response.json()["answer"]
