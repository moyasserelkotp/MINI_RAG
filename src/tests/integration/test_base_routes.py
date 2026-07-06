"""Integration tests for base routes: /health and /info endpoints."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# We use a lightweight approach: patch all startup dependencies so TestClient
# doesn't need a live MongoDB, Qdrant, or LLM backend.

@pytest.fixture
def client():
    """FastAPI test client with all infrastructure patched out."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    with patch("motor.motor_asyncio.AsyncIOMotorClient") as mock_mongo, \
         patch("stores.llm.LLMProviderFactory.LLMProviderFactory") as mock_llm_factory, \
         patch("stores.vectordb.VectorDBProviderFactory.VectorDBProviderFactory") as mock_vdb_factory:

        # Mock LLM provider
        mock_gen = MagicMock()
        mock_gen.generate_text.return_value = "test"
        mock_emb = MagicMock()
        mock_emb.embed_text.return_value = [0.0] * 768
        mock_emb.embedding_size = 768
        mock_llm_factory.return_value.create.side_effect = [mock_gen, mock_emb]

        # Mock VectorDB
        mock_vdb = MagicMock()
        mock_vdb.connect.return_value = None
        mock_vdb_factory.return_value.create.return_value = mock_vdb

        # Mock MongoDB
        mock_mongo.return_value.__getitem__ = MagicMock(return_value=AsyncMock())

        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data

    def test_health_status_is_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json().get("status") == "ok"


class TestInfoEndpoint:

    def test_info_returns_200(self, client):
        resp = client.get("/api/v1/info")
        assert resp.status_code == 200

    def test_info_has_app_name(self, client):
        resp = client.get("/api/v1/info")
        data = resp.json()
        assert "app_name" in data

    def test_info_app_name_is_mini_rag(self, client):
        resp = client.get("/api/v1/info")
        assert resp.json().get("app_name") == "MINI-RAG"

    def test_info_has_backends_field(self, client):
        resp = client.get("/api/v1/info")
        assert "backends" in resp.json()

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
