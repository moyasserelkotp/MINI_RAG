"""Shared pytest fixtures for MINI-RAG test suite."""
import sys
import os
import pytest
from unittest.mock import MagicMock, AsyncMock

# Ensure src/ is on the path when running from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_embedding_client():
    """Embedding client that returns a fixed 768-dim vector."""
    client = MagicMock()
    client.embed_text.return_value = [0.1] * 768
    client.embed_batch.return_value = [[0.1] * 768]
    client.embedding_size = 768
    return client


@pytest.fixture
def mock_generation_client():
    """Generation client that returns a canned LLM answer."""
    client = MagicMock()
    client.generate_text.return_value = "Test answer from LLM."
    # Provider enum stubs
    enums = MagicMock()
    enums.SYSTEM.value = "system"
    enums.USER.value = "user"
    enums.ASSISTANT.value = "assistant"
    client.enums = enums
    client.construct_prompt.return_value = {"role": "user", "content": "test"}
    return client


@pytest.fixture
def mock_vectordb_client():
    """VectorDB client that returns empty search results by default."""
    client = MagicMock()
    client.is_collection_existed.return_value = True
    client.search_by_vector.return_value = []
    client.hybrid_search.return_value = []
    client.upsert.return_value = True
    return client


@pytest.fixture
def mock_db_client():
    """Minimal async MongoDB client stub."""
    client = AsyncMock()
    return client


@pytest.fixture
def app_settings_mock():
    """Minimal Settings-like object for controller tests."""
    s = MagicMock()
    s.FILE_ALLOWED_TYPES = ["pdf", "txt", "docx", "csv", "html", "doc", "md"]
    s.FILE_MAX_SIZE = 50_000_000
    s.FILE_DEFAULT_CHUNK_SIZE = 512_000
    s.FILE_PROCESS_CHUNK_SIZE = 512
    s.FILE_PROCESS_OVERLAP_SIZE = 50
    s.CHUNK_STRATEGY = "recursive"
    s.ENABLE_AUTH = False
    s.API_KEYS = []
    s.RATE_LIMIT_GLOBAL = "100/minute"
    s.RATE_LIMIT_ANSWER = "10/minute"
    s.RATE_LIMIT_UPLOAD = "20/minute"
    s.USE_WINDOW_MEMORY = True
    s.WINDOW_MEMORY_K = 5
    s.USE_SUMMARY_MEMORY = False
    s.USE_ENTITY_MEMORY = False
    s.USE_VECTOR_MEMORY = False
    s.USE_SEMANTIC_CACHE = False
    s.USE_RERANK = False
    return s
