import pytest
from unittest.mock import MagicMock
from services.cache_service import CacheService

@pytest.fixture
def cache_service():
    vectordb_client = MagicMock()
    embedding_client = MagicMock()
    generation_client = MagicMock()
    
    generation_client.generation_model_id = "gpt-4-turbo"
    embedding_client.embedding_model_id = "text-embedding-3-small"
    
    class MockSettings:
        PRIMARY_LANG = "en"
    app_settings = MockSettings()
    
    return CacheService(
        vectordb_client=vectordb_client,
        embedding_client=embedding_client,
        generation_client=generation_client,
        app_settings=app_settings,
        initialized_collections=set()
    )

def test_cache_key_deterministic(cache_service):
    key1 = cache_service.build_cache_key("proj-1", "What is the capital of France?")
    key2 = cache_service.build_cache_key("proj-1", "What is the capital of France?")
    assert key1 == key2

def test_cache_key_differs_by_project(cache_service):
    key1 = cache_service.build_cache_key("proj-1", "Hello")
    key2 = cache_service.build_cache_key("proj-2", "Hello")
    assert key1 != key2

def test_cache_key_differs_by_query(cache_service):
    key1 = cache_service.build_cache_key("proj-1", "Hello")
    key2 = cache_service.build_cache_key("proj-1", "World")
    assert key1 != key2

def test_cache_key_differs_by_generation_model(cache_service):
    key1 = cache_service.build_cache_key("proj-1", "Hello")
    cache_service.generation_client.generation_model_id = "claude-3-opus"
    key2 = cache_service.build_cache_key("proj-1", "Hello")
    assert key1 != key2

def test_cache_key_differs_by_embedding_model(cache_service):
    key1 = cache_service.build_cache_key("proj-1", "Hello")
    cache_service.embedding_client.embedding_model_id = "text-embedding-3-large"
    key2 = cache_service.build_cache_key("proj-1", "Hello")
    assert key1 != key2

def test_cache_key_differs_by_language(cache_service):
    key1 = cache_service.build_cache_key("proj-1", "Hello")
    cache_service.app_settings.PRIMARY_LANG = "ar"
    key2 = cache_service.build_cache_key("proj-1", "Hello")
    assert key1 != key2

def test_cache_key_is_positive_int64(cache_service):
    key = cache_service.build_cache_key("proj-1", "Hello")
    assert isinstance(key, int)
    assert key > 0
    assert key <= ((1 << 63) - 1)

