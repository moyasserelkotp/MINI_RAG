import pytest
from unittest.mock import MagicMock, AsyncMock
from services.cache_service import CacheService

@pytest.fixture
def cache_service():
    vectordb_client = MagicMock()
    embedding_client = MagicMock()
    generation_client = MagicMock()
    generation_client.generation_model_id = "gpt-4-turbo"
    embedding_client.embedding_size = 1536
    embedding_client.embedding_model_id = "text-embedding-3-small"
    
    class Settings:
        PRIMARY_LANG = "en"
        
    return CacheService(
        vectordb_client=vectordb_client,
        embedding_client=embedding_client,
        generation_client=generation_client,
        app_settings=Settings(),
        initialized_collections=set()
    )

@pytest.mark.asyncio
async def test_cache_search_by_vector_error_graceful(cache_service):
    # Vector DB throws connection error during cache retrieval
    cache_service.vectordb_client.is_collection_existed.return_value = True
    cache_service.vectordb_client.search_by_vector.side_effect = Exception("Qdrant connection dropped")
    
    result = await cache_service.check_semantic_cache(
        project_id="proj-1",
        cache_vec=[0.1] * 1536,
        use_cache=True,
        cache_threshold=0.9
    )
    assert result is None

@pytest.mark.asyncio
async def test_cache_disabled_returns_none(cache_service):
    result = await cache_service.check_semantic_cache(
        project_id="proj-1",
        cache_vec=[0.1] * 1536,
        use_cache=False,
        cache_threshold=0.9
    )
    assert result is None
    cache_service.vectordb_client.search_by_vector.assert_not_called()
