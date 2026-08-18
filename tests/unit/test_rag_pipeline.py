import pytest
from unittest.mock import MagicMock, AsyncMock
from services.retrieval_service import RetrievalService

class FakeDoc:
    def __init__(self, text, source="doc.pdf", score=0.85, page=1):
        self.payload = {"text": text, "metadata": {"source": source, "page": page}}
        self.score = score

@pytest.fixture
def retrieval_service():
    vectordb_client = MagicMock()
    cohere_client = MagicMock()
    
    class Settings:
        USE_RERANK = True
        RERANK_MODEL_ID = "rerank-v3"
        
    return RetrievalService(
        vectordb_client=vectordb_client,
        cohere_client=cohere_client,
        app_settings=Settings()
    )

@pytest.mark.asyncio
async def test_retrieve_and_rerank_no_vector(retrieval_service):
    docs, sources = await retrieval_service.retrieve_and_rerank_context(
        search_function=MagicMock(),
        project=MagicMock(),
        search_query="test",
        limit=5,
        use_hybrid=False,
        score_threshold=0.5,
        metadata_filter=None,
        use_vector=False,
        use_rerank=False
    )
    assert docs == []
    assert sources == []

@pytest.mark.asyncio
async def test_retrieve_and_rerank_with_cohere(retrieval_service):
    raw_docs = [
        FakeDoc(text="Paragraph 1 about Luxor", score=0.7),
        FakeDoc(text="Paragraph 2 about Cairo", score=0.8)
    ]
    
    search_fn = MagicMock(return_value=raw_docs)
    
    class FakeRerankResult:
        def __init__(self, index, relevance_score):
            self.index = index
            self.relevance_score = relevance_score
            
    class FakeRerankResponse:
        results = [
            FakeRerankResult(index=0, relevance_score=0.98),
            FakeRerankResult(index=1, relevance_score=0.65)
        ]
        
    retrieval_service.cohere_client.rerank.return_value = FakeRerankResponse()
    
    docs, sources = await retrieval_service.retrieve_and_rerank_context(
        search_function=search_fn,
        project=MagicMock(),
        search_query="Luxor monuments",
        limit=2,
        use_hybrid=True,
        score_threshold=0.5,
        metadata_filter=None,
        use_vector=True,
        use_rerank=True
    )
    
    assert len(docs) == 2
    assert docs[0].score == 0.98
    assert sources[0]["score"] == 0.98
    assert "Luxor" in sources[0]["text"]

@pytest.mark.asyncio
async def test_retrieve_and_rerank_cohere_failure_fallback(retrieval_service):
    raw_docs = [
        FakeDoc(text="Fallback paragraph", score=0.75)
    ]
    search_fn = MagicMock(return_value=raw_docs)
    retrieval_service.cohere_client.rerank.side_effect = Exception("Cohere API rate limit exceeded")
    
    docs, sources = await retrieval_service.retrieve_and_rerank_context(
        search_function=search_fn,
        project=MagicMock(),
        search_query="Fallback test",
        limit=1,
        use_hybrid=False,
        score_threshold=0.5,
        metadata_filter=None,
        use_vector=True,
        use_rerank=True
    )
    
    # Graceful fallback: original docs preserved despite reranker exception
    assert len(docs) == 1
    assert docs[0].score == 0.75
    assert len(sources) == 1
