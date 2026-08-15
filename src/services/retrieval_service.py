import asyncio
import logging
import time
from typing import Optional
from utils.metrics import (
    record_retrieval_latency,
    record_chunks_retrieved,
    record_retrieval_error
)

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, vectordb_client, cohere_client, app_settings):
        self.vectordb_client = vectordb_client
        self.cohere_client = cohere_client
        self.app_settings = app_settings

    async def retrieve_and_rerank_context(self, search_function, project, search_query, limit, use_hybrid, score_threshold, metadata_filter, use_vector, use_rerank):
        if not use_vector:
            return [], []
        
        retrieved_documents = await asyncio.to_thread(
            search_function,
            project=project, text=search_query, limit=limit,
            use_hybrid=use_hybrid, score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )
        if retrieved_documents is None:
            return [], []

        sources = [
            {
                "text": d.payload.get("text", "")[:300],
                "source": (d.payload.get("metadata") or {}).get("source", "unknown"),
                "page": (d.payload.get("metadata") or {}).get("page"),
                "score": round(float(getattr(d, "score", 0.0)), 4),
            }
            for d in retrieved_documents
        ]

        should_rerank = use_rerank and getattr(self.app_settings, "USE_RERANK", False)
        if should_rerank and retrieved_documents:
            try:
                cohere_client = self.cohere_client
                if cohere_client:
                    docs_texts = [d.payload.get("text", "") for d in retrieved_documents]
                    rerank_model = getattr(self.app_settings, "RERANK_MODEL_ID", "rerank-multilingual-v3.0")
                    reranked = await asyncio.to_thread(
                        cohere_client.rerank,
                        model=rerank_model, query=search_query, documents=docs_texts, top_n=min(limit, len(docs_texts))
                    )
                    reranked_docs = []
                    for r in reranked.results:
                        d = retrieved_documents[r.index]
                        d.score = float(r.relevance_score)
                        reranked_docs.append(d)
                    retrieved_documents = reranked_docs
                    sources = [
                        {
                            "text": d.payload.get("text", "")[:300],
                            "source": (d.payload.get("metadata") or {}).get("source", "unknown"),
                            "page": (d.payload.get("metadata") or {}).get("page"),
                            "score": round(float(getattr(d, "score", 0.0)), 4),
                        }
                        for d in reranked_docs
                    ]
                else:
                    logger.warning("USE_RERANK=True but cohere_client is not initialised; skipping rerank.")
            except Exception as e:
                logger.error("Cohere Rerank failed: %s", e)

        return retrieved_documents, sources
