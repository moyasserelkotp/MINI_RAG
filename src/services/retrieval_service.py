import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from utils.metrics import (
    record_retrieval_latency,
    record_chunks_retrieved,
    record_retrieval_error
)

logger = logging.getLogger(__name__)


def _build_sources(documents: list, full_text: bool = False) -> List[Dict[str, Any]]:
    """Build a sources list from a list of retrieved documents.

    Args:
        documents: list of Qdrant ScoredPoint objects.
        full_text: if True, include the full chunk text (for faithful evaluation).
                   if False, truncate to 300 chars (for API response size).
    """
    sources = []
    for d in documents:
        text = d.payload.get("text", "")
        meta = d.payload.get("metadata") or {}
        sources.append({
            "text": text if full_text else text[:300],
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "section": meta.get("section"),
            "score": round(float(getattr(d, "score", 0.0)), 4),
        })
    return sources


class RetrievalService:
    def __init__(self, vectordb_client, cohere_client, app_settings):
        self.vectordb_client = vectordb_client
        self.cohere_client = cohere_client
        self.app_settings = app_settings

    async def retrieve_and_rerank_context(
        self,
        search_function,
        project,
        search_query: str,
        limit: int,
        use_hybrid: bool,
        score_threshold: Optional[float],
        metadata_filter: Optional[dict],
        use_vector: bool,
        use_rerank: bool,
    ) -> Tuple[list, List[Dict[str, Any]]]:
        """
        Full retrieval pipeline:
          1. Fetch-K: retrieve fetch_k candidates from the vector DB.
          2. Optionally rerank with Cohere Reranker over all candidates.
          3. Return top `limit` documents and their source metadata.

        Sources are built AFTER reranking so the returned list exactly matches
        the documents actually passed to the LLM.
        """
        if not use_vector:
            return [], []

        should_rerank = use_rerank and getattr(self.app_settings, "USE_RERANK", False)

        # Fetch-K strategy: retrieve more candidates than the final limit so the
        # reranker and hybrid search have a meaningful pool to choose from.
        if should_rerank and use_hybrid:
            fetch_k = limit * 20
        elif should_rerank or use_hybrid:
            fetch_k = limit * 10
        else:
            fetch_k = limit

        retrieved_documents = await asyncio.to_thread(
            search_function,
            project=project,
            text=search_query,
            limit=limit,           # used for hybrid fusion slice
            use_hybrid=use_hybrid,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            fetch_limit=fetch_k,   # actual candidate pool size passed to the DB
        )

        if retrieved_documents is None:
            return [], []

        if not retrieved_documents:
            return [], []

        # ── Reranking ─────────────────────────────────────────────────────────
        if should_rerank:
            cohere_client = self.cohere_client
            if cohere_client:
                try:
                    docs_texts = [d.payload.get("text", "") for d in retrieved_documents]
                    rerank_model = getattr(
                        self.app_settings, "RERANK_MODEL_ID", "rerank-multilingual-v3.0"
                    )
                    reranked = await asyncio.to_thread(
                        cohere_client.rerank,
                        model=rerank_model,
                        query=search_query,
                        documents=docs_texts,
                        top_n=min(limit, len(docs_texts)),
                    )
                    reranked_docs = []
                    for r in reranked.results:
                        doc = retrieved_documents[r.index]
                        doc.score = round(float(r.relevance_score), 4)
                        reranked_docs.append(doc)

                    # Use the reranked list — already sliced to `limit` by top_n
                    retrieved_documents = reranked_docs
                    logger.info(
                        "Reranked %d candidates → %d final docs for query: %s",
                        len(docs_texts), len(retrieved_documents), search_query[:80],
                    )
                except Exception as e:
                    logger.error("Cohere Rerank failed: %s — using pre-rerank order", e)
                    # Fall back to original order, capped to `limit`
                    retrieved_documents = retrieved_documents[:limit]
            else:
                logger.warning("USE_RERANK=True but cohere_client is not initialised; skipping rerank.")
                retrieved_documents = retrieved_documents[:limit]
        else:
            # No reranking: cap the result to `limit`
            retrieved_documents = retrieved_documents[:limit]

        # ── Post-rerank threshold filter ──────────────────────────────────────
        # Apply the threshold AFTER reranking so we use Cohere scores (0-1 scale)
        # when reranking was performed, or the normalized RRF scores otherwise.
        # This is more meaningful than filtering raw RRF scores pre-rerank.
        score_threshold_val = getattr(self.app_settings, "SEARCH_SCORE_THRESHOLD", 0.0)
        if score_threshold_val and score_threshold_val > 0 and retrieved_documents:
            original_len = len(retrieved_documents)
            filtered = [d for d in retrieved_documents if getattr(d, "score", 0.0) >= score_threshold_val]
            
            # Always apply the filter. If filtered is empty, it correctly signals
            # NLPController to use the NO_CONTEXT fallback path.
            retrieved_documents = filtered
            
            logger.info(
                "Post-rerank threshold %.3f removed %d low-confidence chunks; %d remain.",
                score_threshold_val,
                original_len - len(filtered),
                len(filtered),
            )

        # ── Build sources AFTER reranking ─────────────────────────────────────
        # This ensures sources exactly reflect what the LLM receives.
        sources = _build_sources(retrieved_documents)

        return retrieved_documents, sources
