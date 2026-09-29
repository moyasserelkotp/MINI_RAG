from typing import Any, Dict, List, Optional
from .base import BaseTool
import logging
import asyncio
from helpers.config import get_settings

logger = logging.getLogger(__name__)

class SearchDocumentsTool(BaseTool):
    """Tool for semantic search over project documents."""
    
    def __init__(self, retrieval_service, nlp_controller, project, llm_semaphore: asyncio.Semaphore = None):
        self.retrieval_service = retrieval_service
        self.nlp_controller = nlp_controller
        self.project = project
        self._semaphore = llm_semaphore or asyncio.Semaphore(10)
        
    @property
    def name(self) -> str:
        return "SEARCH_DOCUMENTS"
        
    @property
    def description(self) -> str:
        return (
            "Searches the project's knowledge base for relevant document chunks based on semantic meaning. "
            "Use this tool to find factual information, answer user questions from documents, or get context. "
            "It automatically handles hybrid search and reranking."
        )
        
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to match against documents."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of chunks to return. Default is 5.",
                    "default": 5
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum similarity score (0.0 to 1.0). Leave null for default.",
                },
                "filter_metadata": {
                    "type": "object",
                    "description": "Optional dictionary for exact match metadata filtering (e.g. {'source': 'file.pdf'})."
                }
            },
            "required": ["query"]
        }
        
    async def execute(self, **kwargs) -> Any:
        query = kwargs.get("query")
        if not query:
            return {"error": "Missing required parameter: query"}
            
        limit = kwargs.get("limit", 5)
        score_threshold = kwargs.get("score_threshold")
        filter_metadata = kwargs.get("filter_metadata")
        
        # 1. Generate query variations for Multi-Query Search
        queries = [query]
        settings = get_settings()
        if getattr(settings, "ENABLE_MULTI_QUERY", False):
            try:
                # We use the generation client to get variations
                schema = {
                    "type": "object",
                    "properties": {
                        "variations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Up to 3 variations of the original search query."
                        }
                    },
                    "required": ["variations"]
                }
                
                prompt = f"""You are an AI assistant tasked with generating search queries to improve retrieval from a vector database.
Your goal is to generate up to 3 variations of the given user query. These variations should use different keywords, synonyms, or rephrasings to capture the same intent but match different potential documents.
Do not answer the query, just return the variations.

Original Query: "{query}"
"""
                async with self._semaphore:
                    result = await asyncio.to_thread(
                        self.nlp_controller.generation_client.generate_structured_output,
                        prompt=prompt,
                        schema=schema
                    )
                
                if result and "variations" in result:
                    variations = result.get("variations", [])
                    if isinstance(variations, list):
                        # Filter out empty strings and exact duplicates
                        valid_variations = [v for v in variations if isinstance(v, str) and v.strip() and v.strip().lower() != query.lower()]
                        queries.extend(valid_variations[:3])
                        
                logger.info(f"Multi-Query Search will use {len(queries)} queries: {queries}")
            except Exception as e:
                logger.warning(f"Failed to generate query variations, falling back to original query: {e}")
        
        try:
            docs, sources = await self.retrieval_service.retrieve_and_rerank_context(
                search_function=self.nlp_controller.search_vector_db_collection,
                project=self.project,
                search_query=queries,
                limit=limit,
                use_hybrid=True,
                score_threshold=score_threshold,
                metadata_filter=filter_metadata,
                use_vector=True,
                use_rerank=True,
            )
            
            # Format results for the agent
            formatted_results = []
            for doc in docs:
                # Assuming doc is a Qdrant ScoredPoint or similar with payload
                payload = getattr(doc, "payload", {})
                score = getattr(doc, "score", 0.0)
                formatted_results.append({
                    "text": payload.get("text", ""),
                    "score": round(score, 4),
                    "metadata": payload.get("metadata", {})
                })
                
            return {
                "success": True,
                "count": len(formatted_results),
                "results": formatted_results
            }
        except Exception as e:
            logger.error(f"SearchDocumentsTool error: {e}")
            return {"error": str(e)}
