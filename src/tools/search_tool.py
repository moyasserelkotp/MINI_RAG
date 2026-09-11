from typing import Any, Dict, List, Optional
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)

class SearchDocumentsTool(BaseTool):
    """Tool for semantic search over project documents."""
    
    def __init__(self, retrieval_service, nlp_controller, project):
        self.retrieval_service = retrieval_service
        self.nlp_controller = nlp_controller
        self.project = project
        
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
        
        try:
            docs, sources = await self.retrieval_service.retrieve_and_rerank_context(
                search_function=self.nlp_controller.search_vector_db_collection,
                project=self.project,
                search_query=query,
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
