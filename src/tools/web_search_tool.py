from typing import Any, Dict, List
from .base import BaseTool
import logging
from helpers.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Tool for searching the live internet via Tavily."""

    def __init__(self, max_results: int = 5):
        self._max_results = max_results
        self._settings = get_settings()

    @property
    def name(self) -> str:
        return "WEB_SEARCH"

    @property
    def description(self) -> str:
        return (
            "Searches the live internet for up-to-date information not available in the "
            "project's knowledge base. Use this tool when the user asks about current events, "
            "recent news, real-time data, or topics that are unlikely to be in uploaded documents. "
            "Do NOT use this if the answer is likely already in the project documents."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the internet."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of web results to return (default 5, max 10).",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, **kwargs) -> Any:
        query = kwargs.get("query")
        if not query:
            return {"error": "Missing required parameter: query"}

        max_results = min(int(kwargs.get("max_results", self._max_results)), 10)
        api_key = self._settings.TAVILY_API_KEY
        
        if not api_key:
            return {
                "error": "TAVILY_API_KEY is not set in the configuration or .env file."
            }

        try:
            from tavily import AsyncTavilyClient
            import asyncio
            
            client = AsyncTavilyClient(api_key=api_key)
            response = await client.search(
                query, 
                search_depth="basic", 
                max_results=max_results
            )
            
            results = response.get("results", [])

            if not results:
                return {
                    "success": True,
                    "count": 0,
                    "results": [],
                    "message": "No web results found for this query."
                }

            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                })

            logger.info("WebSearchTool: returned %d results for query '%s'", len(formatted), query)
            return {
                "success": True,
                "count": len(formatted),
                "results": formatted
            }

        except ImportError:
            return {
                "error": "tavily-python package is not installed. Run: pip install tavily-python"
            }
        except Exception as e:
            logger.error("WebSearchTool error: %s", e)
            return {"error": str(e)}
