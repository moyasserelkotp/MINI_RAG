from typing import Any, Dict, List, Optional
from .base import BaseTool
import logging
from helpers.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Tool for searching the live internet via Tavily."""

    def __init__(self, max_results: int = 5):
        self._max_results = max_results
        self._settings = get_settings()
        self._client: Optional[Any] = self._init_client()

    def _init_client(self) -> Optional[Any]:
        """Initialise AsyncTavilyClient once at construction time."""
        api_key = self._settings.TAVILY_API_KEY
        if not api_key:
            logger.warning("WebSearchTool: TAVILY_API_KEY is not set — web search disabled.")
            return None
        try:
            # pyrefly: ignore [missing-import]
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=api_key)
            logger.info("WebSearchTool: Tavily async client initialised.")
            return client
        except ImportError:
            logger.error(
                "WebSearchTool: tavily-python is not installed. "
                "Run: pip install tavily-python"
            )
            return None

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

        if not self._settings.TAVILY_API_KEY:
            return {"error": "TAVILY_API_KEY is not set in the configuration or .env file."}

        if self._client is None:
            return {"error": "tavily-python package is not installed. Run: pip install tavily-python"}

        max_results = min(int(kwargs.get("max_results", self._max_results)), 10)

        try:
            # Reuse the shared client — no new session created per call
            response = await self._client.search(
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

            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
                for r in results
            ]

            logger.info("WebSearchTool: returned %d results for query '%s'", len(formatted), query)
            return {
                "success": True,
                "count": len(formatted),
                "results": formatted
            }

        except Exception as e:
            logger.error("WebSearchTool error: %s", e)
            return {"error": str(e)}
