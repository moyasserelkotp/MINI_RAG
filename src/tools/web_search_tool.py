from typing import Any, Dict, List
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Tool for searching the live internet via DuckDuckGo (no API key required)."""

    def __init__(self, max_results: int = 5):
        self._max_results = max_results

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

        try:
            # pyrefly: ignore [missing-import]
            from ddgs import DDGS
            import asyncio

            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            results = await asyncio.to_thread(_search)

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
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

            logger.info("WebSearchTool: returned %d results for query '%s'", len(formatted), query)
            return {
                "success": True,
                "count": len(formatted),
                "results": formatted
            }

        except ImportError:
            return {
                "error": "duckduckgo-search package is not installed. Run: pip install duckduckgo-search"
            }
        except Exception as e:
            logger.error("WebSearchTool error: %s", e)
            return {"error": str(e)}
