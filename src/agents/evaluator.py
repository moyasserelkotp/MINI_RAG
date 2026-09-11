import logging
import asyncio
from helpers.config import get_settings

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Evaluates the quality of retrieved context."""

    def __init__(self, llm_client, score_threshold=0.35):
        self.llm_client = llm_client
        self.score_threshold = score_threshold
        self._settings = get_settings()

    async def evaluate(self, query: str, retrieved_context: list, query_category: str = None) -> dict:
        """
        Determines if the retrieved context is sufficient to answer the query.
        Returns a dict with 'is_sufficient' (bool), 'reason' (str), and 'action' (str).
        """
        # Fast-pass web search results directly to Answer node
        if query_category == "WEB_SEARCH":
            return {
                "is_sufficient": True,
                "reason": "Web search results are passed directly to answer generation.",
                "action": "ANSWER"
            }

        if not retrieved_context:
            return {
                "is_sufficient": False,
                "reason": "No context retrieved.",
                "action": "REWRITE_QUERY"
            }

        # Fast deterministic check — normalize scores to [0, 1] range
        max_score = max([min(max(c.get("score", 0), 0.0), 1.0) for c in retrieved_context])
        if max_score < self.score_threshold:
            return {
                "is_sufficient": False,
                "reason": f"Max retrieval score {max_score:.2f} is below threshold {self.score_threshold}.",
                "action": "REWRITE_QUERY"
            }

        # Optional LLM-based check — only when explicitly enabled in config
        enable_llm_eval = getattr(self._settings, "ENABLE_RETRIEVAL_EVALUATION", True)
        if not enable_llm_eval or not self.llm_client:
            # Score passed deterministic check — treat as sufficient
            return {
                "is_sufficient": True,
                "reason": f"Deterministic score {max_score:.2f} meets threshold {self.score_threshold}.",
                "action": "ANSWER"
            }

        schema = {
            "type": "object",
            "properties": {
                "is_sufficient": {
                    "type": "boolean",
                    "description": "True if the context contains enough information to answer the query."
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation of why the context is or isn't sufficient."
                }
            },
            "required": ["is_sufficient", "reason"]
        }

        context_str = "\n\n---\n\n".join([c.get("text", "") for c in retrieved_context])

        prompt = f"""Evaluate if the following context contains enough information to accurately and fully answer the user's query.
Do NOT attempt to answer the query, just evaluate the context.

User Query: "{query}"

Retrieved Context:
{context_str}
"""
        try:
            result = await asyncio.to_thread(
                self.llm_client.generate_structured_output,
                prompt=prompt,
                schema=schema
            )

            if result and "is_sufficient" in result:
                is_sufficient = result["is_sufficient"]
                action = "ANSWER" if is_sufficient else "REWRITE_QUERY"
                return {
                    "is_sufficient": is_sufficient,
                    "reason": result.get("reason", "Evaluated by LLM."),
                    "action": action
                }

        except Exception as e:
            logger.error(f"RetrievalEvaluator LLM error: {e}")

        # Fallback: deterministic score already passed, treat as sufficient
        return {
            "is_sufficient": True,
            "reason": "Fallback evaluation: deterministic score passed; LLM evaluation unavailable.",
            "action": "ANSWER"
        }

