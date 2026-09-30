import logging
import asyncio

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Evaluates the quality of retrieved context."""

    def __init__(self, llm_client, score_threshold=0.35, llm_semaphore: asyncio.Semaphore = None,
                 enable_llm_eval: bool = False):
        self.llm_client = llm_client
        self.score_threshold = score_threshold
        # MEDIUM-7 FIX: accept flag via DI instead of calling get_settings() in
        # __init__ — this makes the class unit-testable without a real .env file.
        self._enable_llm_eval = enable_llm_eval
        self._semaphore = llm_semaphore or asyncio.Semaphore(10)

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

        enable_llm_eval = self._enable_llm_eval
        is_complex = query_category == "COMPLEX_MULTI_STEP"
        if not enable_llm_eval or not is_complex or not self.llm_client:
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
            # THREAD-SAFE: generate_structured_output must use a thread-safe HTTP
            # client. See LLMProviderFactory — each provider owns its client.
            async with self._semaphore:
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
            logger.error("RetrievalEvaluator LLM error [%s]: %s", type(e).__name__, e, exc_info=True)

        # Fallback: deterministic score already passed, treat as sufficient
        return {
            "is_sufficient": True,
            "reason": "Fallback evaluation: deterministic score passed; LLM evaluation unavailable.",
            "action": "ANSWER"
        }

