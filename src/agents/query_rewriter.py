import logging
import asyncio

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Rewrites queries for better retrieval."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def rewrite(self, query: str, chat_history: list = None) -> str:
        """
        Rewrites the query to be standalone, resolving pronouns and context
        from the chat history.
        """
        if not chat_history or len(chat_history) == 0:
            return query
            
        schema = {
            "type": "object",
            "properties": {
                "rewritten_query": {
                    "type": "string",
                    "description": "The standalone, rewritten query."
                }
            },
            "required": ["rewritten_query"]
        }
        
        prompt = f"""Given the following conversation history and a follow-up user query, 
rewrite the follow-up query to be a standalone query that can be understood without the conversation history.
Resolve any pronouns or implicit references. 
If the query is already standalone, return it as is. Do NOT answer the query.

Follow-up Query: "{query}"
"""
        try:
            result = await asyncio.to_thread(
                self.llm_client.generate_structured_output,
                prompt=prompt,
                schema=schema,
                chat_history=chat_history
            )
            
            if result and "rewritten_query" in result:
                return result["rewritten_query"]
                
        except Exception as e:
            logger.error(f"QueryRewriter error: {e}")
            
        return query
