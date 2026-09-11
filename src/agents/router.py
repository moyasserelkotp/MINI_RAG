import logging
import asyncio

logger = logging.getLogger(__name__)

class AgentRouter:
    """Decides the next action for the agent based on the state."""
    
    def __init__(self, llm_client, tool_registry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        
    async def route(self, query: str, category: str, chat_history: list = None) -> dict:
        """
        Determines the next tool to call or action to take.
        Returns a dict with 'action', 'reason', and optionally 'tool_kwargs'.
        """
        
        # Fast path based on category
        if category == "PROJECT_METADATA":
            return {"action": "GET_PROJECT_INFO", "reason": "Query asks about project metadata."}
        if category == "ASSET_METADATA":
            return {"action": "LIST_ASSETS", "reason": "Query asks for list of files/assets."}
        if category == "GENERAL_CONVERSATION":
            return {"action": "DIRECT_ANSWER", "reason": "General conversational query."}
        if category == "WEB_SEARCH":
            return {"action": "WEB_SEARCH", "reason": "Query requires live internet information.", "tool_kwargs": {"query": query}}
            
        # For DOCUMENT_QUESTION or CONVERSATION_REFERENCE, we use the LLM to decide
        # exactly how to use the search or memory tools.
        
        tools_schema = self.tool_registry.get_tools_schema()
        
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The name of the tool to use, or 'DIRECT_ANSWER' if no tool is needed."
                },
                "reason": {
                    "type": "string",
                    "description": "Why this action was chosen."
                },
                "tool_kwargs": {
                    "type": "object",
                    "description": "Arguments to pass to the tool, if any. Omit if action is DIRECT_ANSWER."
                }
            },
            "required": ["action", "reason"]
        }
        
        prompt = f"""You are an intelligent routing agent. Given a user query and a list of available tools, 
decide the best tool to use to fulfill the user's request.
If you can answer the query without any tools (e.g., general greetings), choose the action "DIRECT_ANSWER".

User Query: "{query}"

Available Tools:
"""
        for t in tools_schema:
            prompt += f"- {t['name']}: {t['description']}\n  Parameters: {t['parameters']}\n\n"
            
        prompt += "Choose the best action and provide the necessary tool arguments."
        
        try:
            result = await asyncio.to_thread(
                self.llm_client.generate_structured_output,
                prompt=prompt,
                schema=schema,
                chat_history=chat_history
            )
            
            if result and "action" in result:
                return result
                
        except Exception as e:
            logger.error(f"AgentRouter error: {e}")
            
        # Safe fallback
        return {
            "action": "SEARCH_DOCUMENTS",
            "reason": "Fallback routing.",
            "tool_kwargs": {"query": query}
        }
