import logging
import asyncio

logger = logging.getLogger(__name__)

class AgentPlanner:
    """Creates execution plans for complex queries."""
    
    def __init__(self, llm_client, tool_registry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        
    async def create_plan(self, query: str, chat_history: list = None) -> list:
        """
        Generates a sequence of steps to fulfill a complex query.
        Returns a list of step dictionaries.
        """
        tools_schema = self.tool_registry.get_tools_schema()
        
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "action": {"type": "string"},
                            "tool_kwargs": {"type": "object"}
                        },
                        "required": ["id", "action"]
                    }
                }
            },
            "required": ["steps"]
        }
        
        prompt = f"""You are an intelligent planner agent. Break down the following complex user query into a sequence of executable steps using the available tools.
Keep the plan as simple as possible. Often a single search is enough. 
Only create multiple steps if information must be gathered from distinctly different searches or tools.
The final step should always be an action to synthesize or directly answer if all information is gathered (use action: DIRECT_ANSWER).

User Query: "{query}"

Available Tools:
"""
        for t in tools_schema:
            prompt += f"- {t['name']}: {t['description']}\n  Parameters: {t['parameters']}\n\n"
            
        try:
            result = await asyncio.to_thread(
                self.llm_client.generate_structured_output,
                prompt=prompt,
                schema=schema,
                chat_history=chat_history
            )
            
            if result and "steps" in result:
                return result["steps"]
                
        except Exception as e:
            logger.error(f"AgentPlanner error: {e}")
            
        # Fallback plan
        return [
            {"id": 1, "action": "SEARCH_DOCUMENTS", "tool_kwargs": {"query": query}},
            {"id": 2, "action": "DIRECT_ANSWER"}
        ]
