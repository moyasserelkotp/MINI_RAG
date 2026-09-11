from typing import Any, Dict
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)

class GetConversationContextTool(BaseTool):
    """Tool for retrieving conversation memory."""
    
    def __init__(self, memory_service, project_id: str, session_id: str):
        self.memory_service = memory_service
        self.project_id = project_id
        self.session_id = session_id
        
    @property
    def name(self) -> str:
        return "GET_MEMORY"
        
    @property
    def description(self) -> str:
        return (
            "Retrieves the recent conversation history and conversation summary for the current user session. "
            "Use this if you need to recall what was discussed earlier in the conversation."
        )
        
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
        
    async def execute(self, **kwargs) -> Any:
        if not self.session_id:
            return {"error": "No session ID provided. Cannot retrieve memory."}
            
        try:
            # We use prepare_session which gets window and summary memory
            # The tool itself just returns the raw data for the agent to review
            session_obj, session_messages, entities_text = await self.memory_service.prepare_session(
                session_id=self.session_id,
                project_id=self.project_id,
                query="[TOOL_GET_MEMORY]" # Placeholder query
            )
            
            messages_formatted = []
            for msg in session_messages:
                messages_formatted.append({
                    "role": msg.role,
                    "text": msg.text,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None
                })
                
            return {
                "success": True,
                "session_id": self.session_id,
                "summary": session_obj.summary if session_obj else None,
                "recent_messages": messages_formatted,
                "entities": entities_text if entities_text else None
            }
        except Exception as e:
            logger.error(f"GetConversationContextTool error: {e}")
            return {"error": str(e)}
