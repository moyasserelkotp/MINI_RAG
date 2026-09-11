from typing import Any, Dict
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)

class GetProjectInfoTool(BaseTool):
    """Tool for fetching project metadata."""
    
    def __init__(self, project_model, project_id: str):
        self.project_model = project_model
        self.project_id = project_id
        
    @property
    def name(self) -> str:
        return "GET_PROJECT_INFO"
        
    @property
    def description(self) -> str:
        return (
            "Retrieves metadata about the current project, such as its name, creation date, "
            "and other configured settings. Does not retrieve document contents."
        )
        
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
        
    async def execute(self, **kwargs) -> Any:
        try:
            project = await self.project_model.get_project_by_id(self.project_id)
            if not project:
                return {"error": "Project not found"}
                
            return {
                "success": True,
                "project_id": str(project.project_id),
                "project_name": getattr(project, "project_name", None) or str(project.project_id),
                "created_at": project.project_created_at.isoformat() if project.project_created_at else None,
                "document_count": project.project_documents_count,
            }
        except Exception as e:
            logger.error(f"GetProjectInfoTool error: {e}")
            return {"error": str(e)}
