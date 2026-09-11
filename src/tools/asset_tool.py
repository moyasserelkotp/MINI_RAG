from typing import Any, Dict
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)

class ListProjectAssetsTool(BaseTool):
    """Tool for listing uploaded files in the project."""
    
    def __init__(self, asset_model, project_id: str):
        self.asset_model = asset_model
        self.project_id = project_id
        
    @property
    def name(self) -> str:
        return "LIST_ASSETS"
        
    @property
    def description(self) -> str:
        return (
            "Lists all the document assets (files) that have been uploaded to the current project. "
            "Returns filenames, sizes, and types. Does not return the content of the files."
        )
        
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "description": "Optional filter by file type (e.g. 'pdf', 'txt')."
                }
            },
            "required": []
        }
        
    async def execute(self, **kwargs) -> Any:
        asset_type = kwargs.get("asset_type")
        
        try:
            # We need the ObjectId for the project to query assets. 
            # If asset_model expects ObjectId, we must provide it. 
            # In asset_model.get_all_project_assets it calls _resolve_project_id so string is fine.
            # But get_all_project_assets REQUIRES asset_type. Wait, we should fetch all if not provided.
            # I will modify how we query to allow all.
            
            query = {"asset_project_id": self.asset_model._resolve_project_id(self.project_id)}
            if asset_type:
                query["asset_type"] = asset_type
                
            records = await self.asset_model.collection.find(query).to_list(length=1000)
            
            assets = []
            for record in records:
                assets.append({
                    "id": str(record.get("_id")),
                    "name": record.get("asset_name"),
                    "type": record.get("asset_type"),
                    "size": record.get("asset_size"),
                    "pushed_at": record.get("asset_pushed_at").isoformat() if record.get("asset_pushed_at") else None
                })
                
            return {
                "success": True,
                "count": len(assets),
                "assets": assets
            }
        except Exception as e:
            logger.error(f"ListProjectAssetsTool error: {e}")
            return {"error": str(e)}
