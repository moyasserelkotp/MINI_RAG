from pydantic import BaseModel, Field
from typing import Optional, List
from .system import BaseResponse

class ProjectRequest(BaseModel):
    project_id: str = Field(..., min_length=1, description="The unique identifier for the project")
    description: Optional[str] = Field(None, description="Optional description of the project")

class ProjectItem(BaseModel):
    id: str
    project_id: str

class ProjectListResponse(BaseResponse):
    page: int
    total_pages: int
    projects: List[ProjectItem]

class DeleteProjectResponse(BaseResponse):
    project_id: str
    partial_errors: Optional[List[str]] = None

class ProjectResponse(BaseResponse):
    project: ProjectItem
