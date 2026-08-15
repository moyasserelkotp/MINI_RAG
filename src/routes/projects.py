from fastapi import APIRouter, Request, status, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from .schemes.project import ProjectListResponse, ProjectItem, ProjectRequest, ProjectResponse
from .schemes.system import BaseResponse
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from controllers.NLPController import NLPController
from models import ResponseSignal
import logging

logger = logging.getLogger("uvicorn.error")

_PROJECT_ID_PATH = Path(
    ...,
    regex=r"^[a-zA-Z0-9_-]{1,64}$",
    description="Project identifier (alphanumeric, underscores, hyphens, max 64 chars)",
)

projects_router = APIRouter(
    prefix="/api/v1/projects",
    tags=["Projects"],
)
@projects_router.post("/", summary="Create a new project", response_model=ProjectResponse)
async def create_project(request: Request, project_req: ProjectRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    
    project = await project_model.get_project_or_create_one(project_id=project_req.project_id)
    
    return ProjectResponse(
        signal="PROJECT_CREATED",
        project=ProjectItem(
            id=str(project.id),
            project_id=project.project_id
        )
    )

@projects_router.get("/", summary="List all projects", response_model=ProjectListResponse)
async def list_projects(
    request: Request,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    projects, total_pages = await project_model.get_all_projects(page=page, page_size=page_size)

    return ProjectListResponse(
        signal="PROJECTS_RETRIEVED",
        page=page,
        total_pages=total_pages,
        projects=[
            ProjectItem(
                id=str(p.id),
                project_id=p.project_id
            )
            for p in projects
        ]
    )


@projects_router.delete("/{project_id}", summary="Delete a project and all its data", response_model=BaseResponse)
async def delete_project(request: Request, project_id: str = _PROJECT_ID_PATH):
    """Deletes the project record, all its assets, all its chunks, and the vector collection."""
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    # CR-02: use get_project_by_id so missing projects correctly return 404
    project = await project_model.get_project_by_id(project_id=project_id)

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    errors = []

    # Delete vector DB collection
    try:
        nlp_controller = NLPController(
            db_client=request.app.db_client,
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
        )
        nlp_controller.reset_vector_db_collection(project=project)
    except Exception as e:
        logger.error("Failed to delete vector collection for %s: %s", project_id, e)
        errors.append("vectordb")

    # Delete chunks
    try:
        chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
        await chunk_model.delete_chunks_by_project_id(project_id=project.id)
    except Exception as e:
        logger.error("Failed to delete chunks for %s: %s", project_id, e)
        errors.append("chunks")

    # Delete assets
    try:
        asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
        await asset_model.delete_all_project_assets(asset_project_id=project.id)
    except Exception as e:
        logger.error("Failed to delete assets for %s: %s", project_id, e)
        errors.append("assets")

    # Delete project record
    try:
        await project_model.delete_project(project_id=project_id)
    except Exception as e:
        logger.error("Failed to delete project record %s: %s", project_id, e)
        errors.append("project_record")

    if errors:
        raise HTTPException(
            status_code=status.HTTP_207_MULTI_STATUS,
            detail={
                "signal": ResponseSignal.DELETE_PROJECT_ERROR.value,
                "partial_errors": errors,
                "project_id": project_id,
            },
        )

    return BaseResponse(
        signal=ResponseSignal.DELETE_PROJECT_SUCCESS.value
    )
