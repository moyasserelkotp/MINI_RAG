from fastapi import APIRouter, Request, status, Query
from fastapi.responses import JSONResponse
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel
from controllers.NLPController import NLPController
from models import ResponseSignal
import logging

logger = logging.getLogger("uvicorn.error")

projects_router = APIRouter(
    prefix="/api/v1/projects",
    tags=["api_v1/projects"],
)


@projects_router.get("/", summary="List all projects")
async def list_projects(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    projects, total_pages = await project_model.get_all_projects(page=page, page_size=page_size)

    return JSONResponse(
        content={
            "signal": ResponseSignal.LIST_PROJECTS_SUCCESS.value,
            "page": page,
            "total_pages": total_pages,
            "projects": [
                {"id": str(p.id), "project_id": p.project_id}
                for p in projects
            ],
        }
    )


@projects_router.delete("/{project_id}", summary="Delete a project and all its data")
async def delete_project(request: Request, project_id: str):
    """Deletes the project record, all its assets, all its chunks, and the vector collection."""
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    errors = []

    # Delete vector DB collection
    try:
        nlp_controller = NLPController(
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
        return JSONResponse(
            status_code=status.HTTP_207_MULTI_STATUS,
            content={
                "signal": ResponseSignal.DELETE_PROJECT_ERROR.value,
                "partial_errors": errors,
                "project_id": project_id,
            },
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.DELETE_PROJECT_SUCCESS.value,
            "project_id": project_id,
        }
    )
