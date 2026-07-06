from fastapi import APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
import time

from .schemes.data import ProcessRequest
from models import ResponseSignal
from models.AssetModel import AssetModel
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemes import DataChunk, Asset
from utils.metrics import record_chunking_latency

import os
from bson.objectid import ObjectId
import logging
from models.enums.AssetTypeEnum import AssetTypeEnum

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1/data"],
)


# ── Upload ────────────────────────────────────────────────────────────────────

@data_router.post("/upload/{project_id}")
async def upload_data(
    request: Request,
    project_id: str,
    file: UploadFile,
    app_settings: Settings = Depends(get_settings),
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result_signal},
        )

    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id,
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error("Error while uploading file: %s", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset_resource = Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
    )
    try:
        asset_record = await asset_model.create_asset(asset=asset_resource)
    except Exception as e:
        logger.error("Failed to create asset record in DB: %s", e)
        # Clean up the already-written file to avoid orphaned files on disk
        try:
            os.remove(file_path)
        except OSError:
            pass
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_id": file_id,
            "asset_id": str(asset_record.id),
        }
    )


# ── Core processing logic (shared) ───────────────────────────────────────────

async def _process_project_files(
    request: Request,
    project_id: str,
    *,
    file_id,
    chunk_size: int,
    overlap_size: int,
    do_reset: int,
    app_settings: Settings = None,
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    project_files_ids: dict = {}

    if file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=file_id,
            asset_type=AssetTypeEnum.FILE.value,
        )
        if asset_record is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"signal": ResponseSignal.FILE_ID_ERROR.value},
            )
        project_files_ids = {asset_record.id: asset_record.asset_name}

    else:
        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
        )
        project_files_ids = {record.id: record.asset_name for record in project_files}

    if not project_files_ids:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_ERROR.value},
        )

    process_controller = ProcessController(project_id=project_id)
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    no_records = 0
    no_files = 0
    failed_files = []

    # FIX: resolve settings and inject embedding client ONCE outside the loop
    _settings = get_settings()
    chunk_strategy = getattr(_settings, "CHUNK_STRATEGY", "recursive")
    if hasattr(request.app, "embedding_client"):
        process_controller.embedding_client = request.app.embedding_client

    if do_reset == 1:
        await chunk_model.delete_chunks_by_project_id(project_id=project.id)

    for asset_id, asset_file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=asset_file_id)

        if file_content is None:
            logger.error("Could not load file content for: %s", asset_file_id)
            failed_files.append(asset_file_id)
            # Bug fix: continue instead of aborting the whole request
            continue

        # FIX: time the chunking step and record to Prometheus
        _chunk_start = time.monotonic()
        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=asset_file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            chunk_strategy=chunk_strategy,
        )
        try:
            record_chunking_latency(strategy=chunk_strategy, duration=time.monotonic() - _chunk_start)
        except Exception:
            pass

        if not file_chunks:
            logger.error("No chunks produced for file: %s", asset_file_id)
            failed_files.append(asset_file_id)
            # Bug fix: continue instead of returning 400 on first failure
            continue

        file_chunks_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i + 1,
                chunk_project_id=project.id,
                chunk_asset_id=ObjectId(asset_id) if isinstance(asset_id, str) else asset_id,
            )
            for i, chunk in enumerate(file_chunks)
        ]

        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files += 1

    if no_files == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROCESSING_FAILED.value,
                "failed_files": failed_files,
            },
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files,
            "failed_files": failed_files,
        }
    )


# ── Process single file ───────────────────────────────────────────────────────

@data_router.post(
    "/process-file/{project_id}",
    deprecated=True,  # Use POST /api/v1/tasks/process-file/{project_id} instead
)
async def process_single_file(
    request: Request,
    project_id: str,
    process_request: ProcessRequest,
):
    """DEPRECATED — use POST /api/v1/tasks/process-file/{project_id} for async processing.
    This endpoint blocks the event loop during chunking and is not suitable for production.
    """
    if not process_request.file_id:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "file_id is required for process-file endpoint"},
        )

    return await _process_project_files(
        request,
        project_id,
        file_id=process_request.file_id,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
        app_settings=request.app.state.settings if hasattr(request.app.state, "settings") else None,
    )


# ── Process all files ─────────────────────────────────────────────────────────

@data_router.post(
    "/process-all/{project_id}",
    deprecated=True,  # Use POST /api/v1/tasks/process-all/{project_id} instead
)
async def process_all_files(
    request: Request,
    project_id: str,
    process_request: ProcessRequest,
):
    """DEPRECATED — use POST /api/v1/tasks/process-all/{project_id} for async processing.
    This endpoint blocks the event loop during chunking and is not suitable for production.
    """
    return await _process_project_files(
        request,
        project_id,
        file_id=None,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
        app_settings=request.app.state.settings if hasattr(request.app.state, "settings") else None,
    )


# ── List assets ───────────────────────────────────────────────────────────────

@data_router.get("/assets/{project_id}")
async def list_assets(request: Request, project_id: str):
    """Return all uploaded assets for a project."""
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    assets = await asset_model.get_all_project_assets(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
    )

    return JSONResponse(
        content={
            "signal": "GET_ASSETS_SUCCESS",
            "project_id": project_id,
            "total": len(assets),
            "assets": [
                {
                    "id": str(a.id),
                    "asset_name": a.asset_name,
                    "asset_size": a.asset_size,
                    "asset_type": a.asset_type,
                    "asset_pushed_at": a.asset_pushed_at.isoformat() if getattr(a, "asset_pushed_at", None) else None,
                }
                for a in assets
            ],
        }
    )



# ── Delete single asset ───────────────────────────────────────────────────────

@data_router.delete("/assets/{project_id}/{asset_id}")
async def delete_asset(request: Request, project_id: str, asset_id: str):
    """Delete a single asset: removes the MongoDB record, its text chunks, and the
    physical file on disk. Note: vector embeddings in the vector DB become orphaned
    and will be cleaned up on the next full re-index (nlp/index/push with do_reset=1).
    """
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset = await asset_model.get_asset_by_id(asset_id=asset_id)

    if not asset:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseSignal.ASSET_NOT_FOUND.value},
        )

    # Validate asset belongs to this project
    if str(asset.asset_project_id) != str(project.id):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"signal": ResponseSignal.DELETE_ASSET_ERROR.value, "error": "Asset does not belong to this project"},
        )

    # 1. Delete MongoDB chunks for this asset
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
    deleted_chunks = await chunk_model.delete_chunks_by_asset_id(asset_id=asset.id)

    # 2. Delete the physical file
    data_controller = DataController()
    file_deleted = data_controller.delete_file_by_name(file_id=asset.asset_name)
    if not file_deleted:
        logger.warning("Physical file not found for asset %s — may have already been deleted", asset_id)

    # 3. Delete the asset record from MongoDB
    deleted = await asset_model.delete_asset_by_id(asset_id=asset_id)

    if not deleted:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.DELETE_ASSET_ERROR.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.DELETE_ASSET_SUCCESS.value,
            "asset_id": asset_id,
            "asset_name": asset.asset_name,
            "deleted_chunks": deleted_chunks,
        }
    )

