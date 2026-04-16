from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles

from .schemes.data import ProcessRequest
from models import ResponseSignal
from models.AssetModel import AssetModel
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemes import DataChunk, Asset

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
    asset_record = await asset_model.create_asset(asset=asset_resource)

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

    if do_reset == 1:
        await chunk_model.delete_chunks_by_project_id(project_id=project.id)

    for asset_id, asset_file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=asset_file_id)

        if file_content is None:
            logger.error("Could not load file content for: %s", asset_file_id)
            failed_files.append(asset_file_id)
            # Bug fix: continue instead of aborting the whole request
            continue

        # Get strategy from app context/settings
        _settings = get_settings()
        chunk_strategy = getattr(_settings, "CHUNK_STRATEGY", "recursive")
        
        # Inject embedding client into process controller if exists
        if hasattr(request.app, "embedding_client"):
            process_controller.embedding_client = request.app.embedding_client

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=asset_file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            chunk_strategy=chunk_strategy,
        )

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

@data_router.post("/process-file/{project_id}")
async def process_single_file(
    request: Request,
    project_id: str,
    process_request: ProcessRequest,
):
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

@data_router.post("/process-all/{project_id}")
async def process_all_files(
    request: Request,
    project_id: str,
    process_request: ProcessRequest,
):
    return await _process_project_files(
        request,
        project_id,
        file_id=None,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
        app_settings=request.app.state.settings if hasattr(request.app.state, "settings") else None,
    )
