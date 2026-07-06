from fastapi import APIRouter, Depends, UploadFile, File, status, Request
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
import time

from .schemes.data import ProcessRequest, URLIngestRequest
from models import ResponseSignal
from models.AssetModel import AssetModel
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemes import DataChunk, Asset
from utils.metrics import record_chunking_latency

import os
import re
import hashlib
from typing import List as ListType
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

# ── Phase 9: Batch Upload ─────────────────────────────────────────────────────

@data_router.post("/upload/batch/{project_id}", summary="Upload multiple files at once")
async def upload_batch(
    request: Request,
    project_id: str,
    files: ListType[UploadFile] = File(...),
    app_settings: Settings = Depends(get_settings),
):
    if len(files) > 20:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "BATCH_UPLOAD_LIMIT_EXCEEDED", "error": "Maximum 20 files allowed per batch"}
        )
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    data_controller = DataController()
    asset_model = await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    results = []
    for file in files:
        is_valid, signal = data_controller.validate_uploaded_file(file=file)
        if not is_valid:
            results.append({"filename": file.filename, "status": "failed", "signal": signal.value})
            continue

        file_path, file_id = data_controller.generate_unique_filepath(
            orig_file_name=file.filename, project_id=project_id
        )

        try:
            async with aiofiles.open(file_path, "wb") as f:
                while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                    await f.write(chunk)
        except Exception as e:
            logger.error(f"Error while uploading file {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "failed", "signal": ResponseSignal.FILE_UPLOAD_FAILED.value})
            continue

        asset = Asset(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_name=file_id,
            asset_size=os.path.getsize(file_path),
        )
        try:
            asset_record = await asset_model.create_asset(asset=asset)
        except Exception as e:
            logger.error("Failed to create asset record in DB: %s", e)
            try:
                os.remove(file_path)
            except OSError:
                pass
            results.append({"filename": file.filename, "status": "failed", "signal": "ASSET_CREATION_FAILED"})
            continue

        results.append({
            "filename": file.filename,
            "status": "success",
            "file_id": str(asset_record.id),
            "asset_name": file_id
        })

    return JSONResponse(
        content={
            "signal": "BATCH_UPLOAD_COMPLETED",
            "project_id": project_id,
            "results": results
        }
    )


# ── Phase 7: URL Ingestion ────────────────────────────────────────────────────

@data_router.post("/ingest/url/{project_id}", summary="Ingest content from a public URL")
async def ingest_url(
    request: Request,
    project_id: str,
    ingest_request: URLIngestRequest,
):
    import httpx
    from bs4 import BeautifulSoup
    import urllib.parse
    import ipaddress
    import asyncio

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    # 1. SSRF Protection & Fetch URL
    parsed = urllib.parse.urlparse(ingest_request.url)
    if not parsed.hostname:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"signal": "INVALID_URL", "error": "Invalid URL provided"})

    try:
        loop = asyncio.get_running_loop()
        addr_info = await loop.getaddrinfo(parsed.hostname, None)
        for res in addr_info:
            ip = ipaddress.ip_address(res[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    content={"signal": "URL_BLOCKED", "error": "Private or internal IPs are not allowed"}
                )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, 
            content={"signal": "URL_RESOLUTION_FAILED", "error": f"Could not resolve hostname: {e}"}
        )

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            async with client.stream("GET", ingest_request.url) as resp:
                resp.raise_for_status()
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > 5 * 1024 * 1024:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST, 
                        content={"signal": "FILE_TOO_LARGE", "error": "URL content exceeds 5MB limit"}
                    )
                
                chunks = []
                bytes_downloaded = 0
                async for chunk in resp.aiter_bytes():
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > 5 * 1024 * 1024:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST, 
                            content={"signal": "FILE_TOO_LARGE", "error": "URL content exceeds 5MB limit"}
                        )
                    chunks.append(chunk)
                html_content = b"".join(chunks).decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to fetch URL {ingest_request.url}: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "URL_FETCH_FAILED", "error": str(e)}
        )

    # 2. Extract text using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove scripts, styles
    for script_or_style in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        script_or_style.decompose()

    text = soup.get_text(separator="\n")
    # Clean up whitespace
    text = re.sub(r'\n+', '\n', text).strip()

    if not text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "URL_EXTRACT_FAILED", "error": "No text content found"}
        )

    # 3. Create a pseudo-Asset to represent the URL
    url_hash = hashlib.md5(ingest_request.url.encode()).hexdigest()[:12]
    asset_name = f"url_{url_hash}"

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    # Check if exists and do_reset is requested
    existing_asset = await asset_model.collection.find_one({
        "asset_project_id": project.id,
        "asset_name": asset_name
    })

    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    if existing_asset:
        if not ingest_request.do_reset:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": "URL_ALREADY_INGESTED", "asset_id": str(existing_asset["_id"])}
            )
        # Reset: delete old chunks
        await chunk_model.delete_chunks_by_asset_id(asset_id=existing_asset["_id"])
        asset_id = existing_asset["_id"]
    else:
        new_asset = Asset(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,  # treating URL as a file for simplicity
            asset_name=asset_name,
            asset_size=len(text),
        )
        new_asset_record = await asset_model.create_asset(asset=new_asset)
        asset_id = new_asset_record.id

    # 4. Chunk text directly in a separate thread to unblock event loop
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    doc = Document(page_content=text, metadata={"source": ingest_request.url, "type": "url"})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ingest_request.chunk_size,
        chunk_overlap=ingest_request.overlap_size,
    )
    chunks = await asyncio.to_thread(splitter.split_documents, [doc])

    file_chunks = []
    for chunk in chunks:
        file_chunks.append(
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=1,
                chunk_project_id=project.id,
                chunk_asset_id=ObjectId(asset_id),
            )
        )

    if file_chunks:
        inserted_count = await chunk_model.insert_many_chunks(chunks=file_chunks)
    else:
        inserted_count = 0

    return JSONResponse(
        content={
            "signal": "URL_INGESTION_SUCCESS",
            "url": ingest_request.url,
            "inserted_chunks": inserted_count,
            "asset_id": str(asset_id)
        }
    )

