from fastapi import APIRouter, Depends, UploadFile, File, status, Request, HTTPException, Path
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
import time

from .schemes.data import (
    ProcessRequest, URLIngestRequest, AssetUploadResponse, 
    AssetListResponse, BatchUploadResponse, URLIngestResponse, 
    AssetItem, BatchUploadResult
)
from .schemes.system import BaseResponse
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
from middleware.rate_limiter import limiter

_PROJECT_ID = Path(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["Data Management"],
)


# Upload 
@data_router.post("/upload/{project_id}", response_model=AssetUploadResponse)
@limiter.limit("30/minute")
async def upload_data(
    request: Request,
    file: UploadFile,
    project_id: str = _PROJECT_ID,
    app_settings: Settings = Depends(get_settings),
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": result_signal},
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
        )

    return AssetUploadResponse(
        signal=ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        file_id=file_id,
        asset_id=str(asset_record.id),
    )


# List assets
@data_router.get("/assets/{project_id}", response_model=AssetListResponse)
@limiter.limit("60/minute")
async def list_assets(request: Request, project_id: str = _PROJECT_ID):
    """Return all uploaded assets for a project."""
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    assets = await asset_model.get_all_project_assets(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
    )

    return AssetListResponse(
        signal="GET_ASSETS_SUCCESS",
        project_id=project_id,
        total=len(assets),
        assets=[
            AssetItem(
                id=str(a.id),
                asset_name=a.asset_name,
                asset_size=a.asset_size,
                asset_type=a.asset_type,
                asset_pushed_at=a.asset_pushed_at.isoformat() if getattr(a, "asset_pushed_at", None) else None,
            )
            for a in assets
        ],
    )



#  Delete single asset 
@data_router.delete("/assets/{project_id}/{asset_id}", response_model=BaseResponse)
@limiter.limit("60/minute")
async def delete_asset(request: Request, asset_id: str, project_id: str = _PROJECT_ID):
    """Delete a single asset: removes the MongoDB record, its text chunks, and the
    physical file on disk. Note: vector embeddings in the vector DB become orphaned
    and will be cleaned up on the next full re-index (nlp/index/push with do_reset=1).
    """
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset = await asset_model.get_asset_by_id(asset_id=asset_id)

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": ResponseSignal.ASSET_NOT_FOUND.value},
        )

    # Validate asset belongs to this project
    if str(asset.asset_project_id) != str(project.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"signal": ResponseSignal.DELETE_ASSET_ERROR.value, "error": "Asset does not belong to this project"},
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": ResponseSignal.DELETE_ASSET_ERROR.value},
        )
    
    return BaseResponse(
        signal="DELETE_ASSET_SUCCESS"
    )

#  Phase 9: Batch Upload
@data_router.post("/upload/batch/{project_id}", summary="Upload multiple files at once", response_model=BatchUploadResponse)
@limiter.limit("10/minute")
async def upload_batch(
    request: Request,
    project_id: str = _PROJECT_ID,
    files: ListType[UploadFile] = File(...),
    app_settings: Settings = Depends(get_settings),
):
    if len(files) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": "BATCH_UPLOAD_LIMIT_EXCEEDED", "error": "Maximum 20 files allowed per batch"}
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
            logger.error("Error while uploading file %s: %s", file.filename, e)
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

        results.append(BatchUploadResult(
            filename=file.filename,
            status="success",
            file_id=str(asset_record.id),
            asset_name=file_id
        ))

    return BatchUploadResponse(
        signal="BATCH_UPLOAD_COMPLETED",
        project_id=project_id,
        results=results
    )


#  Phase 7: URL Ingestion
@data_router.post("/ingest/url/{project_id}", summary="Ingest content from a public URL", response_model=URLIngestResponse)
@limiter.limit("10/minute")
async def ingest_url(
    request: Request,
    ingest_request: URLIngestRequest,
    project_id: str = _PROJECT_ID,
):
    import httpx
    from bs4 import BeautifulSoup
    import urllib.parse
    import ipaddress
    import asyncio

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    # 1a. URL scheme validation — only http/https allowed
    parsed_early = urllib.parse.urlparse(ingest_request.url)
    if parsed_early.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": "INVALID_URL_SCHEME", "error": "Only http:// and https:// URLs are allowed"}
        )

    # 1b. SSRF Protection & Fetch URL
    parsed = urllib.parse.urlparse(ingest_request.url)
    if not parsed.hostname:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"signal": "INVALID_URL", "error": "Invalid URL provided"})

    try:
        loop = asyncio.get_running_loop()
        addr_info = await loop.getaddrinfo(parsed.hostname, None)
        for res in addr_info:
            ip = ipaddress.ip_address(res[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local or 
                ip.is_multicast or ip.is_unspecified or ip.is_reserved):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail={"signal": "URL_BLOCKED", "error": "Private, internal, or reserved IPs are not allowed"}
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={"signal": "URL_RESOLUTION_FAILED", "error": f"Could not resolve hostname: {e}"}
        )

    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=15.0) as client:
            async with client.stream("GET", ingest_request.url) as resp:
                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    raise HTTPException(400, detail={"signal": "REDIRECT_NOT_ALLOWED", "error": "Redirects are not allowed for security reasons"})
                resp.raise_for_status()
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > 5 * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST, 
                        detail={"signal": "FILE_TOO_LARGE", "error": "URL content exceeds 5MB limit"}
                    )
                
                chunks = []
                bytes_downloaded = 0
                async for chunk in resp.aiter_bytes():
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > 5 * 1024 * 1024:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST, 
                            detail={"signal": "FILE_TOO_LARGE", "error": "URL content exceeds 5MB limit"}
                        )
                    chunks.append(chunk)
                html_content = b"".join(chunks).decode("utf-8", errors="ignore")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch URL %s: %s", ingest_request.url, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": "URL_FETCH_FAILED", "error": str(e)}
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": "URL_EXTRACT_FAILED", "error": "No text content found"}
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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"signal": "URL_ALREADY_INGESTED", "asset_id": str(existing_asset["_id"])}
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
    for i, chunk in enumerate(chunks):
        file_chunks.append(
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i + 1,  # FIX: was hardcoded to 1
                chunk_project_id=project.id,
                chunk_asset_id=ObjectId(asset_id),
            )
        )

    if file_chunks:
        inserted_count = await chunk_model.insert_many_chunks(chunks=file_chunks)
    else:
        inserted_count = 0

    return URLIngestResponse(
        signal="URL_INGESTION_SUCCESS",
        url=ingest_request.url,
        inserted_chunks=inserted_count,
        asset_id=str(asset_id)
    )

