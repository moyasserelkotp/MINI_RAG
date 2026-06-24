from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from celery.result import AsyncResult
from celery_app.celery_config import celery_app
from celery_app.tasks.processing import process_project_files
from celery_app.tasks.indexing import index_project_into_vectordb

logger = logging.getLogger(__name__)

tasks_router = APIRouter(
    prefix="/api/v1/tasks",
    tags=["api_v1/tasks"],
)


#  Request schemas 
class ProcessTaskRequest(BaseModel):
    file_id: Optional[str] = Field(
        None,
        description="Asset file ID. Required for process-file endpoint; "
                    "omit for process-all.",
    )
    chunk_size: int = Field(512, ge=64, le=8192, description="Tokens per chunk")
    overlap_size: int = Field(50, ge=0, le=512, description="Overlap between chunks")
    do_reset: int = Field(
        0, ge=0, le=1,
        description="1 = delete existing chunks before inserting new ones",
    )


class IndexTaskRequest(BaseModel):
    do_reset: int = Field(
        0, ge=0, le=1,
        description="1 = wipe the vector collection before re-indexing",
    )


#  Helpers 
def _task_response(task_id: str, http_status: int = status.HTTP_202_ACCEPTED) -> JSONResponse:
    """Return a uniform 202 payload pointing the client at the status endpoint."""
    return JSONResponse(
        status_code=http_status,
        content={
            "signal": "TASK_SUBMITTED",
            "task_id": task_id,
            "status_url": f"/api/v1/tasks/{task_id}",
        },
    )


def _build_state(result: AsyncResult) -> dict:
    """Translate a Celery AsyncResult into a JSON-serialisable dict."""
    state = result.state  # PENDING | STARTED | RETRY | FAILURE | SUCCESS

    payload: dict = {
        "task_id": result.id,
        "state": state,
    }

    if state == "PENDING":
        payload["info"] = "Task is waiting in the queue or does not exist."

    elif state == "STARTED":
        payload["info"] = result.info or "Task is running."

    elif state == "RETRY":
        exc = result.info
        payload["info"] = str(exc) if exc else "Task is being retried."

    elif state == "FAILURE":
        exc = result.info
        payload["error"] = str(exc) if exc else "Unknown error"

    elif state == "SUCCESS":
        payload["result"] = result.result

    else:
        payload["info"] = str(result.info)

    return payload


#  Submit: process single file 
@tasks_router.post(
    "/process-file/{project_id}",
    summary="Submit async document chunking (single file)",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_process_file(
    request: Request,
    project_id: str,
    body: ProcessTaskRequest,
):
    """
    Enqueue a background job to chunk **one** uploaded file.

    Returns a ``task_id`` which can be polled at
    ``GET /api/v1/tasks/{task_id}``.
    """
    if not body.file_id:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "'file_id' is required for process-file endpoint"},
        )

    task = process_project_files.apply_async(
        kwargs={
            "project_id": project_id,
            "file_id": body.file_id,
            "chunk_size": body.chunk_size,
            "overlap_size": body.overlap_size,
            "do_reset": body.do_reset,
        },
        queue="processing",
    )
    logger.info(
        "Submitted processing task %s | project=%s file=%s",
        task.id, project_id, body.file_id,
    )
    return _task_response(task.id)


#  Submit: process all files 
@tasks_router.post(
    "/process-all/{project_id}",
    summary="Submit async document chunking (all files)",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_process_all(
    request: Request,
    project_id: str,
    body: ProcessTaskRequest,
):
    """
    Enqueue a background job to chunk **all** uploaded files for a project.
    """
    task = process_project_files.apply_async(
        kwargs={
            "project_id": project_id,
            "file_id": None,
            "chunk_size": body.chunk_size,
            "overlap_size": body.overlap_size,
            "do_reset": body.do_reset,
        },
        queue="processing",
    )
    logger.info(
        "Submitted process-all task %s | project=%s",
        task.id, project_id,
    )
    return _task_response(task.id)


#  Submit: index / push to vector DB 
@tasks_router.post(
    "/index/{project_id}",
    summary="Submit async vector-DB indexing",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_index(
    request: Request,
    project_id: str,
    body: IndexTaskRequest,
):
    """
    Enqueue a background job to embed stored chunks and push them to the
    vector database.
    """
    task = index_project_into_vectordb.apply_async(
        kwargs={
            "project_id": project_id,
            "do_reset": body.do_reset,
        },
        queue="indexing",
    )
    logger.info(
        "Submitted indexing task %s | project=%s",
        task.id, project_id,
    )
    return _task_response(task.id)


#  Poll task status
@tasks_router.get(
    "/{task_id}",
    summary="Get task status and result",
)
async def get_task_status(task_id: str):
    """
    Poll the status of any background task.

    Possible ``state`` values:

    | State   | Meaning                                         |
    |---------|-------------------------------------------------|
    | PENDING | Waiting in queue / task ID unknown              |
    | STARTED | Worker picked it up, currently executing        |
    | RETRY   | Task failed and is being retried                |
    | FAILURE | Terminal failure (see ``error`` field)          |
    | SUCCESS | Completed (see ``result`` field)                |
    """
    result = AsyncResult(task_id, app=celery_app)
    return JSONResponse(content=_build_state(result))


#  Revoke / cancel task 

@tasks_router.delete(
    "/{task_id}",
    summary="Revoke / cancel a pending or running task",
)
async def revoke_task(task_id: str, terminate: bool = False):
    """
    Cancel a queued task.

    - ``terminate=false`` (default): tell the broker to discard the task
        if it has not started yet.
    - ``terminate=true``: send SIGTERM to the worker process handling the task
        (use with care — may leave DB in an inconsistent state).
    """
    celery_app.control.revoke(task_id, terminate=terminate, signal="SIGTERM")
    logger.info("Revoked task %s (terminate=%s)", task_id, terminate)
    return JSONResponse(
        content={
            "signal": "TASK_REVOKED",
            "task_id": task_id,
            "terminate": terminate,
        }
    )


"""
routes/tasks.py
===============
REST API for submitting background Celery jobs and polling their status.

Endpoints
---------
  POST /api/v1/tasks/process-file/{project_id}   – async document chunking (single file)
  POST /api/v1/tasks/process-all/{project_id}    – async document chunking (all files)
  POST /api/v1/tasks/index/{project_id}          – async vector-DB indexing
  GET  /api/v1/tasks/{task_id}                   – poll task status + result
  DELETE /api/v1/tasks/{task_id}                 – revoke / cancel a pending task
"""
