from __future__ import annotations

import logging
import time
from typing import Optional

from celery import Task
from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from celery_app.celery_config import celery_app
from helpers.config import get_settings
from controllers.DataController import DataController
from controllers.ProcessController import ProcessController
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.db_schemes import DataChunk
from models.enums.AssetTypeEnum import AssetTypeEnum
from utils.metrics import record_chunking_latency

import asyncio

logger = logging.getLogger(__name__)


#  Helper: run async code from sync Celery task 

def _run_async(coro):
    """Execute an async coroutine inside a Celery (sync) task.

    Celery workers are synchronous, so asyncio.run() is always safe here.
    get_event_loop() is deprecated in Python 3.10+ and raises RuntimeError
    in 3.12+ when no event loop exists, so we avoid it entirely.
    """
    return asyncio.run(coro)


#  Base task class with retry behaviour 

class ProcessingBaseTask(Task):
    abstract = True
    # max_retries / retry_backoff are set on the task decorator below
    # (single source of truth — avoids config drift)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "Processing task %s FAILED permanently after retries | error=%s",
            task_id, exc, exc_info=einfo,
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(
            "Processing task %s RETRYING (attempt %d/%d) | error=%s",
            task_id,
            self.request.retries + 1,
            self.max_retries,
            exc,
        )


#  Main processing task 

@celery_app.task(
    bind=True,
    base=ProcessingBaseTask,
    name="celery_app.tasks.processing.process_project_files",
    queue="processing",
    track_started=True,
    trail=True,
    # Retry config (single source of truth)
    max_retries=3,
    default_retry_delay=10,      # seconds before first retry
    retry_backoff=True,          # exponential: 10s, 20s, 40s
    retry_backoff_max=120,       # cap at 2 minutes
    retry_jitter=True,           # add randomness to avoid thundering herd
)
def process_project_files(
    self,
    project_id: str,
    file_id: Optional[str],
    chunk_size: int,
    overlap_size: int,
    do_reset: int,
) -> dict:
    """
    Chunk documents and store them in MongoDB.

    Parameters
    ----------
    project_id  : RAG project identifier
    file_id     : Process only this file (None → all files)
    chunk_size  : Characters per chunk
    overlap_size: Overlap between chunks
    do_reset    : 1 = delete existing chunks before inserting

    Returns
    -------
    dict with keys: signal, inserted_chunks, processed_files, failed_files
    """
    task_id = self.request.id
    logger.info(
        "Task %s | START process_project_files | project=%s file=%s",
        task_id, project_id, file_id,
    )

    settings = get_settings()

    async def _async_process():
        db_client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = db_client[settings.MONGODB_DATABASE]

        try:
            project_model = await ProjectModel.create_instance(db_client=db)
            project = await project_model.get_project_or_create_one(project_id=project_id)

            asset_model = await AssetModel.create_instance(db_client=db)
            project_files_ids: dict = {}

            if file_id:
                asset_record = await asset_model.get_asset_record(
                    asset_project_id=project.id,
                    asset_name=file_id,
                    asset_type=AssetTypeEnum.FILE.value,
                )
                if asset_record is None:
                    return {
                        "signal": "FILE_ID_ERROR",
                        "error": f"File '{file_id}' not found in project '{project_id}'",
                    }
                project_files_ids = {asset_record.id: asset_record.asset_name}
            else:
                project_files = await asset_model.get_all_project_assets(
                    asset_project_id=project.id,
                    asset_type=AssetTypeEnum.FILE.value,
                )
                project_files_ids = {r.id: r.asset_name for r in project_files}

            if not project_files_ids:
                return {"signal": "NO_FILES_ERROR", "error": "No files found for project"}

            process_controller = ProcessController(project_id=project_id)
            chunk_model = await ChunkModel.create_instance(db_client=db)

            chunk_strategy = getattr(settings, "CHUNK_STRATEGY", "recursive")

            if do_reset == 1:
                await chunk_model.delete_chunks_by_project_id(project_id=project.id)
                logger.info("Task %s | Reset chunks for project %s", task_id, project_id)

            no_records = 0
            no_files = 0
            failed_files = []

            for asset_id, asset_file_id in project_files_ids.items():
                logger.info("Task %s | Processing file: %s", task_id, asset_file_id)
                file_content = process_controller.get_file_content(file_id=asset_file_id)

                if file_content is None:
                    logger.error("Task %s | Cannot load file: %s", task_id, asset_file_id)
                    failed_files.append(asset_file_id)
                    continue

                _chunk_start = time.monotonic()
                file_chunks = process_controller.process_file_content(
                    file_content=file_content,
                    file_id=asset_file_id,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                    chunk_strategy=chunk_strategy,
                )

                try:
                    record_chunking_latency(
                        strategy=chunk_strategy,
                        duration=time.monotonic() - _chunk_start,
                    )
                except Exception:
                    pass

                if not file_chunks:
                    logger.error("Task %s | No chunks for file: %s", task_id, asset_file_id)
                    failed_files.append(asset_file_id)
                    continue

                file_chunks_records = [
                    DataChunk(
                        chunk_text=chunk.page_content,
                        chunk_metadata=chunk.metadata,
                        chunk_order=i + 1,
                        chunk_project_id=project.id,
                        chunk_asset_id=(
                            ObjectId(asset_id) if isinstance(asset_id, str) else asset_id
                        ),
                    )
                    for i, chunk in enumerate(file_chunks)
                ]

                inserted = await chunk_model.insert_many_chunks(chunks=file_chunks_records)
                no_records += inserted
                no_files += 1
                logger.info(
                    "Task %s | Inserted %d chunks for file %s",
                    task_id, inserted, asset_file_id,
                )

            if no_files == 0:
                # All files failed to load/chunk — this is retryable
                # (e.g. storage temporarily unavailable)
                raise RuntimeError(
                    f"All files failed to process. failed={failed_files}"
                )

            return {
                "signal": "PROCESSING_SUCCESS",
                "inserted_chunks": no_records,
                "processed_files": no_files,
                "failed_files": failed_files,
            }

        finally:
            db_client.close()

    try:
        result = _run_async(_async_process())
        logger.info("Task %s | DONE | result=%s", task_id, result.get("signal"))
        return result
    except self.MaxRetriesExceededError:
        # All retries exhausted — task goes to dead_letters queue
        logger.error("Task %s | MAX RETRIES EXCEEDED — moving to dead_letters", task_id)
        raise
    except Exception as exc:
        logger.exception("Task %s | UNHANDLED ERROR (retry %d/%d): %s",
                         task_id, self.request.retries, self.max_retries, exc)
        raise self.retry(exc=exc)
