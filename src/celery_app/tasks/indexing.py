from __future__ import annotations

import asyncio
import logging
from typing import Optional

from celery import Task
from motor.motor_asyncio import AsyncIOMotorClient

from celery_app.celery_config import celery_app
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from controllers.NLPController import NLPController
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel

logger = logging.getLogger(__name__)

_clients = {}

def get_shared_clients():
    if not _clients:
        settings = get_settings()
        llm_factory = LLMProviderFactory(settings)
        vectordb_factory = VectorDBProviderFactory(settings)

        generation_client = llm_factory.create(provider=settings.GENERATION_BACKEND)
        generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

        embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
        embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE,
        )

        vectordb_client = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
        vectordb_client.connect()

        _clients["generation"] = generation_client
        _clients["embedding"] = embedding_client
        _clients["vectordb"] = vectordb_client
    return _clients["generation"], _clients["embedding"], _clients["vectordb"]

#  Helper: run async from sync context 
def _run_async(coro):
    """Execute an async coroutine inside a Celery (sync) task.

    Celery workers are synchronous, so asyncio.run() is always safe here.
    get_event_loop() is deprecated in Python 3.10+ and raises RuntimeError
    in 3.12+, so we avoid it entirely.
    """
    return asyncio.run(coro)


#  Base task 
class IndexingBaseTask(Task):
    abstract = True
    # max_retries / retry_backoff are set on the task decorator below
    # (single source of truth — avoids config drift)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Indexing task %s FAILED permanently after retries | error=%s",
                     task_id, exc, exc_info=einfo)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning("Indexing task %s RETRYING (attempt %d/%d) | error=%s",
                       task_id, self.request.retries + 1, self.max_retries, exc)


#  Index-push task 
@celery_app.task(
    bind=True,
    base=IndexingBaseTask,
    name="celery_app.tasks.indexing.index_project_into_vectordb",
    queue="indexing",
    track_started=True,
    trail=True,
    # Retry config (single source of truth)
    max_retries=3,
    default_retry_delay=15,      # seconds before first retry
    retry_backoff=True,          # exponential: 15s, 30s, 60s
    retry_backoff_max=180,       # cap at 3 minutes
    retry_jitter=True,           # avoid thundering herd
)
def index_project_into_vectordb(
    self,
    project_id: str,
    do_reset: int = 0,
) -> dict:
    """
    Embed all stored chunks for *project_id* and push them to the vector DB.

    Parameters
    ----------
    project_id : RAG project identifier
    do_reset   : 1 = delete the vector collection before re-indexing

    Returns
    -------
    dict with keys: signal, inserted_items_count
    """
    task_id = self.request.id
    logger.info(
        "Task %s | START index_project_into_vectordb | project=%s do_reset=%s",
        task_id, project_id, do_reset,
    )

    settings = get_settings()

    generation_client, embedding_client, vectordb_client = get_shared_clients()

    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    async def _async_index():
        db_client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = db_client[settings.MONGODB_DATABASE]

        try:
            project_model = await ProjectModel.create_instance(db_client=db)
            chunk_model   = await ChunkModel.create_instance(db_client=db)
            project = await project_model.get_project_or_create_one(project_id=project_id)

            if not project:
                return {"signal": "PROJECT_NOT_FOUND_ERROR"}

            nlp_controller = NLPController(
                db_client=db,
                vectordb_client=vectordb_client,
                generation_client=generation_client,
                embedding_client=embedding_client,
                template_parser=template_parser,
                cohere_client=None,
            )

            has_records  = True
            page_no      = 1
            inserted_total = 0
            idx          = 0

            if do_reset:
                await asyncio.to_thread(nlp_controller.reset_vector_db_collection, project)

            while has_records:
                page_chunks = await chunk_model.get_project_chunks(
                    project_id=project.id, page_no=page_no
                )

                if not page_chunks:
                    has_records = False
                    break

                page_no += 1
                chunks_ids = list(range(idx, idx + len(page_chunks)))
                idx += len(page_chunks)

                is_inserted = await asyncio.to_thread(
                    nlp_controller.index_into_vector_db,
                    project=project,
                    chunks=page_chunks,
                    do_reset=False,
                    chunks_ids=chunks_ids,
                )

                if not is_inserted:
                    # Qdrant write failed — raise so Celery retries the task
                    raise RuntimeError(
                        f"Vector DB insert failed on page {page_no - 1} "
                        f"for project '{project_id}'"
                    )

                inserted_total += len(page_chunks)
                logger.info(
                    "Task %s | Indexed page %d (%d chunks) | total so far: %d",
                    task_id, page_no - 1, len(page_chunks), inserted_total,
                )

            return {
                "signal": "INSERT_INTO_VECTORDB_SUCCESS",
                "inserted_items_count": inserted_total,
            }

        finally:
            db_client.close()

    try:
        result = _run_async(_async_index())
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
