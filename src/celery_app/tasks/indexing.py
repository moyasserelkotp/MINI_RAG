"""
celery_app/tasks/indexing.py
============================
Celery tasks for embedding chunks and pushing them into the vector database.

Queue: ``indexing``
"""

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


# ── Helper: run async from sync context ─────────────────────────────────────

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Base task ────────────────────────────────────────────────────────────────

class IndexingBaseTask(Task):
    abstract = True
    max_retries = 3
    default_retry_delay = 15

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Indexing task %s FAILED | error=%s", task_id, exc, exc_info=einfo)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning("Indexing task %s RETRYING | error=%s", task_id, exc)


# ── Index-push task ──────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    base=IndexingBaseTask,
    name="celery_app.tasks.indexing.index_project_into_vectordb",
    queue="indexing",
    track_started=True,
    trail=True,
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

    # Build stateless clients (no FastAPI app context available in worker)
    llm_factory      = LLMProviderFactory(settings)
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
            is_first_page = True

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

                is_inserted = nlp_controller.index_into_vector_db(
                    project=project,
                    chunks=page_chunks,
                    do_reset=bool(do_reset) and is_first_page,
                    chunks_ids=chunks_ids,
                )
                is_first_page = False

                if not is_inserted:
                    return {"signal": "INSERT_INTO_VECTORDB_ERROR"}

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
                vectordb_client.disconnect()
            except Exception:
                pass

    try:
        result = _run_async(_async_index())
        logger.info("Task %s | DONE | result=%s", task_id, result.get("signal"))
        return result
    except Exception as exc:
        logger.exception("Task %s | UNHANDLED ERROR: %s", task_id, exc)
        raise self.retry(exc=exc, countdown=15)
