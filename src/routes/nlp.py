from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers.NLPController import NLPController
from models import ResponseSignal

import logging

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1/nlp"],
)


def _make_nlp_controller(request: Request) -> NLPController:
    return NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )


# ── Push / Index ─────────────────────────────────────────────────────────────

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value},
        )

    nlp_controller = _make_nlp_controller(request)

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0
    # Bug fix: do_reset should only apply to the FIRST page so that the
    # collection is deleted once (not on every batch, which would wipe data).
    is_first_page = True

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(
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
            do_reset=bool(push_request.do_reset) and is_first_page,
            chunks_ids=chunks_ids,
        )
        is_first_page = False

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value},
            )

        inserted_items_count += len(page_chunks)

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count,
        }
    )


# ── Index info ────────────────────────────────────────────────────────────────

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = _make_nlp_controller(request)
    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info,
        }
    )


# ── Delete index ─────────────────────────────────────────────────────────────

@nlp_router.delete("/index/delete/{project_id}")
async def delete_project_index(request: Request, project_id: str):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = _make_nlp_controller(request)
    deleted = nlp_controller.reset_vector_db_collection(project=project)

    if deleted is False:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_DELETE_ERROR.value},
        )

    return JSONResponse(
        content={"signal": ResponseSignal.VECTORDB_DELETE_SUCCESS.value}
    )


# ── Search ────────────────────────────────────────────────────────────────────

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = _make_nlp_controller(request)
    results = nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit,
        use_hybrid=search_request.use_hybrid,
        score_threshold=search_request.score_threshold,
    )


    # None = embed/search infrastructure failed; [] = no hits above threshold
    if results is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "total": len(results),
            "results": [
                {
                    "id": r.id,
                    "score": round(float(r.score), 6),
                    "text": r.payload.get("text", ""),
                    "metadata": r.payload.get("metadata", {}),
                }
                for r in results
            ],
        }
    )


# ── Answer (RAG) ──────────────────────────────────────────────────────────────

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = _make_nlp_controller(request)
    answer, full_prompt, chat_history = nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
        use_hybrid=search_request.use_hybrid,
        score_threshold=search_request.score_threshold,
    )

    if answer is False:
        # Search infra failed completely (e.g. Qdrant is down)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.RAG_ANSWER_ERROR.value, "error": "Search infrastructure failed"},
        )

    if answer is None and full_prompt is None:
        # Both being None means the search itself failed (embed/infra error)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.RAG_ANSWER_ERROR.value},
        )

    if not answer and full_prompt is None:
        # Search returned 0 results above threshold — not a server error
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
                "answer": "No relevant documents found for your query. Please try a different question or lower the score threshold.",
                "full_prompt": None,
                "chat_history": None,
            },
        )

    if answer is None and full_prompt is not None:
        # Search worked, found docs, BUT the LLM returned None
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.RAG_ANSWER_ERROR.value, 
                "error": "LLM generation failed."
            },
        )

    if isinstance(answer, str) and ("error:" in answer.lower() or "not initialized" in answer or "was not set" in answer):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.RAG_ANSWER_ERROR.value, 
                "error": answer
            },
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history,
        }
    )
