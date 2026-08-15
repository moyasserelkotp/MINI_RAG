from fastapi import APIRouter, status, Request, HTTPException, Path
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from .schemes.nlp import (
    SearchRequest, InfoIndexResponse,
    SearchResponse, AnswerResponse, SearchResultItem
)
from .schemes.system import BaseResponse
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers.NLPController import NLPController
from models import ResponseSignal
import logging
import json

logger = logging.getLogger("uvicorn.error")
from middleware.rate_limiter import limiter

_PROJECT_ID = Path(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["Search & NLP"],
)


def _make_nlp_controller(request: Request) -> NLPController:
    return NLPController(
        db_client=request.app.db_client,
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        cohere_client=getattr(request.app, "cohere_client", None),
    )


def _project_not_found(project_id: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value, "project_id": project_id},
    )




# ── Index info ────────────────────────────────────────────────────────────────
@nlp_router.get("/index/info/{project_id}", response_model=InfoIndexResponse)
@limiter.limit("60/minute")
async def get_project_index_info(request: Request, project_id: str = _PROJECT_ID):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=project_id)
    if not project:
        _project_not_found(project_id)

    nlp_controller = _make_nlp_controller(request)
    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return InfoIndexResponse(
        signal=ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
        collection_info=collection_info,
    )


# ── Delete index ──────────────────────────────────────────────────────────────
# Canonical new route: DELETE /api/v1/nlp/documents/{project_id}
@nlp_router.delete("/documents/{project_id}", response_model=BaseResponse, summary="Delete project vector index")
@limiter.limit("5/minute")
async def delete_project_index(request: Request, project_id: str = _PROJECT_ID):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=project_id)
    if not project:
        _project_not_found(project_id)

    nlp_controller = _make_nlp_controller(request)
    deleted = nlp_controller.reset_vector_db_collection(project=project)

    if deleted is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": ResponseSignal.VECTORDB_DELETE_ERROR.value},
        )

    return BaseResponse(signal=ResponseSignal.VECTORDB_DELETE_SUCCESS.value)


# Deprecated aliases
@nlp_router.delete(
    "/index/{project_id}",
    response_model=BaseResponse,
    deprecated=True,
    summary="[DEPRECATED] Use DELETE /documents/{project_id}",
    include_in_schema=False,
)
@nlp_router.delete(
    "/index/delete/{project_id}",
    response_model=BaseResponse,
    deprecated=True,
    summary="[DEPRECATED] Use DELETE /documents/{project_id}",
    include_in_schema=False,
)
@limiter.limit("5/minute")
async def delete_project_index_deprecated(request: Request, project_id: str = _PROJECT_ID):
    return await delete_project_index(request, project_id)


# ── Search ────────────────────────────────────────────────────────────────────
@nlp_router.post("/retrieve/{project_id}", response_model=SearchResponse)
@limiter.limit("60/minute")
async def retrieve_documents(request: Request, search_request: SearchRequest, project_id: str = _PROJECT_ID):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=project_id)

    if not project:
        _project_not_found(project_id)

    nlp_controller = _make_nlp_controller(request)

    metadata_filter = {}
    if search_request.filter_source:
        metadata_filter["source"] = search_request.filter_source
    if search_request.filter_metadata:
        metadata_filter.update(search_request.filter_metadata)

    import asyncio
    results = await asyncio.to_thread(
        nlp_controller.search_vector_db_collection,
        project=project,
        text=search_request.text,
        limit=search_request.limit,
        use_hybrid=search_request.use_hybrid,
        score_threshold=search_request.score_threshold,
        metadata_filter=metadata_filter or None,
    )

    if not results:
        return SearchResponse(
            signal=ResponseSignal.VECTORDB_SEARCH_ERROR.value,
            total=0,
            results=[],
        )

    mapped_results = []
    for r in results:
        mapped_results.append(
            SearchResultItem(
                id=str(getattr(r, "id", "")),
                score=float(getattr(r, "score", 0.0)),
                text=r.payload.get("text", "") if hasattr(r, "payload") else "",
                metadata=r.payload.get("metadata", {}) if hasattr(r, "payload") else {}
            )
        )

    return SearchResponse(
        signal=ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
        total=len(mapped_results),
        results=mapped_results,
    )

@nlp_router.post("/index/search/{project_id}", response_model=SearchResponse, deprecated=True, include_in_schema=False)
@limiter.limit("60/minute")
async def search_index_deprecated(request: Request, search_request: SearchRequest, project_id: str = _PROJECT_ID):
    return await retrieve_documents(request, search_request, project_id)

# ── Answer (RAG) — canonical route ───────────────────────────────────────────
@nlp_router.post("/answer/{project_id}", response_model=AnswerResponse, summary="RAG answer")
@limiter.limit("60/minute")
async def answer_rag(request: Request, search_request: SearchRequest, project_id: str = _PROJECT_ID):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=project_id)
    if not project:
        _project_not_found(project_id)

    nlp_controller = _make_nlp_controller(request)

    metadata_filter = {}
    if search_request.filter_source:
        metadata_filter["source"] = search_request.filter_source
    if search_request.filter_metadata:
        metadata_filter.update(search_request.filter_metadata)

    answer, full_prompt, chat_history, sources, cached = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
        use_hybrid=search_request.use_hybrid,
        score_threshold=search_request.score_threshold,
        session_id=search_request.session_id,
        metadata_filter=metadata_filter or None,
    )

    if answer is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": ResponseSignal.RAG_ANSWER_ERROR.value, "error": "Search infrastructure failed"},
        )

    if answer is None and full_prompt is None and chat_history is None:
        return AnswerResponse(
            signal=ResponseSignal.RAG_ANSWER_SUCCESS.value,
            answer="No relevant documents found for your query. Please try a different question or lower the score threshold.",
            sources=[],
            cached=False,
            session_id=search_request.session_id,
        )

    if answer is None and full_prompt is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": ResponseSignal.RAG_ANSWER_ERROR.value, "error": "LLM generation failed."},
        )

    if isinstance(answer, str) and (
        "error:" in answer.lower() or "not initialized" in answer or "was not set" in answer
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"signal": ResponseSignal.RAG_ANSWER_ERROR.value, "error": answer},
        )

    return AnswerResponse(
        signal=ResponseSignal.RAG_ANSWER_SUCCESS.value,
        answer=answer,
        sources=sources,
        cached=cached,
        session_id=search_request.session_id,
    )


# Deprecated alias — kept for backward compatibility.
@nlp_router.post(
    "/index/answer/{project_id}",
    response_model=AnswerResponse,
    summary="[DEPRECATED] RAG answer — use POST /answer/{project_id}",
    include_in_schema=False,
)
@limiter.limit("60/minute")
async def answer_rag_deprecated(request: Request, search_request: SearchRequest, project_id: str = _PROJECT_ID):
    """Deprecated: kept for backward compatibility. Use POST /answer/{project_id}."""
    return await answer_rag(request, search_request, project_id)


# ── Streaming Answer — canonical route ───────────────────────────────────────
@nlp_router.post(
    "/answer/stream/{project_id}",
    summary="Stream RAG answer via SSE",
    response_class=StreamingResponse,
)
@limiter.limit("60/minute")
async def answer_rag_stream(
    request: Request, search_request: SearchRequest, project_id: str = _PROJECT_ID
):
    """Stream the RAG answer token-by-token using Server-Sent Events (SSE).

    Each event is:  data: {"token": "<text>"}\\n\\n
    Final event is: data: {"done": true, "sources": [...], "cached": false}\\n\\n
    """
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=project_id)
    if not project:
        _project_not_found(project_id)

    nlp_controller = _make_nlp_controller(request)

    async def event_stream():
        full_answer = []
        try:
            gen_client = request.app.generation_client
            if not hasattr(gen_client, "stream_text"):
                # Fallback: run full answer and emit in one shot
                answer, _, _, sources, cached = await nlp_controller.answer_rag_question(
                    project=project,
                    query=search_request.text,
                    limit=search_request.limit,
                    use_hybrid=search_request.use_hybrid,
                    score_threshold=search_request.score_threshold,
                    session_id=search_request.session_id,
                )
                if answer:
                    yield f"data: {json.dumps({'token': answer})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'cached': cached})}\n\n"
                return

            async for token in nlp_controller.answer_rag_stream(
                project=project,
                query=search_request.text,
                limit=search_request.limit,
                use_hybrid=search_request.use_hybrid,
                score_threshold=search_request.score_threshold,
                session_id=search_request.session_id,
            ):
                if isinstance(token, dict) and "error" in token:
                    yield f"data: {json.dumps({'error': token['error']})}\n\n"
                    return
                full_answer.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            logger.error("Streaming RAG error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Deprecated streaming alias
@nlp_router.post(
    "/index/answer/stream/{project_id}",
    summary="[DEPRECATED] Stream RAG answer — use POST /answer/stream/{project_id}",
    response_class=StreamingResponse,
    include_in_schema=False,
)
@limiter.limit("60/minute")
async def answer_rag_stream_deprecated(
    request: Request, search_request: SearchRequest, project_id: str = _PROJECT_ID
):
    """Deprecated: kept for backward compatibility. Use POST /answer/stream/{project_id}."""
    return await answer_rag_stream(request, search_request, project_id)
