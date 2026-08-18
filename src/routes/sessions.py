"""Session management routes.

Sessions are always scoped to their project. A request using project-B's ID
can never retrieve or delete project-A's session.
"""
from fastapi import APIRouter, Request, status, Query, HTTPException, Path
from fastapi.responses import JSONResponse
from models.ProjectModel import ProjectModel
from models.SessionModel import SessionModel
from models.MessageModel import MessageModel
from .schemes.session import SessionListResponse, SessionMessagesResponse, DeleteSessionResponse, SessionItem, MessageItem
from models import ResponseSignal

import logging

logger = logging.getLogger("uvicorn.error")
from middleware.rate_limiter import limiter

_PROJECT_ID = Path(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")

sessions_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["Sessions"],
)


async def _require_project(db_client, project_id: str):
    """Fetch project or raise 404. Never auto-creates."""
    project_model = await ProjectModel.create_instance(db_client=db_client)
    project = await project_model.get_project_by_id(project_id=project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value, "project_id": project_id},
        )
    return project


#  List all sessions for a project 
@sessions_router.get("/sessions/{project_id}", summary="List all sessions for a project", response_model=SessionListResponse)
@limiter.limit("60/minute")
async def list_sessions(
    request: Request,
    project_id: str = _PROJECT_ID,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
):
    """Return metadata for all conversation sessions belonging to a project."""
    project = await _require_project(request.app.db_client, project_id)

    session_model = await SessionModel.create_instance(db_client=request.app.db_client)
    skip = (page - 1) * page_size
    sessions, total = await session_model.get_all_sessions(
        project_id=project.project_id, skip=skip, limit=page_size
    )

    return SessionListResponse(
        signal="LIST_SESSIONS_SUCCESS",
        project_id=project_id,
        page=page,
        page_size=page_size,
        total=total,
        sessions=[
            SessionItem(
                session_id=s.session_id,
                message_count=s.message_count,
                summary=s.summary,
                created_at=s.created_at.isoformat() if s.created_at else None,
                updated_at=s.updated_at.isoformat() if s.updated_at else None,
            )
            for s in sessions
        ],
    )


#  Get messages from a specific session
@sessions_router.get(
    "/sessions/{project_id}/{session_id}",
    summary="Get all messages in a session",
    response_model=SessionMessagesResponse,
)
@limiter.limit("120/minute")
async def get_session_messages(
    request: Request,
    session_id: str,
    project_id: str = _PROJECT_ID,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
):
    """Return messages in a conversation session scoped to the given project."""
    project = await _require_project(request.app.db_client, project_id)

    # Verify the session belongs to this project (prevents cross-project access)
    session_model = await SessionModel.create_instance(db_client=request.app.db_client)
    session = await session_model.get_session_scoped(
        session_id=session_id, project_id=project.project_id
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": "SESSION_NOT_FOUND", "session_id": session_id},
        )

    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    skip = (page - 1) * page_size
    messages = await message_model.get_messages_by_session(
        session_id=session_id, skip=skip, limit=page_size
    )
    total = await message_model.get_message_count(session_id=session_id)

    return SessionMessagesResponse(
        signal="GET_SESSION_SUCCESS",
        session_id=session_id,
        page=page,
        page_size=page_size,
        total=total,
        messages=[
            MessageItem(
                role=m.role,
                content=m.text,
                created_at=m.created_at.isoformat() if m.created_at else None,
            )
            for m in messages
        ],
    )


#  Delete a session and all its messages
@sessions_router.delete(
    "/sessions/{project_id}/{session_id}",
    summary="Delete a session and all its messages",
    response_model=DeleteSessionResponse,
)
@limiter.limit("60/minute")
async def delete_session(request: Request, session_id: str, project_id: str = _PROJECT_ID):
    """Permanently delete a conversation session scoped to the given project."""
    project = await _require_project(request.app.db_client, project_id)

    # Verify the session belongs to this project before deleting
    session_model = await SessionModel.create_instance(db_client=request.app.db_client)
    session = await session_model.get_session_scoped(
        session_id=session_id, project_id=project.project_id
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"signal": "SESSION_NOT_FOUND", "session_id": session_id},
        )

    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    deleted_messages = await message_model.delete_messages_by_session(session_id=session_id)
    await session_model.delete_session(session_id=session_id)

    return DeleteSessionResponse(
        signal="DELETE_SESSION_SUCCESS",
        session_id=session_id,
        deleted_messages=deleted_messages,
    )
