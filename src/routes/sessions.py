"""Session management routes — Phase 6.

Endpoints to list, inspect, and delete conversation sessions.
"""
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from models.ProjectModel import ProjectModel
from models.SessionModel import SessionModel
from models.MessageModel import MessageModel

import logging

logger = logging.getLogger("uvicorn.error")

sessions_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1/sessions"],
)


# ── List all sessions for a project ──────────────────────────────────────────

@sessions_router.get("/sessions/{project_id}", summary="List all sessions for a project")
async def list_sessions(request: Request, project_id: str):
    """Return metadata for all conversation sessions belonging to a project."""
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "PROJECT_NOT_FOUND", "project_id": project_id},
        )

    session_model = await SessionModel.create_instance(db_client=request.app.db_client)
    sessions = await session_model.get_all_sessions(project_id=project.id)

    return JSONResponse(content={
        "signal": "LIST_SESSIONS_SUCCESS",
        "project_id": project_id,
        "total": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "message_count": getattr(s, "message_count", 0),
                "summary": getattr(s, "summary", None),
                "created_at": s.created_at.isoformat() if getattr(s, "created_at", None) else None,
                "updated_at": s.updated_at.isoformat() if getattr(s, "updated_at", None) else None,
            }
            for s in sessions
        ],
    })


# ── Get messages from a specific session ─────────────────────────────────────

@sessions_router.get(
    "/sessions/{project_id}/{session_id}",
    summary="Get all messages in a session"
)
async def get_session_messages(request: Request, project_id: str, session_id: str):
    """Return all messages in a conversation session (up to 500)."""
    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    messages = await message_model.get_messages_by_session(session_id=session_id, limit=500)

    return JSONResponse(content={
        "signal": "GET_SESSION_SUCCESS",
        "session_id": session_id,
        "total": len(messages),
        "messages": [
            {
                "role": getattr(m, "role", "unknown"),
                "content": getattr(m, "text", ""),
                "created_at": m.created_at.isoformat() if getattr(m, "created_at", None) else None,
            }
            for m in messages
        ],
    })


# ── Delete a session and all its messages ─────────────────────────────────────

@sessions_router.delete(
    "/sessions/{project_id}/{session_id}",
    summary="Delete a session and all its messages"
)
async def delete_session(request: Request, project_id: str, session_id: str):
    """Permanently delete a conversation session and every message it contains."""
    message_model = await MessageModel.create_instance(db_client=request.app.db_client)
    session_model = await SessionModel.create_instance(db_client=request.app.db_client)

    deleted_messages = await message_model.delete_messages_by_session(session_id=session_id)
    deleted_session = await session_model.delete_session(session_id=session_id)

    if not deleted_session and deleted_messages == 0:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": "SESSION_NOT_FOUND", "session_id": session_id},
        )

    return JSONResponse(content={
        "signal": "DELETE_SESSION_SUCCESS",
        "session_id": session_id,
        "deleted_messages": deleted_messages,
    })
