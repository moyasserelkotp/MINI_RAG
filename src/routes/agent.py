# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from models import ResponseSignal
from models.db_schemes import Project
from models.ProjectModel import ProjectModel
from controllers import AgentController
# pyrefly: ignore [missing-import]
from motor.motor_asyncio import AsyncIOMotorClient
import os
import certifi

# Normally dependencies are injected. We define router logic here.
agent_router = APIRouter(
    prefix="/api/v1/agent",
    tags=["Agentic RAG"]
)

class AgentQueryRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None

class AgentQueryResponse(BaseModel):
    signal: ResponseSignal
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    run_id: Optional[str] = None
    steps: Optional[int] = None
    error: Optional[str] = None

# pyrefly: ignore [missing-import]
from fastapi import Request

def _make_agent_controller(request: Request) -> AgentController:
    return AgentController(
        db_client=request.app.db_client,
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        cohere_client=getattr(request.app, "cohere_client", None),
        initialized_collections=request.app.initialized_collections,
    )

@agent_router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
    request: Request,
    payload: AgentQueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Query the Agentic RAG system for a specific project.
    """
    # 1. Get controllers
    agent_controller = _make_agent_controller(request)
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=payload.project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    # 3. Execute
    try:
        result = await agent_controller.execute_agent(
            project=project,
            query=payload.query,
            session_id=payload.session_id
        )
        
        if "error" in result:
            return AgentQueryResponse(
                signal=ResponseSignal.AGENT_CHAT_ERROR,
                error=result["error"]
            )
            
        return AgentQueryResponse(
            signal=ResponseSignal.AGENT_CHAT_SUCCESS,
            answer=result.get("answer"),
            sources=result.get("sources"),
            run_id=result.get("run_id"),
            steps=result.get("steps")
        )
        
    except Exception as e:
        return AgentQueryResponse(
            signal=ResponseSignal.AGENT_CHAT_ERROR,
            error=str(e)
        )
