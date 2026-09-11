# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
# pyrefly: ignore [missing-import]
from bson.objectid import ObjectId
from datetime import datetime, timezone


class AgentRun(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    run_id: str = Field(..., description="Unique UUID for this agent run")
    project_id: ObjectId
    session_id: Optional[str] = None
    status: str = Field(..., description="running, success, failed, max_steps")
    query: str = Field(..., description="Truncated user query (max 200 chars for privacy)")
    final_answer: Optional[str] = None
    steps_count: int = 0
    tool_calls_count: int = 0
    retrieval_attempts: int = 0
    tools_used: List[str] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    error: Optional[str] = None

    @field_validator("query", mode="before")
    @classmethod
    def truncate_query(cls, v: str) -> str:
        """Truncate stored query to 200 chars to limit PII exposure in traces."""
        if isinstance(v, str) and len(v) > 200:
            return v[:200] + "…"
        return v

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [("project_id", 1)],
                "name": "agent_run_project_id_index_1",
                "unique": False,
            },
            {
                "key": [("run_id", 1)],
                "name": "agent_run_run_id_index_1",
                "unique": True,
            },
            {
                "key": [("session_id", 1)],
                "name": "agent_run_session_id_index_1",
                "unique": False,
            },
        ]
