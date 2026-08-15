from pydantic import BaseModel
from typing import List, Optional
from .system import BaseResponse

class SessionItem(BaseModel):
    session_id: str
    message_count: int
    summary: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class SessionListResponse(BaseResponse):
    project_id: str
    page: int
    page_size: int
    total: int
    sessions: List[SessionItem]

class MessageItem(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None

class SessionMessagesResponse(BaseResponse):
    session_id: str
    page: int
    page_size: int
    total: int
    messages: List[MessageItem]

class DeleteSessionResponse(BaseResponse):
    session_id: str
    deleted_messages: int
