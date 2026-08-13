from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson.objectid import ObjectId

class ChatSession(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    session_id: str = Field(..., description="Unique ID for the session")
    project_id: str = Field(..., description="The ID of the project this session belongs to")
    summary: Optional[str] = Field(None, description="Rolling summary of the conversation")
    message_count: int = Field(0, description="Number of messages in the session so far")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }
