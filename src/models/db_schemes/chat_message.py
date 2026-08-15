from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from bson.objectid import ObjectId

class ChatMessage(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    session_id: str = Field(..., description="Unique ID for the session")
    role: str = Field(..., description="Either 'user', 'assistant', or 'system'")
    text: str = Field(..., description="Content of the message")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [("session_id", 1)],
                "name": "session_id_idx",
                "unique": False,
            }
        ]
