from pydantic import BaseModel, Field
from typing import Optional


class PushRequest(BaseModel):
    do_reset: Optional[int] = Field(0, ge=0, le=1, description="Set to 1 to delete and rebuild the collection from scratch")


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Query text to search")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Minimum relevance score between 0.0 and 1.0 (cosine similarity)"
    )
    use_hybrid: Optional[bool] = Field(True, description="Use hybrid (BM25 + semantic) search")
    session_id: Optional[str] = Field(
        None, max_length=128,
        description="Optional Session ID for chat memory (max 128 chars)"
    )
