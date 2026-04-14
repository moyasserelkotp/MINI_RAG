from pydantic import BaseModel, Field
from typing import Optional


class PushRequest(BaseModel):
    do_reset: Optional[int] = Field(0, ge=0, le=1, description="Set to 1 to delete and rebuild the collection from scratch")


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Query text to search")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum relevance score (0 = no filter)")
    use_hybrid: Optional[bool] = Field(True, description="Use hybrid (BM25 + semantic) search")
