from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from .system import BaseResponse

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
    filter_source: Optional[str] = Field(
        None, max_length=256,
        description="Only search chunks from this filename (e.g. 'policy_2024.pdf')"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Key-value pairs to filter chunks by metadata (e.g. {'department': 'HR'})"
    )

class InfoIndexResponse(BaseResponse):
    collection_info: Dict[str, Any]

class SearchResultItem(BaseModel):
    id: Union[str, int]
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseResponse):
    total: int
    results: List[SearchResultItem]

class AnswerResponse(BaseResponse):
    answer: str
    sources: List[Dict[str, Any]]
    cached: bool
    session_id: Optional[str] = None
    # full_prompt and chat_history intentionally omitted — internal details
    # must not be exposed on the public API surface.

class EvaluationResponse(BaseResponse):
    metric: str
    score: float
    reasoning: str
    parse_error: bool = False
    error: Optional[str] = None
