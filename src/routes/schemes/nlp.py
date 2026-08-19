from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from .system import BaseResponse

class SearchRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=4096, 
        description="""The primary query text to search for or answer.
- **Role**: This is the input provided by the user. It is converted into a vector embedding for semantic matching.
- **Constraints**: Must be between 1 and 4096 characters.""",
        examples=["What is AI?"]
    )
    limit: Optional[int] = Field(
        5, 
        ge=1, 
        le=50, 
        description="""The maximum number of context chunks to retrieve from the database.
- **Default**: 5
- **Range**: 1 to 50
- **Impact**: Higher limits provide more context to the LLM but consume more tokens and increase latency.""",
        examples=[5]
    )
    score_threshold: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="""Minimum similarity score for a retrieved chunk to be considered relevant.
- **Type**: Float between 0.0 and 1.0 (e.g., `0.5`).
- **Behavior**: Any document scoring below this threshold is discarded. If no documents pass, the system responds that it cannot answer.
- **Recommended**: 0.5 to 0.7 for cosine similarity.
- **Note**: Ensure you use a leading zero (e.g., `0.5`, not `.5`).""",
        examples=[0.5]
    )
    use_hybrid: Optional[bool] = Field(
        True, 
        description="""Toggle the search mechanism used to retrieve documents.
- **`true` (Default)**: Uses Hybrid Search (combines BM25 keyword search with Dense Semantic Vector search via Reciprocal Rank Fusion). Best for general RAG as it captures both exact terms and contextual meaning.
- **`false`**: Uses pure Dense Vector Search. Best for abstract, purely conceptual queries where exact keyword matching is unnecessary.""",
        examples=[True]
    )
    use_cache: Optional[bool] = Field(
        None,
        description="""Toggle semantic caching for this specific request. 
- **`true`**: Force cache usage.
- **`false`**: Force cache bypass (useful for evaluation/testing).
- **`null` (Default)**: Uses the global app setting.""",
        examples=[False]
    )
    session_id: Optional[str] = Field(
        None, 
        max_length=128,
        description="""Optional identifier to maintain conversational memory across multiple requests.
- **Role**: Links the current query to previous Q&A pairs.
- **Behavior**: If provided, the system retrieves past conversation turns to give the LLM context. If omitted, the request is treated as a standalone, stateless query.
- **Format**: Max 128 characters (e.g., UUID or custom string).""",
        examples=["session_12345"]
    )
    filter_source: Optional[str] = Field(
        None, 
        max_length=256,
        description="""Optional exact-match filter to restrict the search to a specific source document.
- **Behavior**: If provided, the system will ONLY search chunks that originated from this exact filename.
- **Use Case**: "Chat with this specific document" features.""",
        examples=["AI_ML_Course_Master.pdf"]
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="""Optional dictionary of key-value pairs to restrict search results based on custom metadata.
- **Behavior**: Acts as a strict equality filter (AND condition). Only chunks containing matching metadata will be retrieved.
- **Use Case**: Filtering by department, access level, date, or category.""",
        examples=[{"department": "HR"}]
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
