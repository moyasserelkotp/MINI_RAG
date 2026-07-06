from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from helpers.config import get_settings


class ProcessRequest(BaseModel):

    file_id: Optional[str] = None
    chunk_size: Optional[int] = Field(default_factory=lambda: get_settings().FILE_PROCESS_CHUNK_SIZE)
    overlap_size: Optional[int] = Field(default_factory=lambda: get_settings().FILE_PROCESS_OVERLAP_SIZE)
    do_reset: Optional[int] = 0


# Phase 7: URL ingestion request schema

class URLIngestRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=10,
        max_length=2048,
        description="Public URL to fetch and index (must be http:// or https://)"
    )
    chunk_size: Optional[int] = Field(
        default_factory=lambda: get_settings().FILE_PROCESS_CHUNK_SIZE,
        description="Chunk size for text splitting"
    )
    overlap_size: Optional[int] = Field(
        default_factory=lambda: get_settings().FILE_PROCESS_OVERLAP_SIZE,
        description="Overlap size between chunks"
    )
    do_reset: Optional[int] = Field(0, ge=0, le=1, description="Set 1 to rebuild chunks for this URL")
