from pydantic import BaseModel, Field
from typing import Optional
from helpers.config import get_settings


class ProcessRequest(BaseModel):

    file_id: Optional[str] = None
    chunk_size: Optional[int] = Field(default_factory=lambda: get_settings().FILE_PROCESS_CHUNK_SIZE)
    overlap_size: Optional[int] = Field(default_factory=lambda: get_settings().FILE_PROCESS_OVERLAP_SIZE)
    do_reset: Optional[int] = 0
