from pydantic import BaseModel
from typing import Optional


class ProcessRequest(BaseModel):
    # optional: when omitted or null, all project files are processed
    file_id: Optional[str] = None
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0
