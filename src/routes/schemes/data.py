from pydantic import BaseModel
from typing import Optional


class ProcessRequest(BaseModel):

    file_id: Optional[str] = None
    chunk_size: Optional[int] = 512
    overlap_size: Optional[int] = 50
    do_reset: Optional[int] = 0
