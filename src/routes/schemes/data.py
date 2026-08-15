from pydantic import BaseModel, Field, validator
from typing import Optional, List
from helpers.config import get_settings
from .system import BaseResponse


class ProcessRequest(BaseModel):
    """Request body for /data/process endpoint.

    Invariant enforced: overlap_size must be strictly less than chunk_size.
    do_reset accepts both bool (True/False) and legacy int (0/1) for backward
    compatibility with existing clients.
    """
    file_id: Optional[str] = None
    chunk_size: Optional[int] = Field(
        default_factory=lambda: get_settings().FILE_PROCESS_CHUNK_SIZE,
        ge=10,
        description="Token/character size of each chunk (minimum 10)",
    )
    overlap_size: Optional[int] = Field(
        default_factory=lambda: get_settings().FILE_PROCESS_OVERLAP_SIZE,
        ge=0,
        description="Overlap between consecutive chunks (must be < chunk_size)",
    )
    # Accept both bool and int 0/1 for backward compatibility
    do_reset: Optional[int] = Field(
        0, ge=0, le=1,
        description="Set 1 (or true) to discard existing chunks and reprocess from scratch",
    )

    @validator("overlap_size", always=True)
    def _overlap_lt_chunk(cls, v, values):
        chunk_size = values.get("chunk_size")
        if chunk_size is not None and v is not None and v >= chunk_size:
            raise ValueError(
                f"overlap_size ({v}) must be strictly less than chunk_size ({chunk_size})"
            )
        return v


class URLIngestRequest(BaseModel):
    """Request body for URL ingestion.

    Only http:// and https:// schemes are permitted.  The SSRF guard in the
    route still applies after scheme validation.
    """
    url: str = Field(
        ...,
        min_length=10,
        max_length=2048,
        description="Public URL to fetch and index (must be http:// or https://)",
    )
    chunk_size: Optional[int] = Field(
        default_factory=lambda: get_settings().FILE_PROCESS_CHUNK_SIZE,
        ge=10,
        description="Chunk size for text splitting",
    )
    overlap_size: Optional[int] = Field(
        default_factory=lambda: get_settings().FILE_PROCESS_OVERLAP_SIZE,
        ge=0,
        description="Overlap size between chunks (must be < chunk_size)",
    )
    do_reset: Optional[int] = Field(
        0, ge=0, le=1,
        description="Set 1 to rebuild chunks for this URL",
    )

    @validator("url")
    def _require_http_scheme(cls, v: str) -> str:
        import urllib.parse
        scheme = urllib.parse.urlparse(v).scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Only http:// and https:// URLs are allowed; got scheme '{scheme}'"
            )
        return v

    @validator("overlap_size", always=True)
    def _overlap_lt_chunk(cls, v, values):
        chunk_size = values.get("chunk_size")
        if chunk_size is not None and v is not None and v >= chunk_size:
            raise ValueError(
                f"overlap_size ({v}) must be strictly less than chunk_size ({chunk_size})"
            )
        return v


class AssetUploadResponse(BaseResponse):
    file_id: str
    asset_id: str


class AssetItem(BaseModel):
    id: str
    asset_name: str
    asset_size: int
    asset_type: str
    asset_pushed_at: Optional[str] = None


class AssetListResponse(BaseResponse):
    project_id: str
    total: int
    assets: List[AssetItem]


class BatchUploadResult(BaseModel):
    filename: str
    status: str
    signal: Optional[str] = None
    file_id: Optional[str] = None
    asset_name: Optional[str] = None


class BatchUploadResponse(BaseResponse):
    project_id: str
    results: List[BatchUploadResult]


class URLIngestResponse(BaseResponse):
    url: Optional[str] = None
    inserted_chunks: Optional[int] = None
    asset_id: Optional[str] = None
    error: Optional[str] = None
