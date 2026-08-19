# pyrefly: ignore [missing-import]
from slowapi import Limiter
# pyrefly: ignore [missing-import]
from slowapi.util import get_remote_address
from helpers.config import get_settings

settings = get_settings()

# Build Redis URI from settings — shared across all uvicorn workers so rate
# limits are enforced globally, not per-process.
# Falls back to in-memory storage if Redis is not reachable (safe for local dev).
_redis_uri = (
    f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/2"
    if settings.REDIS_HOST and settings.REDIS_PASSWORD
    else None
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_GLOBAL],
    storage_uri=_redis_uri,   # None → falls back to thread-safe in-memory store
)
