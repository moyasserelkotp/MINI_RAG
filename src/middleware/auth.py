from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

# Paths that are always accessible without an API key
EXEMPT_PATHS = {
    "/api/v1/",
    "/api/v1/health",
    "/api/v1/health/detailed",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
}

async def api_key_middleware(request: Request, call_next):
    """Middleware that enforces API key authentication when ENABLE_AUTH=True.

    Clients must pass the key via:   X-API-Key: <your-key>

    Health, docs, and metrics endpoints are always accessible without a key.
    """
    from helpers.config import get_settings
    settings = get_settings()

    # Auth disabled — pass through
    if not settings.ENABLE_AUTH:
        return await call_next(request)

    # Exempt paths — always allowed
    if request.url.path in EXEMPT_PATHS or request.url.path.startswith("/docs"):
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    valid_keys = settings.API_KEYS

    if not api_key or api_key not in valid_keys:
        logger.warning("Unauthorized access attempt to %s (Invalid or missing API key)", request.url.path)
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Invalid or missing API key. Pass your key via the 'X-API-Key' request header."
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return await call_next(request)
