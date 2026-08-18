from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {
    "/api/v1/",
    "/api/v1/health",
    "/api/v1/health/detailed",
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
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    # In debug mode, allow docs and metrics without auth
    if settings.DEBUG and request.url.path in {"/docs", "/redoc", "/openapi.json", "/metrics"}:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    valid_keys = settings.API_KEYS

    is_valid = False
    if api_key and valid_keys:
        import hmac
        api_key_bytes = api_key.encode('utf-8')
        for valid_key in valid_keys:
            if hmac.compare_digest(api_key_bytes, valid_key.encode('utf-8')):
                is_valid = True
                break

    if not is_valid:
        logger.warning("Unauthorized access attempt to %s (Invalid or missing API key)", request.url.path)
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Invalid or missing API key. Pass your key via the 'X-API-Key' request header."
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return await call_next(request)
