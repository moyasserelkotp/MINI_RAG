"""Request-ID middleware.

Every HTTP request is tagged with a unique X-Request-ID:
  - If the client sends X-Request-ID the value is preserved (up to 128 chars).
  - Otherwise a UUID4 is generated server-side.

The ID is stored in a ContextVar so that any logger running within the request
context can include it without being passed explicitly.

Usage in a logger:
    from middleware.request_id import get_request_id
    logger.info("Processing started", extra={"request_id": get_request_id()})
"""
import uuid
import logging
from contextvars import ContextVar
# pyrefly: ignore [missing-import]
from starlette.middleware.base import BaseHTTPMiddleware
# pyrefly: ignore [missing-import]
from starlette.requests import Request
# pyrefly: ignore [missing-import]
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Context variable: available anywhere within the current async request context.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Maximum length of a client-supplied X-Request-ID we will trust.
_MAX_REQUEST_ID_LENGTH = 128

HEADER_NAME = "X-Request-ID"


def get_request_id() -> str:
    """Return the request ID for the current context, or an empty string."""
    return _request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID to every request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(HEADER_NAME, "")

        # Validate / truncate client-supplied IDs to prevent huge header abuse.
        if incoming and len(incoming) <= _MAX_REQUEST_ID_LENGTH:
            request_id = incoming
        else:
            if incoming:
                logger.debug(
                    "Ignoring oversized X-Request-ID header (%d chars); generating new ID",
                    len(incoming),
                )
            request_id = str(uuid.uuid4())

        # Store in context so downstream code can access it without thread-locals.
        token = _request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id_var.reset(token)

        # Always echo the ID back so clients can correlate responses.
        response.headers[HEADER_NAME] = request_id
        return response
