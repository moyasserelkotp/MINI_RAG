from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

# ──────────────────────────────────────────────────────────────────────────────
# HTTP Metrics
# ──────────────────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP Request Latency", ["method", "endpoint"]
)

# ──────────────────────────────────────────────────────────────────────────────
# RAG-Specific Metrics (Key)
# ──────────────────────────────────────────────────────────────────────────────

# Retrieval Metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds", "Vector DB retrieval latency", ["project_id"]
)
CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved_total", "Number of chunks retrieved per query", ["project_id"]
)

# Generation Metrics
GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds", "LLM generation latency", ["backend"]
)
GENERATION_TOKENS = Counter(
    "rag_generation_tokens_total", "Total tokens used in generation", ["backend"]
)

# Cache Metrics
CACHE_HITS = Counter(
    "rag_cache_hits_total", "Semantic cache hits", ["project_id"]
)
CACHE_MISSES = Counter(
    "rag_cache_misses_total", "Semantic cache misses", ["project_id"]
)

# Processing Metrics
DOCUMENTS_PROCESSED = Counter(
    "rag_documents_processed_total", "Documents processed and indexed", ["project_id"]
)
CHUNKING_LATENCY = Histogram(
    "rag_chunking_duration_seconds", "Document chunking duration", ["strategy"]
)

# Error Metrics
RETRIEVAL_ERRORS = Counter(
    "rag_retrieval_errors_total", "Retrieval failures", ["project_id"]
)
GENERATION_ERRORS = Counter(
    "rag_generation_errors_total", "Generation failures", ["backend"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Record metrics after request is processed
        duration = time.time() - start_time
        endpoint = request.url.path

        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(
            duration
        )
        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, status=response.status_code
        ).inc()

        return response


def add_prometheus_middleware(app: FastAPI):
    """
    Add Prometheus middleware to the app.
    MUST be called before the app starts serving requests.
    """
    app.add_middleware(PrometheusMiddleware)


def register_metrics_endpoint(app: FastAPI):
    """
    Register the /metrics endpoint.
    Can be called after middleware is registered.
    """
    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_metrics(app: FastAPI):
    """
    Setup Prometheus metrics middleware and endpoint.
    Calls both middleware addition and endpoint registration.
    """
    add_prometheus_middleware(app)
    register_metrics_endpoint(app)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions for easy metric recording
# ──────────────────────────────────────────────────────────────────────────────

def record_retrieval_latency(project_id: str, duration: float):
    """Record vector DB retrieval latency"""
    RETRIEVAL_LATENCY.labels(project_id=project_id).observe(duration)

def record_chunks_retrieved(project_id: str, count: int):
    """Record number of chunks retrieved"""
    CHUNKS_RETRIEVED.labels(project_id=project_id).observe(count)

def record_cache_hit(project_id: str):
    """Record cache hit"""
    CACHE_HITS.labels(project_id=project_id).inc()

def record_cache_miss(project_id: str):
    """Record cache miss"""
    CACHE_MISSES.labels(project_id=project_id).inc()

def record_generation_latency(backend: str, duration: float):
    """Record LLM generation latency"""
    GENERATION_LATENCY.labels(backend=backend).observe(duration)

def record_generation_tokens(backend: str, tokens: int):
    """Record tokens used in generation"""
    GENERATION_TOKENS.labels(backend=backend).inc(tokens)

def record_document_processed(project_id: str, count: int = 1):
    """Record documents processed"""
    DOCUMENTS_PROCESSED.labels(project_id=project_id).inc(count)

def record_chunking_latency(strategy: str, duration: float):
    """Record chunking operation latency"""
    CHUNKING_LATENCY.labels(strategy=strategy).observe(duration)

def record_retrieval_error(project_id: str):
    """Record retrieval error"""
    RETRIEVAL_ERRORS.labels(project_id=project_id).inc()

def record_generation_error(backend: str):
    """Record generation error"""
    GENERATION_ERRORS.labels(backend=backend).inc()
