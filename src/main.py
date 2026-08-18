from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from helpers.config import get_settings
from routes import base, data, nlp
from routes.projects import projects_router
from routes.tasks import tasks_router
from routes.projects import project_alias_router
from routes.sessions import sessions_router
from routes.eval import eval_router
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from utils.metrics import add_prometheus_middleware, register_metrics_endpoint
from starlette.middleware.base import BaseHTTPMiddleware
from middleware.auth import api_key_middleware
from middleware.request_id import RequestIDMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from middleware.rate_limiter import limiter

import logging

#  Configure logging based on settings 
settings = get_settings()

# Convert string log level to logging level
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("=" * 70)
logger.info("Starting MINI-RAG Application")
logger.info("=" * 70)
if settings.DEBUG:
    logger.warning("DEBUG MODE ENABLED - Only use this for development!")
logger.info("Log Level: %s", settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    #  Startup 
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # MongoDB
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    logger.info("MongoDB connected: %s", settings.MONGODB_DATABASE)

    # LLM factory
    llm_factory = LLMProviderFactory(settings)
    vectordb_factory = VectorDBProviderFactory(settings)

    # Generation client
    app.generation_client = llm_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
    logger.info("Generation backend: %s / %s", settings.GENERATION_BACKEND, settings.GENERATION_MODEL_ID)

    # Embedding client
    app.embedding_client = llm_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )
    logger.info("Embedding backend: %s / %s (dim=%s)", settings.EMBEDDING_BACKEND, settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE)

    # Vector DB
    app.vectordb_client = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
    app.vectordb_client.connect()
    logger.info("VectorDB connected: %s", settings.VECTOR_DB_BACKEND)

    # Template parser
    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )
    logger.info("Template parser ready (lang=%s)", settings.PRIMARY_LANG)

    # Cohere rerank client (optional) — instantiated once to avoid per-request overhead
    app.cohere_client = None
    if getattr(settings, "USE_RERANK", False):
        cohere_key = getattr(settings, "COHERE_API_KEY", None)
        if cohere_key:
            try:
                import cohere as _cohere
                app.cohere_client = _cohere.Client(cohere_key)
                logger.info("Cohere rerank client initialised")
            except Exception as exc:
                logger.warning("Could not initialise Cohere client: %s", exc)

    logger.info("Startup complete — ready to serve requests.")

    yield

    #  Shutdown 
    logger.info("Shutting down…")
    app.mongo_conn.close()
    try:
        app.vectordb_client.disconnect()
    except Exception:
        pass
    logger.info("Shutdown complete.")


openapi_tags = [
    {
        "name": "General",
        "description": "Base routes and health checks.",
    },
    {
        "name": "Projects",
        "description": "Manage user projects.",
    },
    {
        "name": "Data Management",
        "description": "Upload and manage project data.",
    },
    {
        "name": "Search & NLP",
        "description": "Query vectors and generate answers.",
    },
    {
        "name": "Background Tasks",
        "description": "Trigger and monitor long-running background tasks.",
    },
    {
        "name": "Sessions",
        "description": "Manage chat sessions and message history.",
    },
    {
        "name": "Evaluation",
        "description": "RAG evaluation endpoints.",
    },
]

app = FastAPI(
    title="MINI-RAG",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Transforms static documents of any domain into an intelligent, context-aware AI expert "
        "with semantic memory, multilingual reasoning (Arabic + English), "
        "and enterprise-grade observability."
    ),
    version="1.0.0",
    contact={
        "name": "MINI-RAG",
        "url": "https://github.com/your-org/mini-rag",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    debug=settings.DEBUG,
    openapi_tags=openapi_tags,
    swagger_ui_parameters={"operationsSorter": "method"},
    # Disable interactive docs in production; enable only when DEBUG=True.
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# Store settings in app state for access in routes
app.state.DEBUG = settings.DEBUG
app.state.LOG_LEVEL = settings.LOG_LEVEL

#  Rate Limiter 
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
logger.info("Rate limiter initialised (global=%s)", settings.RATE_LIMIT_GLOBAL)

if settings.DEBUG:
    logger.warning("⚠️  FastAPI DEBUG MODE ENABLED ⚠️ ")
    logger.warning("This should ONLY be used for development!")
    logger.warning("Sensitive information may be exposed in error messages.")

# Add Prometheus middleware BEFORE the app starts
# This must happen before any requests are processed
try:
    add_prometheus_middleware(app)
    logger.info("Prometheus middleware registered")
except Exception as e:
    logger.exception("Failed to register Prometheus middleware: %s", e)

# Auth Middleware 
# Runs BEFORE CORS — all requests pass through API key validation first
app.add_middleware(BaseHTTPMiddleware, dispatch=api_key_middleware)
if settings.ENABLE_AUTH:
    logger.info("API key auth ENABLED (%d key(s) configured)", len(settings.API_KEYS))
else:
    logger.warning("⚠️  AUTHENTICATION IS DISABLED — set ENABLE_AUTH=True in production")

# Request-ID middleware — attaches X-Request-ID to every request/response
app.add_middleware(RequestIDMiddleware)

# CORS — origins controlled via CORS_ALLOWED_ORIGINS in settings / .env
cors_origins = get_settings().CORS_ALLOWED_ORIGINS
if not settings.DEBUG and cors_origins == ["*"]:
    logger.error("CORS wildcard '*' is not allowed in production. Defaulting to empty list.")
    cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(projects_router)
app.include_router(project_alias_router)
app.include_router(tasks_router)
app.include_router(sessions_router)
app.include_router(eval_router)

# Register the /metrics endpoint after middleware is set up
try:
    register_metrics_endpoint(app)
    logger.info("Prometheus /metrics endpoint registered")
except Exception as e:
    logger.exception("Failed to register /metrics endpoint: %s", e)
