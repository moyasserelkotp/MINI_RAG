import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from helpers.config import get_settings
from middleware.auth import api_key_middleware
from middleware.rate_limiter import limiter
from middleware.request_id import RequestIDMiddleware
from routes import base, data, nlp
from routes.eval import eval_router
from routes.projects import project_alias_router, projects_router
from routes.sessions import sessions_router
from routes.tasks import tasks_router
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from utils.metrics import add_prometheus_middleware, register_metrics_endpoint

#  Logging 
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if settings.DEBUG:
    logger.warning("DEBUG MODE ENABLED — do not use in production!")

#  Lifespan (startup / shutdown) 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all shared resources once at startup; clean up on shutdown."""
    _settings = get_settings()
    logger.info("Starting %s v%s", _settings.APP_NAME, _settings.APP_VERSION)

    # MongoDB
    app.mongo_conn = AsyncIOMotorClient(_settings.MONGODB_URL)
    app.db_client = app.mongo_conn[_settings.MONGODB_DATABASE]
    logger.info("MongoDB connected: %s", _settings.MONGODB_DATABASE)

    # LLM clients
    llm_factory = LLMProviderFactory(_settings)

    app.generation_client = llm_factory.create(provider=_settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=_settings.GENERATION_MODEL_ID)
    logger.info("Generation: %s / %s", _settings.GENERATION_BACKEND, _settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_factory.create(provider=_settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=_settings.EMBEDDING_MODEL_ID,
        embedding_size=_settings.EMBEDDING_MODEL_SIZE,
    )
    logger.info(
        "Embedding: %s / %s (dim=%s)",
        _settings.EMBEDDING_BACKEND, _settings.EMBEDDING_MODEL_ID, _settings.EMBEDDING_MODEL_SIZE,
    )

    # Vector DB
    app.vectordb_client = VectorDBProviderFactory(_settings).create(provider=_settings.VECTOR_DB_BACKEND)
    app.vectordb_client.connect()
    logger.info("VectorDB: %s", _settings.VECTOR_DB_BACKEND)

    # Template parser
    app.template_parser = TemplateParser(
        language=_settings.PRIMARY_LANG,
        default_language=_settings.DEFAULT_LANG,
    )

    # Cohere rerank client
    app.cohere_client = None
    if _settings.USE_RERANK and _settings.COHERE_API_KEY:
        try:
            import cohere as _cohere
            app.cohere_client = _cohere.Client(_settings.COHERE_API_KEY)
            logger.info("Cohere rerank client initialised")
        except Exception as exc:
            logger.warning("Could not initialise Cohere client: %s", exc)

    # Shared per-worker state
    # initialized_collections — skips Qdrant round-trip after first request
    app.initialized_collections = set()
    # llm_semaphore — caps concurrent LLM API calls to avoid provider rate limits
    app.llm_semaphore = asyncio.Semaphore(10)

    logger.info("Startup complete — ready to serve requests.")
    yield

    # Shutdown
    logger.info("Shutting down…")
    app.mongo_conn.close()
    try:
        app.vectordb_client.disconnect()
    except Exception:
        pass
    logger.info("Shutdown complete.")


#  OpenAPI metadata
_OPENAPI_TAGS = [
    {"name": "General",          "description": "Base routes and health checks."},
    {"name": "Projects",         "description": "Manage user projects."},
    {"name": "Data Management",  "description": "Upload and manage project data."},
    {"name": "Search & NLP",     "description": "Query vectors and generate answers."},
    {"name": "Background Tasks", "description": "Trigger and monitor long-running background tasks."},
    {"name": "Sessions",         "description": "Manage chat sessions and message history."},
    {"name": "Evaluation",       "description": "RAG evaluation endpoints."},
]

#  Application 
app = FastAPI(
    title="MINI-RAG",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Transforms static documents of any domain into an intelligent, "
        "context-aware AI expert with semantic memory, multilingual reasoning "
        "(Arabic + English), and enterprise-grade observability."
    ),
    version="1.0.0",
    contact={"name": "MINI-RAG", "url": "https://github.com/your-org/mini-rag"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
    debug=settings.DEBUG,
    openapi_tags=_OPENAPI_TAGS,
    swagger_ui_parameters={"operationsSorter": "method"},
    # Docs are disabled in production
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

#  Middleware (registration order matters — last added = outermost) 

# 1. Prometheus — must be first so it captures all requests
try:
    add_prometheus_middleware(app)
except Exception as exc:
    logger.exception("Failed to register Prometheus middleware: %s", exc)

# 2. Auth — validates API key before anything else reaches the route
app.add_middleware(BaseHTTPMiddleware, dispatch=api_key_middleware)
if settings.ENABLE_AUTH:
    logger.info("API key auth ENABLED (%d key(s) configured)", len(settings.API_KEYS))
else:
    logger.warning("⚠️  Authentication is DISABLED — enable in production")

# 3. Request-ID — attaches X-Request-ID to every request/response
app.add_middleware(RequestIDMiddleware)

# 4. CORS
_cors_origins = settings.CORS_ALLOWED_ORIGINS
if not settings.DEBUG and _cors_origins == ["*"]:
    logger.error("CORS wildcard '*' is not allowed in production. Defaulting to [].")
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#  Routers 
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(projects_router)
app.include_router(project_alias_router)
app.include_router(tasks_router)
app.include_router(sessions_router)
app.include_router(eval_router)

#  Prometheus metrics endpoint 
try:
    register_metrics_endpoint(app)
except Exception as exc:
    logger.exception("Failed to register /metrics endpoint: %s", exc)


"""
MINI-RAG — Application entry point.

Responsibilities:
  - Configure logging
  - Define the FastAPI lifespan (startup / shutdown)
  - Register middleware (auth, CORS, rate-limit, request-ID, Prometheus)
  - Mount all routers
"""
