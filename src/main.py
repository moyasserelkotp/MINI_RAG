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
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from utils.metrics import add_prometheus_middleware, register_metrics_endpoint

import logging

# ── Configure logging based on settings ──────────────────────────────────
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
    # ── Startup ──────────────────────────────────────────────────────────────
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

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down…")
    app.mongo_conn.close()
    try:
        app.vectordb_client.disconnect()
    except Exception:
        pass
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Mini RAG",
    description="A RAG-powered question-answering API for data.",
    version="0.2",
    lifespan=lifespan,
    debug=settings.DEBUG,  # Enable/disable FastAPI debug mode
)

# Store settings in app state for access in routes
app.state.DEBUG = settings.DEBUG
app.state.LOG_LEVEL = settings.LOG_LEVEL

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

# CORS — origins controlled via CORS_ALLOWED_ORIGINS in settings / .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().CORS_ALLOWED_ORIGINS,  # FIX: no longer a wildcard '*'
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(projects_router)

# Register the /metrics endpoint after middleware is set up
try:
    register_metrics_endpoint(app)
    logger.info("Prometheus /metrics endpoint registered")
except Exception as e:
    logger.exception("Failed to register /metrics endpoint: %s", e)
