from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from helpers.config import get_settings
from routes import base, data, nlp
from routes.projects import projects_router

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    title="Mini Tourism RAG",
    description="A RAG-powered question-answering API for tourism data.",
    version="0.2",
    lifespan=lifespan,
)

# CORS — adjust origins as needed for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(projects_router)
