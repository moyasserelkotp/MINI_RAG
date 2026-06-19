from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)


@base_router.get("/", summary="Welcome")
async def welcome(app_settings: Settings = Depends(get_settings)):
    return {
        "app_name": app_settings.APP_NAME,
        "app_version": app_settings.APP_VERSION,
    }


@base_router.get("/info", summary="Application info")
async def info(request: Request, app_settings: Settings = Depends(get_settings)):
    """Returns app name, version, active backends, and enabled memory features.
    Referenced in Quick Start guide and Docker healthchecks."""
    environment = "development" if app_settings.DEBUG else "production"
    return {
        "status": "ok",
        "app_name": app_settings.APP_NAME,
        "version": app_settings.APP_VERSION,
        "environment": environment,
        "backends": {
            "generation": app_settings.GENERATION_BACKEND,
            "generation_model": app_settings.GENERATION_MODEL_ID,
            "embedding": app_settings.EMBEDDING_BACKEND,
            "embedding_model": app_settings.EMBEDDING_MODEL_ID,
            "embedding_dimensions": app_settings.EMBEDDING_MODEL_SIZE,
            "vector_db": app_settings.VECTOR_DB_BACKEND,
        },
        "memory_features": {
            "semantic_cache": app_settings.USE_SEMANTIC_CACHE,
            "window_memory": app_settings.USE_WINDOW_MEMORY,
            "summary_memory": app_settings.USE_SUMMARY_MEMORY,
            "entity_memory": app_settings.USE_ENTITY_MEMORY,
            "vector_memory": app_settings.USE_VECTOR_MEMORY,
            "reranking": app_settings.USE_RERANK,
        },
        "chunk_strategy": app_settings.CHUNK_STRATEGY,
        "supported_languages": app_settings.PRIMARY_LANG,
    }


@base_router.get("/health", summary="Basic health check")
async def health():
    """Returns 200 OK when the app is running."""
    return {"status": "ok"}


@base_router.get("/health/detailed", summary="Detailed health check")
async def health_detailed(request: Request):
    """Returns connectivity status of MongoDB and Qdrant."""
    db_ok = False
    vectordb_ok = False

    # MongoDB ping
    try:
        await request.app.mongo_conn.admin.command("ping")
        db_ok = True
    except Exception:
        pass

    # Qdrant check
    try:
        collections = request.app.vectordb_client.list_all_collections()
        vectordb_ok = collections is not None
    except Exception:
        pass

    all_ok = db_ok and vectordb_ok
    return JSONResponse(
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ok" if all_ok else "degraded",
            "mongodb": "ok" if db_ok else "error",
            "vectordb": "ok" if vectordb_ok else "error",
        },
    )
