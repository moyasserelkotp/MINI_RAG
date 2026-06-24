from typing import List, Optional, Union
import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator


class Settings(BaseSettings):

    APP_NAME: str = "MINI-TOURISM RAG"
    APP_VERSION: str = "1.0.0"

    # Application Settings 
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    FILE_ALLOWED_TYPES: Union[List[str], str] = []
    FILE_MAX_SIZE: int = 0          # bytes (already in bytes, no extra scaling)
    FILE_DEFAULT_CHUNK_SIZE: int = 512_000  # streaming read chunk (bytes)
    FILE_PROCESS_CHUNK_SIZE: int = 512
    FILE_PROCESS_OVERLAP_SIZE: int = 50
    CHUNK_STRATEGY: str = "recursive"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @validator("FILE_ALLOWED_TYPES", pre=True)
    def _split_allowed_types(cls, v):
        if isinstance(v, str):
            value = v.strip()
            if not value:
                return []
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    return [item.strip().lower() for item in decoded if isinstance(item, str)]
            except json.JSONDecodeError:
                pass
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        if isinstance(v, list):
            return [item.strip().lower() for item in v if isinstance(item, str)]
        return v

    # MongoDB 
    MONGODB_URL: str
    MONGODB_DATABASE: str

    # LLM backends 
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    LLAMA_API_KEY: Optional[str] = None
    LLAMA_API_URL: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None

    # Corrected typos: DAFAULT → DEFAULT (keep old names as aliases for .env compat)
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = 1024
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = 512
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = 0.1

    # Friendly aliases used inside code
    @property
    def INPUT_DEFAULT_MAX_CHARACTERS(self) -> int:
        return self.INPUT_DAFAULT_MAX_CHARACTERS or 1024

    @property
    def GENERATION_DEFAULT_MAX_TOKENS(self) -> int:
        return self.GENERATION_DAFAULT_MAX_TOKENS or 512

    @property
    def GENERATION_DEFAULT_TEMPERATURE(self) -> float:
        return self.GENERATION_DAFAULT_TEMPERATURE or 0.1

    #  Vector DB 
    VECTOR_DB_BACKEND: str
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_PATH: str = "qdrant_db"
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = "cosine"
    VECTOR_DB_COLLECTION_PREFIX: str = "collection"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: Optional[str] = None

    # Max texts per batch embedding call
    MAX_EMBEDDING_BATCH_SIZE: int = 96

    # Minimum relevance score to include in search results (0.0 = no filter)
    SEARCH_SCORE_THRESHOLD: float = 0.0

    # Language templates 
    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"
    # RAG Enhancements 
    USE_RERANK: bool = False
    RERANK_MODEL_ID: str = "rerank-multilingual-v3.0"  # Cohere rerank model
    SEMANTIC_CACHE_THRESHOLD: float = 0.95
    TOP_K_RESULTS: int = 5                    # Default top-K for retrieval
    HYBRID_SEARCH_SEMANTIC_WEIGHT: float = 0.6  # Weight for dense vs BM25 in hybrid search

    # RAG Memory Settings 
    USE_WINDOW_MEMORY: bool = True
    WINDOW_MEMORY_K: int = 5
    USE_SUMMARY_MEMORY: bool = True
    SUMMARY_TRIGGER_LENGTH: int = 10          # Messages before auto-summarisation fires
    USE_ENTITY_MEMORY: bool = True
    USE_VECTOR_MEMORY: bool = True
    USE_SEMANTIC_CACHE: bool = True

    #  Celery / RabbitMQ 
    RABBITMQ_DEFAULT_USER: str = "minirag"
    RABBITMQ_DEFAULT_PASS: str = "minirag_rabbit_2222"
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: str = "5672"
    RABBITMQ_VHOST: str = "minirag_vhost"

    # Celery / Redis 
    REDIS_HOST: str = "redis"
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: str = "minirag_redis_2222"
    REDIS_CELERY_DB: str = "1"   # DB index for Celery results
    REDIS_CACHE_DB: str = "0"    # DB index for app-level cache

    # CORS 
    # FIX: restrict allowed origins; wildcard '*' is unsafe in production
    # Set to ["*"] only for local development; list specific domains in prod
    CORS_ALLOWED_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://localhost:8080"]

    @validator("CORS_ALLOWED_ORIGINS", pre=True)
    def _split_cors_origins(cls, v):
        if isinstance(v, str):
            value = v.strip()
            if not value:
                return []
            try:
                import json as _json
                decoded = _json.loads(value)
                if isinstance(decoded, list):
                    return [o.strip() for o in decoded if isinstance(o, str)]
            except Exception:
                pass
            return [o.strip() for o in value.split(",") if o.strip()]
        if isinstance(v, list):
            return v
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
