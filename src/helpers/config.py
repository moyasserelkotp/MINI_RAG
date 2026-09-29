from typing import List, Literal, Optional, Union
import json
from functools import lru_cache
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict
# pyrefly: ignore [missing-import]
from pydantic import validator


class Settings(BaseSettings):

    APP_NAME: str = "MINI-RAG"
    APP_VERSION: str = "1.0.0"

    # Application Settings 
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Authentication
    ENABLE_AUTH: bool = True
    API_KEYS: Union[List[str], str] = []   

    # Rate Limiting
    RATE_LIMIT_ANSWER: str = "10/minute"   
    RATE_LIMIT_UPLOAD: str = "20/minute"   
    RATE_LIMIT_GLOBAL: str = "100/minute"  
    FILE_ALLOWED_TYPES: Union[List[str], str] = []
    FILE_MAX_SIZE: int = 0          
    FILE_DEFAULT_CHUNK_SIZE: int = 512_000 
    FILE_PROCESS_CHUNK_SIZE: int = 512
    FILE_PROCESS_OVERLAP_SIZE: int = 50
    CHUNK_STRATEGY: str = "recursive"

    model_config = SettingsConfigDict(env_file=str(Path(__file__).parent.parent / ".env"), env_file_encoding="utf-8")

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

    @validator("API_KEYS", pre=True)
    def _split_api_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, list):
            return [k.strip() for k in v if isinstance(k, str)]
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
    TAVILY_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None

    INPUT_DEFAULT_MAX_CHARACTERS: int = 1024
    GENERATION_DEFAULT_MAX_TOKENS: int = 512
    GENERATION_DEFAULT_TEMPERATURE: float = 0.1

    #  Vector DB 
    VECTOR_DB_BACKEND: str
    VECTOR_DB_URL: Optional[str] = None         
    VECTOR_DB_API_KEY: Optional[str] = None       
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
    SEMANTIC_CACHE_TTL_SECONDS: int = 86_400   # RAG-01: 24 h default; set 0 to disable TTL

    #  Agent Settings 
    # Master switch — set to False to disable all agentic endpoints
    AGENT_ENABLED: bool = True

    # Mode controls how much agentic reasoning to apply:
    #   OFF         — behaves like Traditional RAG (no agent)
    #   ROUTER      — agent selects the correct tool, no evaluation loop
    #   AGENT       — router + tools + retrieval evaluation
    #   FULL_AGENT  — router + planner + tools + evaluation + retry loops
    AGENT_MODE: str = "FULL_AGENT"

    # Optional separate LLM backend/model for agent reasoning.
    # Falls back to GENERATION_BACKEND / GENERATION_MODEL_ID if not set.
    AGENT_LLM_BACKEND: Optional[str] = None
    AGENT_MODEL_ID: Optional[str] = None

    # Safety limits — MUST be enforced to prevent infinite loops
    MAX_AGENT_STEPS: int = 8
    MAX_RETRIEVAL_ATTEMPTS: int = 3
    MAX_TOOL_CALLS: int = 10

    # Hard wall-clock timeout for a single agent graph execution (seconds).
    # Prevents runaway requests from tying up resources indefinitely.
    AGENT_TIMEOUT_SECONDS: int = 60

    # Feature flags
    ENABLE_QUERY_REWRITE: bool = True
    # When True, an LLM judges retrieval quality — adds ~1-2s per turn.
    # Only fires for COMPLEX_MULTI_STEP queries even when True.
    # Set False to rely on the fast deterministic score check only.
    ENABLE_RETRIEVAL_EVALUATION: bool = False
    ENABLE_QUERY_PLANNING: bool = True
    ENABLE_AGENT_TRACING: bool = True  # persist agent run traces in MongoDB

    # Minimum retrieval quality score to consider results sufficient (0.0–1.0)
    AGENT_SCORE_THRESHOLD: float = 0.35

    # Web Search Tool — uses Tavily (TAVILY_API_KEY required)
    ENABLE_WEB_SEARCH: bool = True

    # Semantic cache for the agent path (mirrors NLP path cache behaviour)
    # When True, identical/near-identical queries skip the full agent graph.
    ENABLE_SEMANTIC_CACHE: bool = True
    # Cosine-similarity threshold for agent cache hits (0.0–1.0).
    # Higher = stricter matching; 0.95 returns cached answer only for near-identical queries.
    CACHE_SIMILARITY_THRESHOLD: float = 0.95
    # TTL (seconds) for agent-path semantic cache entries. 0 = no expiry.
    CACHE_TTL_SECONDS: int = 86_400  # 24 hours
    CACHE_PROMPT_VERSION: str = "v1"

    #  Celery / RabbitMQ 
    RABBITMQ_DEFAULT_USER: str = "minirag"
    RABBITMQ_DEFAULT_PASS: str = "minirag_rabbit_2222"
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: str = "5672"
    RABBITMQ_VHOST: str = "minirag_vhost"

    # Celery / Redis 
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: str = "minirag_redis_2222"
    REDIS_CELERY_DB: str = "1"   # DB index for Celery results
    REDIS_CACHE_DB: str = "0"    # DB index for app-level cache

    # CORS 
    CORS_ALLOWED_ORIGINS: Union[List[str], str] = []

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
