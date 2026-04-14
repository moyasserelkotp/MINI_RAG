from typing import List, Optional, Union
import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator


class Settings(BaseSettings):

    APP_NAME: str = "mini-RAG-App"
    APP_VERSION: str = "0.1"

    # ── File upload ──────────────────────────────────────────────────────────
    FILE_ALLOWED_TYPES: Union[List[str], str] = []
    FILE_MAX_SIZE: int = 0          # bytes (already in bytes, no extra scaling)
    FILE_DEFAULT_CHUNK_SIZE: int = 512_000  # streaming read chunk (bytes)

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

    # ── MongoDB ──────────────────────────────────────────────────────────────
    MONGODB_URL: str
    MONGODB_DATABASE: str

    # ── LLM backends ─────────────────────────────────────────────────────────
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    LLAMA_API_KEY: Optional[str] = None
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

    # ── Vector DB ─────────────────────────────────────────────────────────────
    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = "cosine"
    VECTOR_DB_COLLECTION_PREFIX: str = "collection"

    # Max texts per batch embedding call
    MAX_EMBEDDING_BATCH_SIZE: int = 96

    # Minimum relevance score to include in search results (0.0 = no filter)
    SEARCH_SCORE_THRESHOLD: float = 0.0

    # ── Language templates ────────────────────────────────────────────────────
    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
