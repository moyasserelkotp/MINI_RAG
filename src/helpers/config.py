from typing import List, Optional, Union
import json

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator


class Settings(BaseSettings):

    APP_NAME: str = "mini-RAG-App"
    APP_VERSION: str = "0.1"

    OPENAI_API_KEY: Optional[str] = ""

    # allow either a list or a comma‑separated string so that dotenv parsing
    # doesn't attempt JSON decoding (which fails on unquoted comma lists).
    FILE_ALLOWED_TYPES: Union[List[str], str] = []
    FILE_MAX_SIZE: int = 0
    FILE_DEFAULT_CHUNK_SIZE: int = 512_000  # 512KB

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @validator("FILE_ALLOWED_TYPES", pre=True)
    def _split_allowed_types(cls, v):
        if isinstance(v, str):
            value = v.strip()
            if not value:
                return []
            # try JSON first to support explicit arrays
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    return [item.strip().lower() for item in decoded if isinstance(item, str)]
            except json.JSONDecodeError:
                pass
            # fall back to comma-separated
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        if isinstance(v, list):
            return [item.strip().lower() for item in v if isinstance(item, str)]
        return v


def get_settings():
    return Settings()
