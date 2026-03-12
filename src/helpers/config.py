from typing import List, Optional, Union
import json

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator


class Settings(BaseSettings):

    APP_NAME: str = "mini-RAG-App"
    APP_VERSION: str = "0.1"

    OPENAI_API_KEY: Optional[str] = ""

    FILE_ALLOWED_TYPES: Union[List[str], str] = []
    FILE_MAX_SIZE: int = 0
    FILE_DEFAULT_CHUNK_SIZE: int = 512_000  

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

    MONGODB_URL: str
    MONGODB_DATABASE: str


def get_settings():
    return Settings()
