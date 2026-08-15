from pydantic import BaseModel, Field
from typing import Dict, Any

class BaseResponse(BaseModel):
    signal: str = Field(..., description="A signal string indicating the result status")

class WelcomeResponse(BaseModel):
    app_name: str
    app_version: str

class InfoResponse(BaseResponse):
    status: str
    app_name: str
    version: str
    environment: str
    backends: Dict[str, Any]
    memory_features: Dict[str, Any]
    chunk_strategy: str
    supported_languages: str

class HealthResponse(BaseModel):
    status: str

class HealthDetailedResponse(BaseModel):
    status: str
    mongodb: str
    vectordb: str
