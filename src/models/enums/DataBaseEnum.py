from enum import Enum


class DataBaseEnum(Enum):

    COLLECTION_PROJECT_NAME = "projects"
    COLLECTION_CHUNK_NAME = "chunks"
    COLLECTION_ASSET_NAME = "assets"
    COLLECTION_CHAT_SESSION_NAME = "chat_sessions"
    COLLECTION_CHAT_MESSAGE_NAME = "chat_messages"
    COLLECTION_AGENT_RUNS_NAME = "agent_runs"
