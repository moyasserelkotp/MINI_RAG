import uuid
import time
import asyncio
import logging
from typing import Optional
from utils.metrics import record_cache_hit, record_cache_miss



_CACHE_TTL_DEFAULT = 86_400

logger = logging.getLogger(__name__)

_COLLECTION_LOCKS: dict[str, asyncio.Lock] = {}

class CacheService:
    def __init__(self, vectordb_client, embedding_client, generation_client, app_settings, initialized_collections: set):
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.generation_client = generation_client
        self.app_settings = app_settings
        self._initialized_collections = initialized_collections

    def get_cache_collection_name(self, project_id: str) -> str:
        dim = self.embedding_client.embedding_size
        return f"semantic_cache_{project_id}_{dim}".strip()

    async def init_cache_collection(self, project_id: str):
        col_name = self.get_cache_collection_name(project_id)
        if col_name in self._initialized_collections:
            return
        
        lock = _COLLECTION_LOCKS.setdefault(col_name, asyncio.Lock())
        async with lock:
            if col_name in self._initialized_collections:
                return
            exists = await asyncio.to_thread(self.vectordb_client.is_collection_existed, collection_name=col_name)
            if not exists:
                await asyncio.to_thread(
                    self.vectordb_client.create_collection,
                    collection_name=col_name,
                    embedding_size=self.embedding_client.embedding_size,
                    do_reset=False
                )
            self._initialized_collections.add(col_name)

    def build_cache_key(self, project_id: str, query: str) -> int:
        gen_model = getattr(
            self.generation_client, "generation_model_id",
            getattr(self.generation_client, "model_id", "unknown")
        )
        emb_model = getattr(
            self.embedding_client, "embedding_model_id",
            getattr(self.embedding_client, "model_id", "unknown")
        )
        lang = getattr(self.app_settings, "PRIMARY_LANG", "en")
        prompt_version = "v1"

        canonical = "|".join([
            str(project_id),
            str(query),
            str(gen_model),
            str(emb_model),
            prompt_version,
            str(lang),
        ])
        uid = uuid.uuid5(uuid.NAMESPACE_DNS, canonical)
        return uid.int & ((1 << 63) - 1)

    async def check_semantic_cache(self, project_id: str, cache_vec: list, use_cache: bool, cache_threshold: float) -> Optional[str]:
        if not use_cache:
            return None
            
        await self.init_cache_collection(project_id)
        cache_col_name = self.get_cache_collection_name(project_id)
        
        try:
            cached_results = await asyncio.to_thread(
                self.vectordb_client.search_by_vector,
                collection_name=cache_col_name, vector=cache_vec, limit=1, score_threshold=cache_threshold
            )
            if cached_results and len(cached_results) > 0:
                payload = cached_results[0].payload
                metadata = payload.get("metadata", {}) or {}
                # RAG-01: reject TTL-expired entries
                expires_at = metadata.get("expires_at")
                if expires_at and time.time() > expires_at:
                    try:
                        record_cache_miss(project_id=project_id)
                    except Exception:
                        pass
                    return None
                cached_answer = metadata.get("answer")
                if cached_answer:
                    try:
                        record_cache_hit(project_id=project_id)
                    except Exception:
                        pass
                    return cached_answer
                try:
                    record_cache_miss(project_id=project_id)
                except Exception:
                    pass
            else:
                try:
                    record_cache_miss(project_id=project_id)
                except Exception:
                    pass
        except Exception as e:
            logger.error("Error accessing semantic cache: %s", e)
            try:
                record_cache_miss(project_id=project_id)
            except Exception:
                pass
            
        return None
