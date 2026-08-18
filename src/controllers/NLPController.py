import hashlib
from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from models.enums.DataBaseEnum import DataBaseEnum
from typing import List, Optional
import logging
import json
import uuid
import asyncio
import time

from utils.metrics import (
    record_retrieval_latency,
    record_chunks_retrieved,
    record_generation_latency,
    record_document_processed,
    record_retrieval_error,
    record_generation_error,
    record_cache_hit,
    record_cache_miss,
)

from services.cache_service import CacheService
from services.memory_service import MemoryService
from services.retrieval_service import RetrievalService
from services.prompt_service import PromptService

logger = logging.getLogger(__name__)



_BACKGROUND_TASKS: set = set()


def _safe_task_callback(label: str):
    def _cb(t: asyncio.Task):
        _BACKGROUND_TASKS.discard(t)
        try:
            exc = t.exception()
            if exc:
                logger.error("%s failed: %s", label, exc)
        except Exception:
            pass
    return _cb


class NLPController(BaseController):

    def __init__(self, db_client, vectordb_client, generation_client, embedding_client, template_parser, cohere_client=None):
        super().__init__()
        self.db_client = db_client
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.cohere_client = cohere_client
        self._initialized_collections: set = set()

        self.cache_service = CacheService(
            vectordb_client, embedding_client, generation_client, self.app_settings, self._initialized_collections
        )
        self.memory_service = MemoryService(
            db_client, vectordb_client, generation_client, embedding_client, template_parser, self.app_settings, self._initialized_collections
        )
        self.retrieval_service = RetrievalService(
            vectordb_client, cohere_client, self.app_settings
        )
        self.prompt_service = PromptService(
            template_parser, generation_client
        )

    #  Helpers 

    def create_collection_name(self, project_id: str) -> str:
        prefix = getattr(self.app_settings, "VECTOR_DB_COLLECTION_PREFIX", "collection")
        return f"{prefix}_{project_id}".strip()

    #  Collection management 

    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        
        # Also clean up associated memory collections
        try:
            cache_col = self.cache_service.get_cache_collection_name(project.project_id)
            self.vectordb_client.delete_collection(collection_name=cache_col)
        except Exception:
            pass
            
        try:
            ent_col = self.memory_service.get_entity_collection_name(project.project_id)
            self.vectordb_client.delete_collection(collection_name=ent_col)
        except Exception:
            pass
            
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )
        def _default(obj):
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            try:
                return str(obj)
            except Exception:
                return None
        return json.loads(json.dumps(collection_info, default=_default))

    #  Indexing 
    def _embed_texts_batch(self, texts: List[str], document_type: str) -> List[list]:
        batch_size = self.app_settings.MAX_EMBEDDING_BATCH_SIZE

        if hasattr(self.embedding_client, "embed_batch"):
            vectors: List[Optional[list]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_vectors = self.embedding_client.embed_batch(
                    texts=batch, document_type=document_type
                )
                if batch_vectors:
                    vectors.extend(batch_vectors)
                else:
                    vectors.extend([None] * len(batch))
            return vectors

        return [
            self.embedding_client.embed_text(text=t, document_type=document_type)
            for t in texts
        ]

    def index_into_vector_db(
        self,
        project: Project,
        chunks: List[DataChunk],
        chunks_ids: List[int],
        do_reset: bool = False,
    ) -> bool:
        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]

        vectors = self._embed_texts_batch(texts, document_type=DocumentTypeEnum.DOCUMENT.value)

        valid = [
            (t, v, m, rid)
            for t, v, m, rid in zip(texts, vectors, metadata, chunks_ids)
            if v is not None
        ]
        if not valid:
            logger.error("All embeddings failed for project %s", project.project_id)
            return False

        f_texts, f_vectors, f_metadata, f_ids = zip(*valid)

        self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=list(f_texts),
            metadata=list(f_metadata),
            vectors=list(f_vectors),
            record_ids=list(f_ids),
        )

        try:
            record_document_processed(project_id=project.project_id, count=len(f_texts))
        except Exception:
            logger.exception("Failed to record document processed metric")

        return True

    #  Search 

    def search_vector_db_collection(
        self,
        project: Project,
        text: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ):
        collection_name = self.create_collection_name(project_id=project.project_id)

        vector = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.QUERY.value
        )
        if not vector or len(vector) == 0:
            logger.error("Failed to embed query: %s", text)
            return None

        threshold = score_threshold
        if threshold is None:
            threshold = getattr(self.app_settings, "SEARCH_SCORE_THRESHOLD", 0.0)
            if threshold is None:
                threshold = 0.0

        try:
            start = time.monotonic()
            semantic_weight = getattr(self.app_settings, "HYBRID_SEARCH_SEMANTIC_WEIGHT", 0.6)
            if use_hybrid:
                results = self.vectordb_client.hybrid_search(
                    collection_name=collection_name,
                    query_text=text,
                    vector=vector,
                    limit=limit,
                    semantic_weight=semantic_weight,
                )
            else:
                results = self.vectordb_client.search_by_vector(
                    collection_name=collection_name,
                    vector=vector,
                    limit=limit,
                    score_threshold=threshold if threshold > 0 else None,
                )
            duration = time.monotonic() - start
            try:
                record_retrieval_latency(project_id=project.project_id, duration=duration)
                if results:
                    record_chunks_retrieved(project_id=project.project_id, count=len(results))
            except Exception:
                logger.exception("Failed to record retrieval metrics")
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            try:
                record_retrieval_error(project_id=project.project_id)
            except Exception:
                logger.exception("Failed to record retrieval error metric")
            return None

        if results is None:
            return []

        if use_hybrid and threshold and threshold > 0:
            results = [r for r in results if getattr(r, "score", 0.0) >= threshold]

        return list(results)


    #  Answer 

    async def answer_rag_question(
        self,
        project: Project,
        query: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
        use_rerank: bool = True,
        session_id: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
    ):
        """Run full RAG pipeline and return (answer, full_prompt, chat_history, sources, cached)."""
        answer, full_prompt, chat_history_prompts = None, None, None
        sources: List[dict] = []
        cached = False

        use_window = getattr(self.app_settings, "USE_WINDOW_MEMORY", True)
        window_k = getattr(self.app_settings, "WINDOW_MEMORY_K", 5)
        use_summary = getattr(self.app_settings, "USE_SUMMARY_MEMORY", True)
        use_entity = getattr(self.app_settings, "USE_ENTITY_MEMORY", True)
        use_vector = getattr(self.app_settings, "USE_VECTOR_MEMORY", True)
        use_cache = getattr(self.app_settings, "USE_SEMANTIC_CACHE", True)
        cache_threshold = getattr(self.app_settings, "SEMANTIC_CACHE_THRESHOLD", 0.95)

        # 0. Session Initialization
        session_obj, session_messages, session_model, message_model = await self.memory_service.prepare_session(
            session_id, project.project_id, use_window, use_summary, window_k
        )

        # 1. Query Condensation
        search_query = query
        if session_messages:
            search_query = await self.memory_service.condense_query(query, session_messages)

        # 2. Embed Search Query
        cache_vec = await asyncio.to_thread(self.embedding_client.embed_text, text=search_query, document_type=DocumentTypeEnum.QUERY.value)
        if not cache_vec:
            return False, None, None, [], False

        # 3. Semantic Cache Check
        cached_answer = await self.cache_service.check_semantic_cache(project.project_id, cache_vec, use_cache, cache_threshold)
        if cached_answer:
            # Save the exchange to session memory even on a cache hit so that
            # follow-up questions (e.g. "what is my name?") can still use context.
            if session_id and session_model and message_model:
                from models.db_schemes.chat_message import ChatMessage
                await message_model.create_message(ChatMessage(session_id=session_id, role="user", text=query))
                await message_model.create_message(ChatMessage(session_id=session_id, role="assistant", text=cached_answer))
                await session_model.increment_message_count(session_id, 2)
            return cached_answer, "[CACHED RESPONSES BYPASS PROMPT]", [], [], True

        # 4. Entity Memory Retrieval
        entities_text = ""
        if session_id and use_entity:
            try:
                ent_col = self.memory_service.get_entity_collection_name(project.project_id)
                await self.memory_service.init_entity_collection(project.project_id)
                ent_docs = await asyncio.to_thread(
                    self.vectordb_client.search_by_vector,
                    collection_name=ent_col, vector=cache_vec, limit=3, score_threshold=0.6
                )
                if ent_docs:
                    entities_text = "\n".join([d.payload.get("text", "") for d in ent_docs])
            except Exception as e:
                logger.error("Error accessing entity memory: %s", e)

        # 5. Vector DB Retrieval & Rerank
        retrieved_documents, sources = await self.retrieval_service.retrieve_and_rerank_context(
            self.search_vector_db_collection, project, search_query, limit, use_hybrid, score_threshold, metadata_filter, use_vector, use_rerank
        )

        # RAG-05: Return a structured no-context signal so the caller can render
        # a proper "no relevant documents" message instead of an LLM hallucination.
        if not retrieved_documents and use_vector:
            return "NO_CONTEXT", None, [], [], False

        # 6. Prompt Construction
        full_prompt, chat_history_prompts = self.prompt_service.format_system_prompt(
            query, session_obj, session_messages, entities_text, retrieved_documents, use_summary, use_entity, use_window
        )

        # 7. Generate Answer
        answer = await asyncio.to_thread(self.generation_client.generate_text, prompt=full_prompt, chat_history=chat_history_prompts)

        # 8. Post-generation Memory Saves
        if answer:
            await self._post_generation_memory_saves(
                session_id=session_id,
                project=project,
                query=query,
                answer=answer,
                cache_vec=cache_vec,
                session_obj=session_obj,
                session_model=session_model,
                message_model=message_model,
                use_cache=use_cache,
                use_entity=use_entity,
                use_summary=use_summary,
                search_query=search_query,
            )

        return answer, full_prompt, chat_history_prompts, sources, False

    # ─── Shared post-generation memory saves ───────────────────────────────

    async def _post_generation_memory_saves(
        self,
        session_id: Optional[str],
        project,
        query: str,
        answer: str,
        cache_vec: list,
        session_obj,
        session_model,
        message_model,
        use_cache: bool,
        use_entity: bool,
        use_summary: bool,
        search_query: str,
    ):
        """Fire all post-generation memory persistence tasks."""
        is_negative = "cannot answer" in answer.lower() or "not contain the answer" in answer.lower()

        # 1. Semantic cache write
        if use_cache and not is_negative:
            try:
                import time as _time
                cache_ttl = getattr(self.app_settings, "SEMANTIC_CACHE_TTL_SECONDS", 86_400)
                cache_id = self.cache_service.build_cache_key(project.project_id, search_query)
                cache_col_name = self.cache_service.get_cache_collection_name(project.project_id)
                task = asyncio.create_task(
                    asyncio.to_thread(
                        self.vectordb_client.insert_one,
                        collection_name=cache_col_name,
                        text=search_query,
                        vector=cache_vec,
                        metadata={"answer": answer, "expires_at": _time.time() + cache_ttl},
                        record_id=cache_id,
                    )
                )
                _BACKGROUND_TASKS.add(task)
                task.add_done_callback(_safe_task_callback("Cache write"))
            except Exception:
                pass

        # 2. Save user + assistant messages
        if session_id and session_model and message_model:
            from models.db_schemes.chat_message import ChatMessage
            await message_model.create_message(ChatMessage(session_id=session_id, role="user", text=query))
            await message_model.create_message(ChatMessage(session_id=session_id, role="assistant", text=answer))
            await session_model.increment_message_count(session_id, 2)

            # 3. Summary trigger
            if use_summary and session_obj and (
                session_obj.message_count + 2 >= getattr(self.app_settings, "SUMMARY_TRIGGER_LENGTH", 10)
            ):
                summary_task = asyncio.create_task(
                    self.memory_service.update_session_summary(session_id, project.id, session_model, message_model)
                )
                _BACKGROUND_TASKS.add(summary_task)
                summary_task.add_done_callback(_safe_task_callback("Session summary"))

        # 4. Entity extraction
        if session_id and use_entity:
            entity_task = asyncio.create_task(
                self.memory_service.extract_and_save_entities(session_id, project, query, answer, cache_vec)
            )
            _BACKGROUND_TASKS.add(entity_task)
            entity_task.add_done_callback(_safe_task_callback("Entity extraction"))

    async def answer_rag_stream(
        self,
        project: Project,
        query: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
    ):
        """Run the RAG pipeline and yield LLM tokens one-by-one (async generator)."""
        import asyncio

        use_window = getattr(self.app_settings, "USE_WINDOW_MEMORY", True)
        window_k = getattr(self.app_settings, "WINDOW_MEMORY_K", 5)
        use_summary = getattr(self.app_settings, "USE_SUMMARY_MEMORY", True)
        use_entity = getattr(self.app_settings, "USE_ENTITY_MEMORY", True)
        use_vector = getattr(self.app_settings, "USE_VECTOR_MEMORY", True)
        use_cache = getattr(self.app_settings, "USE_SEMANTIC_CACHE", True)
        cache_threshold = getattr(self.app_settings, "SEMANTIC_CACHE_THRESHOLD", 0.95)

        session_obj, session_messages, session_model, message_model = await self.memory_service.prepare_session(
            session_id, project.project_id, use_window, use_summary, window_k
        )

        search_query = query
        if session_messages:
            search_query = await self.memory_service.condense_query(query, session_messages)

        cache_vec = await asyncio.to_thread(self.embedding_client.embed_text, text=search_query, document_type=DocumentTypeEnum.QUERY.value)
        if not cache_vec:
            yield {"error": "embedding failed"}
            return

        cached_answer = await self.cache_service.check_semantic_cache(project.project_id, cache_vec, use_cache, cache_threshold)
        if cached_answer:
            # Save exchange to session memory even on cache hit
            if session_id and session_model and message_model:
                from models.db_schemes.chat_message import ChatMessage
                await message_model.create_message(ChatMessage(session_id=session_id, role="user", text=query))
                await message_model.create_message(ChatMessage(session_id=session_id, role="assistant", text=cached_answer))
                await session_model.increment_message_count(session_id, 2)
            yield cached_answer
            return

        entities_text = ""
        if session_id and use_entity:
            try:
                ent_col = self.memory_service.get_entity_collection_name(project.project_id)
                await self.memory_service.init_entity_collection(project.project_id)
                ent_docs = await asyncio.to_thread(
                    self.vectordb_client.search_by_vector, collection_name=ent_col, vector=cache_vec, limit=3, score_threshold=0.6
                )
                if ent_docs:
                    entities_text = "\n".join([d.payload.get("text", "") for d in ent_docs])
            except Exception:
                pass

        retrieved_documents, _ = await self.retrieval_service.retrieve_and_rerank_context(
            self.search_vector_db_collection, project, search_query, limit, use_hybrid, score_threshold, metadata_filter, use_vector, use_rerank=True
        )
        
        if not retrieved_documents and use_vector:
            yield "No relevant documents found for your query."
            return

        full_prompt, chat_history_prompts = self.prompt_service.format_system_prompt(
            query, session_obj, session_messages, entities_text, retrieved_documents, use_summary, use_entity, use_window
        )

        system_prompt = self.template_parser.get("rag", "system_prompt")
        if chat_history_prompts and chat_history_prompts[0].get("role") == self.generation_client.enums.SYSTEM.value:
            system_prompt = chat_history_prompts[0].get("content", system_prompt)

        collected = []
        if hasattr(self.generation_client, "stream_text"):
            try:
                for token in self.generation_client.stream_text(
                    prompt=full_prompt, chat_history=chat_history_prompts, system_prompt=system_prompt
                ):
                    collected.append(token)
                    yield token
            except Exception as exc:
                logger.error("Streaming generation failed: %s", exc)
                yield {"error": f"Error during streaming: {exc}"}
                return
        else:
            try:
                full_answer = self.generation_client.generate_text(
                    prompt=full_prompt, chat_history=chat_history_prompts, system_prompt=system_prompt
                )
                if full_answer:
                    yield full_answer
                    collected = [full_answer]
            except Exception as exc:
                yield {"error": str(exc)}
                return

        if collected:
            full_text = "".join(collected)
            await self._post_generation_memory_saves(
                session_id=session_id,
                project=project,
                query=query,
                answer=full_text,
                cache_vec=cache_vec,
                session_obj=session_obj,
                session_model=session_model,
                message_model=message_model,
                use_cache=use_cache,
                use_entity=use_entity,
                use_summary=use_summary,
                search_query=search_query,
            )
