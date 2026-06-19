from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Optional
import logging
import json
import hashlib
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

logger = logging.getLogger(__name__)


class NLPController(BaseController):

    def __init__(self, db_client, vectordb_client, generation_client, embedding_client, template_parser, cohere_client=None):
        super().__init__()
        self.db_client = db_client
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        # FIX: accept pre-built cohere client from app startup (avoids per-request instantiation)
        self.cohere_client = cohere_client
        # Cache to avoid repeated VectorDB round-trips for collection existence
        self._initialized_collections: set = set()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def create_collection_name(self, project_id: str) -> str:
        prefix = getattr(self.app_settings, "VECTOR_DB_COLLECTION_PREFIX", "collection")
        return f"{prefix}_{project_id}".strip()

    def _get_cache_collection_name(self, project_id: str) -> str:
        dim = self.embedding_client.embedding_size
        return f"semantic_cache_{project_id}_{dim}".strip()

    def _init_cache_collection(self, project_id: str):
        col_name = self._get_cache_collection_name(project_id)
        if col_name in self._initialized_collections:
            return
        if not self.vectordb_client.is_collection_existed(collection_name=col_name):
            self.vectordb_client.create_collection(
                collection_name=col_name,
                embedding_size=self.embedding_client.embedding_size,
                do_reset=False
            )
        self._initialized_collections.add(col_name)

    def _get_entity_collection_name(self, project_id: str) -> str:
        dim = self.embedding_client.embedding_size
        return f"entities_{project_id}_{dim}".strip()

    def _init_entity_collection(self, project_id: str):
        col_name = self._get_entity_collection_name(project_id)
        if col_name in self._initialized_collections:
            return
        if not self.vectordb_client.is_collection_existed(collection_name=col_name):
            self.vectordb_client.create_collection(
                collection_name=col_name,
                embedding_size=self.embedding_client.embedding_size,
                do_reset=False
            )
        self._initialized_collections.add(col_name)

    # ── Collection management ─────────────────────────────────────────────────

    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))

    # ── Indexing ──────────────────────────────────────────────────────────────

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

        # Record documents processed metric
        try:
            record_document_processed(project_id=project.project_id, count=len(f_texts))
        except Exception:
            logger.exception("Failed to record document processed metric")

        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search_vector_db_collection(
        self,
        project: Project,
        text: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
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
            threshold = getattr(self.app_settings, "SEARCH_SCORE_THRESHOLD", 0.0) or 0.0

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
            # Record retrieval latency and chunks retrieved
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

    # ── Memory Subroutines ────────────────────────────────────────────────────

    async def _update_session_summary(self, session_id: str, project_id, session_model, message_model):
        try:
            messages = await message_model.get_messages_by_session(session_id, limit=20)
            text_block = "\n".join([f"{m.role}: {m.text}" for m in messages])
            prompt = f"Summarize the following conversation focusing on the main topics and important context. Stay concise:\n\n{text_block}"
            
            chat_history = [
                self.generation_client.construct_prompt("You are a summarization AI.", self.generation_client.enums.SYSTEM.value)
            ]
            # Measure generation latency
            try:
                gen_start = time.monotonic()
                summary = await asyncio.to_thread(self.generation_client.generate_text, prompt=prompt, chat_history=chat_history)
                gen_duration = time.monotonic() - gen_start
                try:
                    record_generation_latency(backend=getattr(self.generation_client, 'generation_model_id', 'unknown'), duration=gen_duration)
                except Exception:
                    logger.exception("Failed to record generation latency")
            except Exception as ge:
                logger.error("Generation failed during summary: %s", ge)
                try:
                    record_generation_error(backend=getattr(self.generation_client, 'generation_model_id', 'unknown'))
                except Exception:
                    logger.exception("Failed to record generation error metric")
                summary = None

            if summary:
                await session_model.update_summary(session_id, summary)
                # Reset counter to 0 to avoid continuous summarizing
                await self.db_client["chat_sessions"].update_one(
                    {"session_id": session_id},
                    {"$set": {"message_count": 0}}
                )
        except Exception as e:
            logger.error("Summary Generation Failed: %s", e)

    async def _extract_and_save_entities(self, session_id: str, project: Project, query: str, answer: str, q_vec: list):
        try:
            prompt = (
                "Identify any crucial facts, user preferences, or distinct entities the user stated about themselves "
                "or the conversation implicitly established that are worth remembering long-term. "
                "Return them as a bulleted list. If there is nothing crucial to remember, reply with nothing.\n\n"
                f"User: {query}\n"
            )
            chat_history = [
                self.generation_client.construct_prompt("You extract explicit long-term memory facts. Keep it short.", self.generation_client.enums.SYSTEM.value)
            ]
            facts = await asyncio.to_thread(self.generation_client.generate_text, prompt=prompt, chat_history=chat_history)
            facts = facts.strip() if facts else ""

            if facts and len(facts) > 5 and "nothing" not in facts.lower():
                self._init_entity_collection(project.project_id)
                col_name = self._get_entity_collection_name(project.project_id)
                # Save facts to entity memory with the query vector
                fact_id = int(hashlib.md5((session_id + query).encode()).hexdigest(), 16) % (2**63 - 1)
                await asyncio.to_thread(
                    self.vectordb_client.insert_one,
                    collection_name=col_name,
                    text=facts,
                    vector=q_vec,
                    metadata={"source": "entity_extractor", "session_id": session_id},
                    record_id=fact_id
                )
        except Exception as e:
            logger.error("Entity Extraction Failed: %s", e)


    async def _condense_query(self, query: str, session_messages: list) -> str:
        """Use chat history to re-write a standalone query for better RAG retrieval."""
        try:
            if not session_messages:
                return query

            history_lines = []
            for m in reversed(session_messages):
                role = "User" if m.role == "user" else "Assistant"
                history_lines.append(f"{role}: {m.text}")

            history_text = "\n".join(history_lines)
            condense_prompt = self.template_parser.get(
                "rag", "condense_prompt", {"chat_history": history_text, "query": query}
            )

            if not condense_prompt:
                return query

            chat_history = [
                self.generation_client.construct_prompt(
                    "You are a query condensation assistant.", 
                    self.generation_client.enums.SYSTEM.value
                )
            ]
            
            condensed = await asyncio.to_thread(
                self.generation_client.generate_text, 
                prompt=condense_prompt, 
                chat_history=chat_history
            )
            
            return condensed.strip() if condensed else query
        except Exception as e:
            logger.error("Query Condensation Failed: %s", e)
            return query


    # ── Answer ────────────────────────────────────────────────────────────────

    async def answer_rag_question(
        self,
        project: Project,
        query: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
        use_rerank: bool = True,
        session_id: Optional[str] = None
    ):
        """Run full RAG pipeline and return (answer, full_prompt, chat_history, sources, cached).

        Returns:
            answer      – str | None | False
            full_prompt – str | None
            chat_history – list | None
            sources     – list[dict] with keys text, source, page, score
            cached      – bool (True when answer came from semantic cache)
        """
        answer, full_prompt, chat_history = None, None, None
        sources: List[dict] = []
        cached = False

        # Feature flags
        use_window = getattr(self.app_settings, "USE_WINDOW_MEMORY", True)
        window_k = getattr(self.app_settings, "WINDOW_MEMORY_K", 5)
        use_summary = getattr(self.app_settings, "USE_SUMMARY_MEMORY", True)
        use_entity = getattr(self.app_settings, "USE_ENTITY_MEMORY", True)
        use_vector = getattr(self.app_settings, "USE_VECTOR_MEMORY", True)
        use_cache = getattr(self.app_settings, "USE_SEMANTIC_CACHE", True)
        cache_threshold = getattr(self.app_settings, "SEMANTIC_CACHE_THRESHOLD", 0.95)

        # 0. Database Session Initialization (Needed early for condensation)
        session_obj = None
        session_messages = []
        session_model = None
        message_model = None
        
        if session_id:
            from models.SessionModel import SessionModel
            from models.MessageModel import MessageModel
            session_model = await SessionModel.create_instance(self.db_client)
            message_model = await MessageModel.create_instance(self.db_client)
            
            if use_window or use_summary:
                session_obj = await session_model.get_session_or_create_one(session_id, project.id)
            if use_window and session_obj:
                # History limit is set by WINDOW_MEMORY_K
                session_messages = await message_model.get_messages_by_session(session_id, limit=window_k)

        # 1. Query Condensation (Context Awareness)
        search_query = query
        if session_messages:
            search_query = await self._condense_query(query, session_messages)
            if search_query != query:
                logger.info("Condensed query: '%s' -> '%s'", query, search_query)

        # 2. Embed Search Query
        cache_vec = await asyncio.to_thread(self.embedding_client.embed_text, text=search_query, document_type=DocumentTypeEnum.QUERY.value)
        if not cache_vec:
            logger.error("Failed to embed search query.")
            return False, None, None, [], False

        # 3. Semantic Cache Check
        self._init_cache_collection(project.project_id)
        cache_col_name = self._get_cache_collection_name(project.project_id)

        if use_cache:
            try:
                cached_results = await asyncio.to_thread(
                    self.vectordb_client.search_by_vector,
                    collection_name=cache_col_name, vector=cache_vec, limit=1, score_threshold=cache_threshold
                )
                if cached_results and len(cached_results) > 0:
                    logger.info("Semantic Cache hit for query: %s", search_query)
                    cached_answer = cached_results[0].payload.get("metadata", {}).get("answer")
                    if cached_answer:
                        try:
                            record_cache_hit(project_id=project.project_id)
                        except Exception:
                            pass
                        # Return immediately with cached=True flag
                        return cached_answer, "[CACHED RESPONSES BYPASS PROMPT]", [], [], True
                    # Cache entry exists but answer field is missing — treat as miss
                    try:
                        record_cache_miss(project_id=project.project_id)
                    except Exception:
                        pass
                else:
                    try:
                        record_cache_miss(project_id=project.project_id)
                    except Exception:
                        pass
            except Exception as e:
                logger.error("Error accessing semantic cache: %s", e)

        # 4. Entity Memory Retrieval
        entities_text = ""
        if session_id and use_entity:
            try:
                ent_col = self._get_entity_collection_name(project.project_id)
                self._init_entity_collection(project.project_id)
                ent_docs = await asyncio.to_thread(
                    self.vectordb_client.search_by_vector,
                    collection_name=ent_col,
                    vector=cache_vec,
                    limit=3,
                    score_threshold=0.6
                )
                if ent_docs:
                    entities_text = "\n".join([d.payload.get("text", "") for d in ent_docs])
            except Exception as e:
                logger.error("Error accessing entity memory: %s", e)

        # 5. Vector DB Retrieval
        retrieved_documents = []
        if use_vector:
            retrieved_documents = await asyncio.to_thread(
                self.search_vector_db_collection,
                project=project, text=search_query, limit=limit, use_hybrid=use_hybrid, score_threshold=score_threshold
            )
            if retrieved_documents is None:
                return False, None, None, [], False

            # Build sources list from retrieved docs before reranking
            sources = [
                {
                    "text": d.payload.get("text", "")[:300],  # Truncate for API response
                    "source": (d.payload.get("metadata") or {}).get("source", "unknown"),
                    "page": (d.payload.get("metadata") or {}).get("page"),
                    "score": round(float(getattr(d, "score", 0.0)), 4),
                }
                for d in retrieved_documents
            ]

            # Cohere Rerank
            should_rerank = use_rerank and getattr(self.app_settings, "USE_RERANK", False)
            if should_rerank and retrieved_documents:
                try:
                    # FIX: use pre-built client injected at startup instead of creating per-request
                    cohere_client = self.cohere_client
                    if cohere_client:
                        docs_texts = [d.payload.get("text", "") for d in retrieved_documents]
                        rerank_model = getattr(self.app_settings, "RERANK_MODEL_ID", "rerank-multilingual-v3.0")
                        reranked = await asyncio.to_thread(
                            cohere_client.rerank,
                            model=rerank_model, query=search_query, documents=docs_texts, top_n=min(limit, len(docs_texts))
                        )
                        reranked_docs = []
                        for r in reranked.results:
                            d = retrieved_documents[r.index]
                            d.score = float(r.relevance_score)
                            reranked_docs.append(d)
                        retrieved_documents = reranked_docs
                        # Rebuild sources list with updated rerank scores
                        sources = [
                            {
                                "text": d.payload.get("text", "")[:300],
                                "source": (d.payload.get("metadata") or {}).get("source", "unknown"),
                                "page": (d.payload.get("metadata") or {}).get("page"),
                                "score": round(float(getattr(d, "score", 0.0)), 4),
                            }
                            for d in reranked_docs
                        ]
                    else:
                        logger.warning("USE_RERANK=True but cohere_client is not initialised; skipping rerank.")
                except Exception as e:
                    logger.error("Cohere Rerank failed: %s", e)

        # 6. Prompt Construction
        system_prompt = self.template_parser.get("rag", "system_prompt")
        
        if use_summary and session_obj and session_obj.summary:
            system_prompt += f"\n\n[Conversation Summary]:\n{session_obj.summary}"

        if use_entity and entities_text:
            system_prompt += f"\n\n[Important Retained Facts]:\n{entities_text}"

        chat_history_prompts = [
            self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)
        ]

        if use_window:
            # We use reversed because get_messages_by_session usually returns latest messages first
            for msg in reversed(session_messages):
                role = self.generation_client.enums.USER.value if msg.role == "user" else self.generation_client.enums.ASSISTANT.value
                chat_history_prompts.append(self.generation_client.construct_prompt(prompt=msg.text, role=role))

        document_parts = []
        for idx, doc in enumerate(retrieved_documents or []):
            chunk_text = doc.payload.get("text", "")
            doc_metadata = doc.payload.get("metadata", {})
            source = doc_metadata.get("source", "unknown") if doc_metadata else "unknown"
            score = round(getattr(doc, "score", 0.0), 4)

            doc_prompt = self.template_parser.get(
                "rag", "document_prompt", {"doc_num": idx + 1, "chunk_text": chunk_text, "source": source, "score": score}
            )
            document_parts.append(doc_prompt)

        documents_prompts = "\n\n".join(document_parts)
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})
        full_prompt = "\n\n".join([documents_prompts, footer_prompt]) if documents_prompts else footer_prompt

        # 7. Generate Answer
        # We no longer short-circuit based on retrieved_documents alone to allow greetings
        answer = await asyncio.to_thread(self.generation_client.generate_text, prompt=full_prompt, chat_history=chat_history_prompts)

        # 8. Post-generation Memory Saves
        if answer:
            # Semantic Cache Sub
            is_negative_answer = "cannot answer" in answer.lower() or "not contain the answer" in answer.lower()

            if use_cache and not is_negative_answer:
                try:
                    # FIX: use search_query (the condensed query) as the stored text so
                    # the vector and stored text are always aligned for future lookups.
                    cache_id = int(hashlib.md5(search_query.encode()).hexdigest(), 16) % (2**63 - 1)
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            self.vectordb_client.insert_one,
                            collection_name=cache_col_name,
                            text=search_query,
                            vector=cache_vec,
                            metadata={"answer": answer},
                            record_id=cache_id,
                        )
                    )
                    # FIX: log errors from fire-and-forget tasks instead of silently dropping them
                    task.add_done_callback(
                        lambda t: logger.error("Cache write failed: %s", t.exception())
                        if t.exception() else None
                    )
                except Exception:
                    pass

            # Window + Summary Sub
            if session_id and session_model and message_model:
                from models.db_schemes.chat_message import ChatMessage
                await message_model.create_message(ChatMessage(session_id=session_id, role="user", text=query))
                await message_model.create_message(ChatMessage(session_id=session_id, role="assistant", text=answer))
                await session_model.increment_message_count(session_id, 2)

                if use_summary and session_obj and (session_obj.message_count + 2 >= getattr(self.app_settings, "SUMMARY_TRIGGER_LENGTH", 10)):
                    summary_task = asyncio.create_task(
                        self._update_session_summary(session_id, project.id, session_model, message_model)
                    )
                    # FIX: log errors from fire-and-forget summary task
                    summary_task.add_done_callback(
                        lambda t: logger.error("Session summary task failed: %s", t.exception())
                        if t.exception() else None
                    )

            # Entity Action Sub
            if session_id and use_entity:
                entity_task = asyncio.create_task(
                    self._extract_and_save_entities(session_id, project, query, answer, cache_vec)
                )
                # FIX: log errors from fire-and-forget entity extraction task
                entity_task.add_done_callback(
                    lambda t: logger.error("Entity extraction task failed: %s", t.exception())
                    if t.exception() else None
                )

        return answer, full_prompt, chat_history_prompts, sources, cached
