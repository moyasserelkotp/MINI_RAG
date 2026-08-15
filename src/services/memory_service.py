import asyncio
import logging
import hashlib
import time
from typing import Optional
from models.db_schemes import Project
from models.enums.DataBaseEnum import DataBaseEnum
from utils.metrics import record_generation_latency, record_generation_error

logger = logging.getLogger(__name__)

def _safe_task_callback(label: str):
    def _cb(t: asyncio.Task):
        try:
            exc = t.exception()
            if exc:
                logger.error("%s failed: %s", label, exc)
        except Exception:
            pass
    return _cb

class MemoryService:
    def __init__(self, db_client, vectordb_client, generation_client, embedding_client, template_parser, app_settings, initialized_collections: set):
        self.db_client = db_client
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.app_settings = app_settings
        self._initialized_collections = initialized_collections

    def get_entity_collection_name(self, project_id: str) -> str:
        dim = self.embedding_client.embedding_size
        return f"entities_{project_id}_{dim}".strip()

    def init_entity_collection(self, project_id: str):
        col_name = self.get_entity_collection_name(project_id)
        if col_name in self._initialized_collections:
            return
        if not self.vectordb_client.is_collection_existed(collection_name=col_name):
            self.vectordb_client.create_collection(
                collection_name=col_name,
                embedding_size=self.embedding_client.embedding_size,
                do_reset=False
            )
        self._initialized_collections.add(col_name)

    async def prepare_session(self, session_id: Optional[str], project_id: str, use_window: bool, use_summary: bool, window_k: int):
        session_obj, session_messages, session_model, message_model = None, [], None, None
        if session_id:
            from models.SessionModel import SessionModel
            from models.MessageModel import MessageModel
            session_model = await SessionModel.create_instance(self.db_client)
            message_model = await MessageModel.create_instance(self.db_client)
            if use_window or use_summary:
                session_obj = await session_model.get_session_or_create_one(session_id, project_id)
            if use_window and session_obj:
                session_messages = await message_model.get_messages_by_session(session_id, limit=window_k)
        return session_obj, session_messages, session_model, message_model

    async def update_session_summary(self, session_id: str, project_id, session_model, message_model):
        try:
            messages = await message_model.get_messages_by_session(session_id, limit=20)
            text_block = "\n".join([f"{m.role}: {m.text}" for m in messages])
            prompt = f"Summarize the following conversation focusing on the main topics and important context. Stay concise:\n\n{text_block}"
            
            chat_history = [
                self.generation_client.construct_prompt("You are a summarization AI.", self.generation_client.enums.SYSTEM.value)
            ]
            try:
                gen_start = time.monotonic()
                summary = await asyncio.to_thread(self.generation_client.generate_text, prompt=prompt, chat_history=chat_history)
                gen_duration = time.monotonic() - gen_start
                try:
                    record_generation_latency(backend=getattr(self.generation_client, 'generation_model_id', 'unknown'), duration=gen_duration)
                except Exception:
                    pass
            except Exception as ge:
                logger.error("Generation failed during summary: %s", ge)
                try:
                    record_generation_error(backend=getattr(self.generation_client, 'generation_model_id', 'unknown'))
                except Exception:
                    pass
                summary = None

            if summary:
                await session_model.update_summary(session_id, summary)
                await self.db_client[DataBaseEnum.COLLECTION_CHAT_SESSION_NAME.value].update_one(
                    {"session_id": session_id},
                    {"$set": {"message_count": 0}}
                )
        except Exception as e:
            logger.error("Summary Generation Failed: %s", e)

    async def extract_and_save_entities(self, session_id: str, project: Project, query: str, answer: str, q_vec: list):
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
                self.init_entity_collection(project.project_id)
                col_name = self.get_entity_collection_name(project.project_id)
                fact_id = int(hashlib.sha256((session_id + query).encode()).hexdigest(), 16) % (2**63 - 1)
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

    async def condense_query(self, query: str, session_messages: list) -> str:
        try:
            if not session_messages:
                return query

            history_lines = []
            for m in session_messages:
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
