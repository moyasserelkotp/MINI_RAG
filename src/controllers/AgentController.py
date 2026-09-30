import uuid
import asyncio
from .BaseController import BaseController
from models.db_schemes import Project
from typing import Optional
import logging


logger = logging.getLogger(__name__)

class AgentController(BaseController):
    def __init__(
        self,
        db_client,
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        cohere_client=None,
        initialized_collections=None,
        llm_semaphore=None,  # FIX-2: semaphore to cap concurrent LLM calls
    ):
        super().__init__()
        self.db_client = db_client
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.cohere_client = cohere_client
        self.llm_semaphore = llm_semaphore  # FIX-2

        # CRITICAL-02: use codebase-consistent relative imports (no `src.` prefix)
        from services.AgentService import AgentService
        from services.memory_service import MemoryService
        from services.retrieval_service import RetrievalService
        from models.ProjectModel import ProjectModel
        from models.AssetModel import AssetModel
        from models.AgentRunModel import AgentRunModel
        from models.MessageModel import MessageModel
        from models.SessionModel import SessionModel
        from controllers.NLPController import NLPController

        if initialized_collections is not None:
            _initialized_collections = initialized_collections
        else:
            from utils.collection_tracker import CollectionInitTracker
            _initialized_collections = CollectionInitTracker()

        self.memory_service = MemoryService(
            db_client, vectordb_client, generation_client, embedding_client, template_parser, self.app_settings, _initialized_collections
        )
        self.retrieval_service = RetrievalService(
            vectordb_client, cohere_client, self.app_settings
        )

        # Set up Models
        self.project_model = ProjectModel(db_client)
        self.asset_model = AssetModel(db_client)
        self.agent_run_model = AgentRunModel(db_client)
        self.msg_model = MessageModel(db_client)
        self.sess_model = SessionModel(db_client)

        self.nlp_controller = NLPController(
            db_client, vectordb_client, generation_client, embedding_client,
            template_parser, cohere_client, _initialized_collections
        )

        self.agent_service = AgentService(
            llm_client=generation_client,
            retrieval_service=self.retrieval_service,
            memory_service=self.memory_service,
            project_model=self.project_model,
            asset_model=self.asset_model,
            agent_run_model=self.agent_run_model,
            nlp_controller=self.nlp_controller,
            llm_semaphore=llm_semaphore,  # FIX-2
        )

    async def init_collections(self):
        """Initialize the agent run collection."""
        await self.agent_run_model.init_collection()

    async def execute_agent(
        self,
        project: Project,
        query: str,
        session_id: Optional[str] = None
    ):
        """Entry point for the Agent API."""
        if not getattr(self.app_settings, "AGENT_ENABLED", True):
            return {"error": "Agent is disabled in settings"}

        # FIX-8: Parallelize chat history fetch and AgentRun creation
        run_id = str(uuid.uuid4())
        chat_history = []
        
        async def fetch_history():
            if not session_id:
                return []
            try:
                history = await self.msg_model.get_messages_by_session(session_id, limit=10)
                hist = [{"role": h.role, "content": h.text} for h in history]
                hist.reverse()
                return hist
            except Exception as e:
                logger.error(f"Error fetching history for agent: {e}")
                return []

        async def create_run():
            from models.db_schemes import AgentRun
            agent_run = AgentRun(
                run_id=run_id,
                project_id=project.id,
                session_id=session_id,
                status="running",
                query=query
            )
            await self.agent_run_model.create_agent_run(agent_run)

        # Run both DB operations concurrently
        chat_history, _ = await asyncio.gather(fetch_history(), create_run())

        # Call the AgentService
        result = await self.agent_service.execute_agent(
            project=project,
            session_id=session_id,
            query=query,
            chat_history=chat_history,
            run_id=run_id
        )

        # Optional: Save response to session history just like traditional RAG
        if session_id and result.get("answer"):
            try:
                from models.db_schemes import ChatMessage

                await self.msg_model.create_message(ChatMessage(session_id=session_id, role="user", text=query))
                await self.msg_model.create_message(ChatMessage(session_id=session_id, role="assistant", text=result["answer"]))
                await self.sess_model.increment_message_count(session_id, 2)
            except Exception as e:
                logger.error(f"Error saving agent conversation: {e}")

        return result
