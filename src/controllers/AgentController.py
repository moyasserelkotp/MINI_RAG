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
        initialized_collections: set = None,
    ):
        super().__init__()
        self.db_client = db_client
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.cohere_client = cohere_client

        # CRITICAL-02: use codebase-consistent relative imports (no `src.` prefix)
        from services.AgentService import AgentService
        from services.memory_service import MemoryService
        from services.retrieval_service import RetrievalService
        from models.ProjectModel import ProjectModel
        from models.AssetModel import AssetModel
        from models.AgentRunModel import AgentRunModel
        from controllers.NLPController import NLPController

        _initialized_collections = initialized_collections if initialized_collections is not None else set()

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
            nlp_controller=self.nlp_controller
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

        # Get chat history if session_id is provided
        chat_history = []
        if session_id:
            try:
                # CRITICAL-02: use consistent import path (no src. prefix)
                from models.MessageModel import MessageModel
                msg_model = MessageModel(self.db_client)
                history = await msg_model.get_messages_by_session(session_id, limit=10)
                for h in history:
                    chat_history.append({"role": h.role, "content": h.text})
                # Reverse to chronological (get_messages_by_session returns newest-first)
                chat_history.reverse()
            except Exception as e:
                logger.error(f"Error fetching history for agent: {e}")

        # Call the AgentService
        result = await self.agent_service.execute_agent(
            project=project,
            session_id=session_id,
            query=query,
            chat_history=chat_history
        )

        # Optional: Save response to session history just like traditional RAG
        if session_id and result.get("answer"):
            try:
                # CRITICAL-05: ChatSessionModel does not exist — correct class is SessionModel
                from models.MessageModel import MessageModel
                from models.SessionModel import SessionModel
                from models.db_schemes import ChatMessage

                msg_model = MessageModel(self.db_client)
                sess_model = SessionModel(self.db_client)

                await msg_model.create_message(ChatMessage(session_id=session_id, role="user", text=query))
                await msg_model.create_message(ChatMessage(session_id=session_id, role="assistant", text=result["answer"]))
                await sess_model.increment_message_count(session_id, 2)
            except Exception as e:
                logger.error(f"Error saving agent conversation: {e}")

        return result
