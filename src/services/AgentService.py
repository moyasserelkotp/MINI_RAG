from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime, timezone

from agents.classifier import QueryClassifier
from agents.router import AgentRouter
from agents.query_rewriter import QueryRewriter
from agents.evaluator import RetrievalEvaluator
from agents.planner import AgentPlanner
from agents.graph import create_agent_graph
from tools.registry import ToolRegistry
from tools.search_tool import SearchDocumentsTool
from tools.project_tool import GetProjectInfoTool
from tools.asset_tool import ListProjectAssetsTool
from tools.memory_tool import GetConversationContextTool

from utils.metrics import record_agent_run, record_agent_steps, record_agent_tool_use

from models.db_schemes.agent_run import AgentRun
from helpers.config import get_settings
import logging

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(
        self,
        llm_client,
        retrieval_service,
        memory_service,
        project_model,
        asset_model,
        agent_run_model,
        nlp_controller=None
    ):
        self.app_settings = get_settings()
        self.llm_client = llm_client
        self.retrieval_service = retrieval_service
        self.memory_service = memory_service
        self.project_model = project_model
        self.asset_model = asset_model
        self.agent_run_model = agent_run_model
        self.nlp_controller = nlp_controller
        
        # Initialize components
        self.classifier = QueryClassifier(self.llm_client)
        
        # Tools will be registered per-project in execute_agent since tools need project_id
        # But for efficiency, we can create the graph structure dynamically or pass context.
        # LangGraph allows passing a state that is just a dictionary.
        
        self.router = None # Will be initialized per execution if tool registry changes
        
        threshold = getattr(self.app_settings, "AGENT_SCORE_THRESHOLD", 0.7)
        self.evaluator = RetrievalEvaluator(self.llm_client, score_threshold=threshold)
        self.rewriter = QueryRewriter(self.llm_client)
        
    async def execute_agent(self, project: Any, session_id: str, query: str, chat_history: list = None) -> Dict[str, Any]:
        """
        Executes the agentic RAG workflow.
        """
        run_id = str(uuid.uuid4())
        
        # Create a new AgentRun in DB
        agent_run = AgentRun(
            run_id=run_id,
            project_id=project.id,
            session_id=session_id,
            status="running",
            query=query
        )
        await self.agent_run_model.create_agent_run(agent_run)
        
        try:
            agent_mode = getattr(self.app_settings, "AGENT_MODE", "FULL_AGENT")
            if agent_mode == "OFF":
                return {"error": "Agentic RAG is disabled."}
                
            # 1. Setup Tools for this specific project/session
            tool_registry = ToolRegistry()
            tool_registry.register(SearchDocumentsTool(self.retrieval_service, self.nlp_controller, project))
            tool_registry.register(GetProjectInfoTool(self.project_model, project.project_id))
            tool_registry.register(ListProjectAssetsTool(self.asset_model, project.project_id))
            tool_registry.register(GetConversationContextTool(self.memory_service, project.project_id, session_id))
            
            # 2. Setup Router and Planner with this registry
            router = AgentRouter(self.llm_client, tool_registry)
            planner = AgentPlanner(self.llm_client, tool_registry)
            
            # 3. Create Graph
            graph = create_agent_graph(
                classifier=self.classifier,
                router=router,
                evaluator=self.evaluator,
                rewriter=self.rewriter,
                planner=planner,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                agent_mode=agent_mode
            )
            
            # 4. Initialize State
            initial_state = {
                "run_id": run_id,
                "project_id": project.project_id,
                "session_id": session_id,
                "original_query": query,
                "current_query": query,
                "messages": chat_history or [],
                "plan": [],
                "current_plan_step": 0,
                "next_action": None,
                "tool_calls": [],
                "selected_tool": None,
                "retrieved_context": [],
                "retrieval_attempts": 0,
                "max_retrieval_attempts": getattr(self.app_settings, "MAX_RETRIEVAL_ATTEMPTS", 3),
                "evaluation_result": None,
                "rewrite_count": 0,
                "errors": [],
                "step_count": 0,
                "max_steps": getattr(self.app_settings, "MAX_AGENT_STEPS", 8),
                "final_answer": None,
                "sources": [],
                "trace": []
            }
            
            # 5. Execute Graph
            final_state = await graph.ainvoke(initial_state)
            
            # 6. Update AgentRun record
            update_data = {
                "status": "success",
                "final_answer": final_state.get("final_answer"),
                "steps_count": final_state.get("step_count", 0),
                "tool_calls_count": len(final_state.get("tool_calls", [])),
                "retrieval_attempts": final_state.get("retrieval_attempts", 0),
                "tools_used": [tc["name"] for tc in final_state.get("tool_calls", [])],
                "trace": final_state.get("trace", []),
                "finished_at": datetime.now(timezone.utc)
            }
            
            if final_state.get("errors"):
                update_data["error"] = "; ".join(final_state["errors"])
                update_data["status"] = "failed"
                
            await self.agent_run_model.update_agent_run(run_id, update_data)
            
            # Metrics
            record_agent_run(agent_mode, update_data["status"])
            record_agent_steps(agent_mode, update_data["steps_count"])
            for tool in update_data["tools_used"]:
                record_agent_tool_use(tool)
                
            # Optional Memory save
            if session_id and update_data["status"] == "success":
                await self.memory_service.save_messages(query, final_state.get("final_answer"), session_id)
            
            return {
                "answer": final_state.get("final_answer"),
                "sources": final_state.get("sources", []),
                "run_id": run_id,
                "steps": final_state.get("step_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            await self.agent_run_model.update_agent_run(run_id, {
                "status": "failed",
                "error": str(e),
                "finished_at": datetime.now(timezone.utc)
            })
            record_agent_run("UNKNOWN", "failed")
            raise
