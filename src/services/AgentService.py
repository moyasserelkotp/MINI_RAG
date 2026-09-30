from typing import Dict, Any
import uuid
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

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
from tools.web_search_tool import WebSearchTool

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
        nlp_controller=None,
        llm_semaphore: asyncio.Semaphore = None,
        cache_service=None,  # I-3: accept CacheService directly to avoid tight coupling
    ):
        self.app_settings = get_settings()
        self.llm_client = llm_client
        self.retrieval_service = retrieval_service
        self.memory_service = memory_service
        self.project_model = project_model
        self.asset_model = asset_model
        self.agent_run_model = agent_run_model
        self.nlp_controller = nlp_controller

        # I-3: prefer the directly-injected instance; fall back to NLPController's
        # instance for backward compat; final fallback is None (cache disabled).
        if cache_service is not None:
            self.cache_service = cache_service
        elif nlp_controller is not None and hasattr(nlp_controller, 'cache_service'):
            self.cache_service = nlp_controller.cache_service
        else:
            self.cache_service = None
            logger.warning("AgentService: no CacheService available — semantic cache disabled")

        # Use the shared semaphore, or fall back to a generous local one
        # so the service is self-contained even if no semaphore is injected.
        self.llm_semaphore: asyncio.Semaphore = llm_semaphore or asyncio.Semaphore(10)

        # Initialize components
        self.classifier = QueryClassifier(self.llm_client, self.llm_semaphore)  # FIX-2

        self.router = None  # Will be initialized per execution if tool registry changes

        threshold = getattr(self.app_settings, "AGENT_SCORE_THRESHOLD", 0.7)
        enable_llm_eval = getattr(self.app_settings, "ENABLE_RETRIEVAL_EVALUATION", False)
        self.evaluator = RetrievalEvaluator(
            self.llm_client,
            score_threshold=threshold,
            llm_semaphore=self.llm_semaphore,
            enable_llm_eval=enable_llm_eval,
        )
        self.rewriter = QueryRewriter(self.llm_client, self.llm_semaphore)

    # -----------------------------------------------------------------
    # FIX-2: context manager that acquires the semaphore before any
    # LLM call — all agent components receive this helper via dependency.
    # -----------------------------------------------------------------
    @asynccontextmanager
    async def _llm_slot(self):
        """Acquire a semaphore slot before an LLM call; release after."""
        async with self.llm_semaphore:
            yield

    async def execute_agent(self, project: Any, session_id: str, query: str, chat_history: list = None, run_id: str = None) -> Dict[str, Any]:
        """
        Executes the agentic RAG workflow.
        """
        if not run_id:
            run_id = str(uuid.uuid4())
            # For standalone/fallback execution, create it here
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

            # Semantic cache check (only when cache_service is available)
            use_cache = getattr(self.app_settings, "ENABLE_SEMANTIC_CACHE", True) and self.cache_service is not None
            query_emb = None
            if use_cache and not chat_history:
                try:
                    # Need embedding for cache check
                    query_emb = await asyncio.to_thread(
                        self.memory_service.embedding_client.embed_text, 
                        query, 
                        "query"
                    )
                    cached_ans = await self.cache_service.check_semantic_cache(
                        project_id=project.project_id,
                        cache_vec=query_emb,
                        use_cache=True,
                        cache_threshold=getattr(self.app_settings, "CACHE_SIMILARITY_THRESHOLD", 0.95)
                    )
                    if cached_ans:
                        # Update run status to cached
                        await self.agent_run_model.update_agent_run(run_id, {"status": "completed"})
                        return {
                            "answer": cached_ans,
                            "sources": [],
                            "run_id": run_id,
                            "steps": 0,
                            "metadata": {"cached": True, "agent_path": True}
                        }
                except Exception as e:
                    logger.error(f"Semantic cache error in agent: {e}")

            # 1. Setup Tools for this specific project/session
            tool_registry = ToolRegistry()
            tool_registry.register(SearchDocumentsTool(self.retrieval_service, self.nlp_controller, project, self.llm_semaphore))
            tool_registry.register(GetProjectInfoTool(self.project_model, project.project_id))
            tool_registry.register(ListProjectAssetsTool(self.asset_model, project.project_id))
            tool_registry.register(GetConversationContextTool(self.memory_service, project.project_id, session_id))
            if getattr(self.app_settings, "ENABLE_WEB_SEARCH", True):
                tool_registry.register(WebSearchTool())

            # 2. Setup Router and Planner with this registry
            router = AgentRouter(self.llm_client, tool_registry, self.llm_semaphore)   # FIX-2
            planner = AgentPlanner(self.llm_client, tool_registry, self.llm_semaphore)  # FIX-2

            # 3. Create Graph
            graph = create_agent_graph(
                classifier=self.classifier,
                router=router,
                evaluator=self.evaluator,
                rewriter=self.rewriter,
                planner=planner,
                tool_registry=tool_registry,
                llm_client=self.llm_client,
                agent_mode=agent_mode,
                llm_semaphore=self.llm_semaphore,  # FIX-2: pass to answer_node
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

            # 5. Execute Graph — FIX-5: hard wall-clock timeout to prevent runaway graphs
            timeout_s = getattr(self.app_settings, "AGENT_TIMEOUT_SECONDS", 60)
            try:
                final_state = await asyncio.wait_for(
                    graph.ainvoke(initial_state),
                    timeout=timeout_s
                )
            except asyncio.TimeoutError:
                logger.error("Agent graph timed out after %ds for query: %s", timeout_s, query)
                await self.agent_run_model.update_agent_run(run_id, {
                    "status": "failed",
                    "error": f"Agent timed out after {timeout_s}s",
                    "finished_at": datetime.now(timezone.utc)
                })
                record_agent_run(agent_mode, "failed")
                return {"answer": "I'm sorry, the request took too long to process. Please try again.", "sources": [], "run_id": run_id, "steps": 0}

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



            # FIX-7: Save to semantic cache on success
            if use_cache and not chat_history and query_emb and final_state.get("final_answer") and update_data["status"] == "success":
                try:
                    await self.cache_service.set_semantic_cache(
                        project_id=project.project_id,
                        query=query,
                        cache_vec=query_emb,
                        answer=final_state.get("final_answer"),
                        use_cache=True
                    )
                except Exception as e:
                    logger.error("Failed to set semantic cache in agent [%s]: %s", type(e).__name__, e)

            return {
                "answer": final_state.get("final_answer"),
                "sources": final_state.get("sources", []),
                "run_id": run_id,
                "steps": final_state.get("step_count", 0)
            }

        except Exception as e:
            logger.error("Agent execution failed [%s]: %s", type(e).__name__, e, exc_info=True)
            await self.agent_run_model.update_agent_run(run_id, {
                "status": "failed",
                "error": str(e),
                "finished_at": datetime.now(timezone.utc)
            })
            record_agent_run("UNKNOWN", "failed")
            raise
