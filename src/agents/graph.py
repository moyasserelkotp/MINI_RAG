# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    get_router_node,
    get_retrieval_node,
    get_evaluation_node,
    get_answer_node,
    get_planner_node
)
import logging

logger = logging.getLogger(__name__)

def create_agent_graph(
    classifier,
    router,
    evaluator,
    rewriter,
    planner,
    tool_registry,
    llm_client,
    agent_mode="FULL_AGENT",
    llm_semaphore=None,  # FIX-2
):
    """Creates and compiles the LangGraph state machine."""
    
    # Initialize Graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("router", get_router_node(classifier, router))
    workflow.add_node("planner", get_planner_node(planner))
    workflow.add_node("retrieval", get_retrieval_node(tool_registry))
    workflow.add_node("evaluation", get_evaluation_node(evaluator, rewriter))
    workflow.add_node("answer", get_answer_node(llm_client, llm_semaphore))  # FIX-2
    
    # ------------------------------------------------------------------
    # MEDIUM-3: renamed from route_after_router — not a graph edge directly.
    # ------------------------------------------------------------------
    def _compute_next_node(state: AgentState) -> str:
        """Internal helper — do NOT register as a graph edge directly."""
        if agent_mode == "FULL_AGENT" and state.get("query_category") == "COMPLEX_MULTI_STEP":
            if not state.get("plan"):
                return "planner"

        if state.get("selected_tool") == "DIRECT_ANSWER":
            return "answer"
        return "retrieval"

    def route_after_evaluation(state: AgentState) -> str:
        eval_result = state.get("evaluation_result", {})
        if eval_result.get("is_sufficient"):
            return "answer"

        attempts = state.get("retrieval_attempts", 0)
        max_attempts = state.get("max_retrieval_attempts", 3)
        if attempts >= max_attempts:
            return "answer"  # Give up and try to answer with what we have

        return "router"  # Loop back with rewritten query

    def route_after_step_limit(state: AgentState) -> str:
        """HIGH-1 FIX: universal step guard — hard-stops the graph at max_steps.
        Called from both the router edge AND the evaluation edge to ensure the
        limit is never bypassed regardless of which nodes increment step_count."""
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 5)
        if step_count >= max_steps:
            logger.warning(
                "Max steps %d reached (current=%d) for query '%s' — forcing answer.",
                max_steps, step_count, state.get("original_query", "")
            )
            return "answer"
        return _compute_next_node(state)

    # Add Edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_after_step_limit,
        {
            "answer": "answer",
            "retrieval": "retrieval",
            "planner": "planner"
        }
    )
    
    workflow.add_edge("planner", "retrieval")
    if agent_mode == "ROUTER":
        workflow.add_edge("retrieval", "answer")
    else:
        workflow.add_edge("retrieval", "evaluation")
    
    workflow.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {
            "answer": "answer",
            "router": "router"
        }
    )
    
    workflow.add_edge("answer", END)
    
    # Compile
    return workflow.compile()
