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
    agent_mode="FULL_AGENT"
):
    """Creates and compiles the LangGraph state machine."""
    
    # Initialize Graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("router", get_router_node(classifier, router))
    workflow.add_node("planner", get_planner_node(planner))
    workflow.add_node("retrieval", get_retrieval_node(tool_registry))
    workflow.add_node("evaluation", get_evaluation_node(evaluator, rewriter))
    workflow.add_node("answer", get_answer_node(llm_client))
    
    # Conditional Edges
    def route_after_router(state: AgentState):
        if agent_mode == "FULL_AGENT" and state.get("query_category") == "COMPLEX_MULTI_STEP":
            if not state.get("plan"):
                return "planner"
                
        if state.get("selected_tool") == "DIRECT_ANSWER":
            return "answer"
        return "retrieval"
        
    def route_after_evaluation(state: AgentState):
        eval_result = state.get("evaluation_result", {})
        if eval_result.get("is_sufficient"):
            return "answer"
            
        attempts = state.get("retrieval_attempts", 0)
        max_attempts = state.get("max_retrieval_attempts", 3)
        if attempts >= max_attempts:
            return "answer"  # Give up and try to answer with what we have
            
        return "router" # Loop back with rewritten query
        
    def route_after_step_limit(state: AgentState):
        """Prevent infinite loops."""
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 5)
        if step_count >= max_steps:
            logger.warning(f"Max steps {max_steps} reached for query {state.get('original_query')}")
            return "answer"
        return route_after_router(state)

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
