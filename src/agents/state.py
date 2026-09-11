from typing import TypedDict, List, Dict, Any, Optional
import operator
from typing import Annotated

class AgentState(TypedDict):
    """
    State representing the agent's memory throughout the LangGraph workflow.
    """
    run_id: str
    project_id: str
    session_id: Optional[str]

    # Query progression
    original_query: str
    current_query: str
    query_category: Optional[str]

    # Chat History
    messages: List[Dict[str, str]]

    # Planning
    plan: List[Dict[str, Any]]
    current_plan_step: int        # Index into `plan`; advanced by planner/retrieval nodes
    next_action: Optional[str]    # Planner's intended next tool action

    # Tool execution
    selected_tool: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]

    # Retrieval
    retrieved_context: List[Dict[str, Any]]
    retrieval_attempts: int
    max_retrieval_attempts: int

    # Evaluation
    evaluation_result: Optional[Dict[str, Any]]
    rewrite_count: int            # How many query rewrites have been attempted

    # Output
    final_answer: Optional[str]
    sources: List[Dict[str, Any]]
    errors: Annotated[List[str], operator.add]

    # Loop tracking
    step_count: int
    max_steps: int

    # Execution Trace for MongoDB
    trace: Annotated[List[Dict[str, Any]], operator.add]

