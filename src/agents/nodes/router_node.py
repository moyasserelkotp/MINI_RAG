from ..state import AgentState
from ..classifier import QueryClassifier
from ..router import AgentRouter

def get_router_node(classifier: QueryClassifier, router: AgentRouter):
    async def router_node(state: AgentState):
        query = state["current_query"]
        history = state.get("messages", [])
        
        # 1. Classify
        category = await classifier.classify(query)
        
        # 2. Route
        decision = await router.route(query, category, history)
        
        # 3. Update state
        action = decision.get("action")
        reason = decision.get("reason", "")
        tool_kwargs = decision.get("tool_kwargs", {})
        
        # Log to trace
        trace_event = {
            "node": "router_node",
            "event": "routed",
            "category": category,
            "decision": action,
            "reason": reason
        }
        
        if action == "DIRECT_ANSWER":
            return {
                "query_category": category,
                "selected_tool": "DIRECT_ANSWER",
                "tool_calls": [],
                "step_count": state.get("step_count", 0) + 1,
                "trace": [trace_event]
            }
            
        # Format tool call
        tool_call = {
            "id": f"call_{state.get('step_count', 0)}",
            "name": action,
            "kwargs": tool_kwargs
        }
        
        return {
            "query_category": category,
            "selected_tool": action,
            "tool_calls": [tool_call],
            "step_count": state.get("step_count", 0) + 1,
            "trace": [trace_event]
        }
        
    return router_node
