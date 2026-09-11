from ..state import AgentState
from ..planner import AgentPlanner

def get_planner_node(planner: AgentPlanner):
    async def planner_node(state: AgentState):
        query = state["current_query"]
        history = state.get("messages", [])
        
        plan = await planner.create_plan(query, history)
        
        trace_event = {
            "node": "planner_node",
            "event": "plan_created",
            "plan": plan
        }
        
        # Take the first tool action from the plan if available
        tool_calls = []
        if plan and len(plan) > 0:
            first_step = plan[0]
            if first_step.get("action") != "DIRECT_ANSWER":
                tool_calls.append({
                    "id": f"plan_step_{first_step.get('id', 1)}",
                    "name": first_step.get("action"),
                    "kwargs": first_step.get("tool_kwargs", {})
                })
        
        return {
            "plan": plan,
            "tool_calls": tool_calls,
            "step_count": state.get("step_count", 0) + 1,
            "trace": [trace_event]
        }
        
    return planner_node
