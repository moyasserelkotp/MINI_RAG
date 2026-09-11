from ..state import AgentState


def get_memory_node():
    """
    Returns a pass-through memory node.

    This node is intentionally NOT wired into the main graph.
    Memory is handled in two other ways:
      1. Chat history is pre-loaded by AgentController and passed in the
         initial state under `messages`.
      2. Post-run memory persistence is handled by AgentService after
         graph.ainvoke() completes (calls memory_service.save_messages()).

    This function is kept for optional future use as an explicit memory-load
    entry node (e.g., if vector memory retrieval is added at graph start).
    """
    async def memory_node(state: AgentState):
        trace_event = {
            "node": "memory_node",
            "event": "processed"
        }
        return {
            "trace": [trace_event]
        }
    return memory_node
