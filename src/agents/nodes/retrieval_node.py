from ..state import AgentState
from tools.registry import ToolRegistry

def get_retrieval_node(tool_registry: ToolRegistry):
    async def retrieval_node(state: AgentState):
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {"trace": [{"node": "retrieval_node", "event": "skipped", "reason": "No tool calls"}]}
            
        # Get the latest tool call
        latest_call = tool_calls[-1]
        tool_name = latest_call["name"]
        kwargs = latest_call.get("kwargs", {})
        
        trace_event = {
            "node": "retrieval_node",
            "event": "tool_execution",
            "tool": tool_name
        }
        
        try:
            tool = tool_registry.get_tool(tool_name)
            result = await tool.execute(**kwargs)
            
            # If search or web search, update retrieved context
            if tool_name in ["SEARCH_DOCUMENTS", "WEB_SEARCH"] and result.get("success"):
                new_context = result.get("results", [])
                
                # If WEB_SEARCH, format the dict into readable text
                if tool_name == "WEB_SEARCH":
                    formatted_context = []
                    for item in new_context:
                        formatted_context.append({
                            "text": f"Title: {item.get('title')}\nURL: {item.get('url')}\nSnippet: {item.get('snippet')}",
                            "score": 1.0,
                            "metadata": {"source": "WEB_SEARCH", "url": item.get('url')}
                        })
                    new_context = formatted_context

                trace_event["status"] = "success"
                trace_event["chunks_retrieved"] = len(new_context)
                
                return {
                    "retrieved_context": state.get("retrieved_context", []) + new_context,
                    "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
                    "step_count": state.get("step_count", 0) + 1,
                    "trace": [trace_event]
                }
            elif result.get("success"):
                # Other tools (Project, Asset, Memory)
                # Treat their results as context for the LLM
                context_item = {
                    "text": str(result),
                    "score": 1.0,
                    "metadata": {"source": tool_name}
                }
                trace_event["status"] = "success"
                return {
                    "retrieved_context": state.get("retrieved_context", []) + [context_item],
                    "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,  # Also increment attempts here to prevent infinite loops
                    "step_count": state.get("step_count", 0) + 1,
                    "trace": [trace_event]
                }
            else:
                trace_event["status"] = "failed"
                trace_event["error"] = result.get("error", "Unknown error")
                return {
                    "errors": [f"Tool {tool_name} failed: {result.get('error')}"],
                    "step_count": state.get("step_count", 0) + 1,
                    "trace": [trace_event]
                }
                
        except Exception as e:
            trace_event["status"] = "error"
            trace_event["error"] = str(e)
            return {
                "errors": [f"Tool execution error: {e}"],
                "step_count": state.get("step_count", 0) + 1,
                "trace": [trace_event]
            }
            
    return retrieval_node
