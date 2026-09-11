from ..state import AgentState
from ..evaluator import RetrievalEvaluator
from ..query_rewriter import QueryRewriter

def get_evaluation_node(evaluator: RetrievalEvaluator, rewriter: QueryRewriter):
    async def evaluation_node(state: AgentState):
        query = state["current_query"]
        query_category = state.get("query_category")
        context = state.get("retrieved_context", [])
        
        # 1. Evaluate context
        eval_result = await evaluator.evaluate(query, context, query_category=query_category)
        
        trace_event = {
            "node": "evaluation_node",
            "event": "evaluated",
            "is_sufficient": eval_result["is_sufficient"],
            "reason": eval_result["reason"]
        }
        
        if eval_result["is_sufficient"]:
            return {
                "evaluation_result": eval_result,
                "step_count": state.get("step_count", 0) + 1,
                "trace": [trace_event]
            }
            
        # 2. If not sufficient, check if we can retry
        attempts = state.get("retrieval_attempts", 0)
        max_attempts = state.get("max_retrieval_attempts", 3)
        
        if attempts >= max_attempts:
            trace_event["action"] = "give_up"
            return {
                "evaluation_result": eval_result,
                "step_count": state.get("step_count", 0) + 1,
                "trace": [trace_event]
            }
            
        # 3. Rewrite query for next attempt
        history = state.get("messages", [])
        new_query = await rewriter.rewrite(query, history)
        
        trace_event["action"] = "rewrite"
        trace_event["new_query"] = new_query
        
        return {
            "evaluation_result": eval_result,
            "current_query": new_query,
            "step_count": state.get("step_count", 0) + 1,
            "trace": [trace_event]
        }
        
    return evaluation_node
