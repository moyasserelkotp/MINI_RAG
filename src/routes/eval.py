"""Evaluation API endpoints — Phase 8.

Simple pass-through to existing RAG evaluator utility functions,
exposed via REST API.
"""
from fastapi import APIRouter, Request, status, Depends, HTTPException
from pydantic import BaseModel, Field
from .schemes.nlp import EvaluationResponse

import logging
import asyncio

logger = logging.getLogger("uvicorn.error")
from middleware.rate_limiter import limiter

eval_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["Evaluation"],
)


class EvaluateRequest(BaseModel):
    query: str = Field(..., description="The user question to evaluate.")
    answer: str = Field(..., description="The generated answer from the RAG system.")
    contexts: list[str] = Field(..., description="The retrieved context chunks used to generate the answer.")


@eval_router.post("/eval/faithfulness", summary="Evaluate if answer is faithful to retrieved contexts", response_model=EvaluationResponse)
@limiter.limit("60/minute")
async def eval_faithfulness(request: Request, eval_req: EvaluateRequest):
    """
    Check if the answer is completely supported by the provided context chunks.
    (Hallucination check). Returns a score from 0.0 to 1.0 and a reasoning string.
    """
    gen_client = request.app.generation_client

    if not hasattr(gen_client, "generate_text"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"signal": "EVAL_UNSUPPORTED", "error": "LLM client not configured"}
        )

    # Note: In a production app you'd import the actual Evaluator class
    # For now, we perform a prompt-based evaluation right here.
    
    prompt = f"""You are an impartial evaluator assessing the faithfulness of an AI assistant's answer.
Given a question, the contexts retrieved from a database, and the assistant's answer, 
you must determine if the answer is completely supported by the contexts. 
If the answer contains any information NOT found in the contexts, it is unfaithful (a hallucination).

Question: {eval_req.query}
Contexts:
{' --- '.join(eval_req.contexts)}

Answer: {eval_req.answer}

Respond with exactly two lines:
Line 1: A single float score between 0.0 (completely unfaithful/hallucinated) and 1.0 (completely faithful).
Line 2: A brief 1-2 sentence explanation for the score.
"""
    
    try:
        eval_result = await asyncio.to_thread(gen_client.generate_text, prompt=prompt)
        lines = eval_result.strip().split('\n')
        score = 0.0
        reasoning = eval_result
        parse_error = False
        if len(lines) >= 1:
            try:
                # Extract first float found in line 1
                import re
                match = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", lines[0])
                if match:
                    score = float(match.group(1))
                else:
                    parse_error = True
            except Exception:
                parse_error = True
        else:
            parse_error = True
        if len(lines) >= 2:
            reasoning = " ".join(lines[1:])
            
        return EvaluationResponse(
            signal="EVAL_SUCCESS",
            metric="faithfulness",
            score=score,
            reasoning=reasoning.strip(),
            parse_error=parse_error
        )
    except Exception as e:
        logger.error(f"Faithfulness eval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"signal": "EVAL_FAILED", "error": str(e)}
        )


@eval_router.post("/eval/relevance", summary="Evaluate relevance of answer to query", response_model=EvaluationResponse)
@limiter.limit("60/minute")
async def eval_relevance(request: Request, eval_req: EvaluateRequest):
    """
    Check if the answer directly addresses the user's query.
    Returns a score from 0.0 to 1.0 and a reasoning string.
    """
    gen_client = request.app.generation_client

    if not hasattr(gen_client, "generate_text"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"signal": "EVAL_UNSUPPORTED", "error": "LLM client not configured"}
        )

    prompt = f"""You are an impartial evaluator assessing the relevance of an AI assistant's answer to a user's question.
You must determine if the answer directly and fully addresses the user's core question.
Ignore whether the answer is factually correct; focus ONLY on whether it is answering what was asked without dodging the question.

Question: {eval_req.query}
Answer: {eval_req.answer}

Respond with exactly two lines:
Line 1: A single float score between 0.0 (completely irrelevant or dodges the question) and 1.0 (completely relevant and direct).
Line 2: A brief 1-2 sentence explanation for the score.
"""
    
    try:
        eval_result = await asyncio.to_thread(gen_client.generate_text, prompt=prompt)
        lines = eval_result.strip().split('\n')
        score = 0.0
        reasoning = eval_result
        parse_error = False
        if len(lines) >= 1:
            try:
                import re
                match = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", lines[0])
                if match:
                    score = float(match.group(1))
                else:
                    parse_error = True
            except Exception:
                parse_error = True
        else:
            parse_error = True
        if len(lines) >= 2:
            reasoning = " ".join(lines[1:])
            
        return EvaluationResponse(
            signal="EVAL_SUCCESS",
            metric="relevance",
            score=score,
            reasoning=reasoning.strip(),
            parse_error=parse_error
        )
    except Exception as e:
        logger.error(f"Relevance eval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"signal": "EVAL_FAILED", "error": str(e)}
        )
