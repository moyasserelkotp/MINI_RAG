import logging
from typing import Optional

logger = logging.getLogger(__name__)

class QueryClassifier:
    """Classifies user queries into specific action categories."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # Fast deterministic heuristics
        self.project_keywords = ["project", "metadata", "created", "name of this project"]
        self.asset_keywords = ["files", "documents uploaded", "assets", "what files"]
        self.complex_keywords = ["compare", "analyze across", "differences between", "similarities", "vs"]
        
    async def classify(self, query: str) -> str:
        """
        Returns one of:
        - DOCUMENT_QUESTION
        - PROJECT_METADATA
        - ASSET_METADATA
        - CONVERSATION_REFERENCE
        - COMPLEX_MULTI_STEP
        - GENERAL_CONVERSATION
        """
        query_lower = query.lower()
        
        # 1. Fast heuristics
        if any(kw in query_lower for kw in self.complex_keywords):
            return "COMPLEX_MULTI_STEP"
            
        if any(kw in query_lower for kw in self.asset_keywords):
            return "ASSET_METADATA"
            
        if any(kw in query_lower for kw in self.project_keywords):
            return "PROJECT_METADATA"
            
        # 2. LLM Classification (if enabled)
        if self.llm_client:
            schema = {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "DOCUMENT_QUESTION", 
                            "PROJECT_METADATA", 
                            "ASSET_METADATA",
                            "CONVERSATION_REFERENCE",
                            "COMPLEX_MULTI_STEP",
                            "GENERAL_CONVERSATION"
                        ]
                    }
                },
                "required": ["category"]
            }
            
            prompt = f"""Classify the following user query into exactly one of the provided categories.
Query: "{query}"

Categories:
- DOCUMENT_QUESTION: Asking for facts, summaries, or details that would be found in uploaded documents.
- PROJECT_METADATA: Asking about the project itself (e.g., when was it created).
- ASSET_METADATA: Asking about what files/documents are uploaded.
- CONVERSATION_REFERENCE: Referring to something said earlier in the chat (e.g., "what did you just say?", "tell me more about that").
- COMPLEX_MULTI_STEP: Queries requiring comparison across multiple documents or complex multi-step reasoning.
- GENERAL_CONVERSATION: Greetings, generic statements (e.g., "hello", "thanks").
"""
            try:
                # Wrap in asyncio.to_thread if the llm_client is synchronous
                import asyncio
                result = await asyncio.to_thread(
                    self.llm_client.generate_structured_output,
                    prompt=prompt,
                    schema=schema
                )
                if result and "category" in result:
                    return result["category"]
            except Exception as e:
                logger.error(f"QueryClassifier LLM error: {e}")
                
        # Default fallback
        return "DOCUMENT_QUESTION"
