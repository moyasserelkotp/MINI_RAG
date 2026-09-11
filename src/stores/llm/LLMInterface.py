from abc import ABC, abstractmethod
from typing import List, Optional


class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str):
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        pass

    def embed_batch(
        self, texts: List[str], document_type: str = None
    ) -> List[Optional[List[float]]]:
        """Embed multiple texts. Providers may override this for efficiency.

        Default implementation calls embed_text sequentially; subclasses that
        support native batch embedding (e.g. Cohere, OpenAI) should override.
        """
        return [self.embed_text(text=t, document_type=document_type) for t in texts]

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        pass

    def generate_structured_output(
        self,
        prompt: str,
        schema: dict,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ) -> dict | None:
        """
        Generate a structured JSON response conforming to `schema`.
        Default: prompt-based JSON extraction via generate_text().
        Providers can override for native structured output support.
        """
        import json
        import re
        
        # Inject schema instruction into prompt
        schema_str = json.dumps(schema, indent=2)
        json_prompt = f"{prompt}\n\nIMPORTANT: You must respond ONLY with a valid JSON object matching this schema. Do not include markdown code blocks or any other text.\nSchema:\n{schema_str}"
        
        response_text = self.generate_text(
            prompt=json_prompt,
            chat_history=chat_history,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
        
        if not response_text:
            return None
            
        try:
            # Try parsing directly
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                # Try extracting JSON from markdown blocks
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                # Fallback: try finding first '{' and last '}'
                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(response_text[start:end+1])
            except Exception:
                import logging
                logging.getLogger(__name__).warning("Failed to parse JSON from structured output")
                pass
        
        return None
