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
