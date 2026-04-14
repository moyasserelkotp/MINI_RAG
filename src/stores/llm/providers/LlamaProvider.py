from ..LLMInterface import LLMInterface
from ..LLMEnums import GenericLLMEnums
import ollama
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class LlamaProvider(LLMInterface):

    def __init__(
        self,
        api_key: str = None,
        api_url: str = "http://localhost:11434",
        default_input_max_characters: int = 1024,
        default_generation_max_output_tokens: int = 512,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        try:
            self.client = ollama.Client(host=api_url) if api_url else ollama.Client()
        except Exception as e:
            logger.error("Failed to initialize Ollama client: %s", e)
            self.client = None

    @property
    def enums(self):
        return GenericLLMEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str) -> str:
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.generation_model_id:
            logger.error("Generation model for Llama was not set")
            return None
        if not self.client:
            logger.error("Llama client was not initialized")
            return None

        temperature = temperature or self.default_generation_temperature
        num_predict = max_output_tokens or self.default_generation_max_output_tokens

        try:
            response = self.client.generate(
                model=self.generation_model_id,
                prompt=self.process_text(prompt),
                options={
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
                stream=False,
            )
            # Bug fix: Ollama returns a dataclass object, not a dict
            # Use attribute access; fall back gracefully
            text_out = getattr(response, "response", None)
            if not text_out and isinstance(response, dict):
                text_out = response.get("response")
            if not text_out:
                logger.error("Empty response from Llama generate")
                return None
            return text_out
        except Exception as e:
            logger.error("Llama generate_text error: %s", e)
            return None

    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        if not self.embedding_model_id:
            logger.error("Embedding model for Llama was not set")
            return None
        if not self.client:
            logger.error("Llama client was not initialized")
            return None

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                input=self.process_text(text),
            )
            # Ollama returns dataclass with .embeddings attribute (list of lists)
            embeddings = getattr(response, "embeddings", None)
            if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
                return embeddings[0]
            # Legacy fallback: dict
            if isinstance(response, dict):
                return response.get("embedding") or response.get("embeddings", [None])[0]
            logger.error("Could not parse Llama embed response")
            return None
        except Exception as e:
            logger.error("Llama embed_text error: %s", e)
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
