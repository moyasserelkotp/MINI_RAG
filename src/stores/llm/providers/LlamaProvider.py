from ..LLMInterface import LLMInterface
from ..LLMEnums import GenericLLMEnums
import ollama
import logging


class LlamaProvider(LLMInterface):

    def __init__(
        self,
        api_key: str = None,
        api_url: str = "http://localhost:11434",
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
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

        # Initialize Ollama client
        self.client = ollama.Client(host=api_url) if api_url else ollama.Client()

        self.logger = logging.getLogger(__name__)

    @property
    def enums(self):
        return GenericLLMEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Set Llama generation model: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Set Llama embedding model: {model_id}")

    def process_text(self, text: str):
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):

        if not self.generation_model_id:
            self.logger.error("Generation model for Llama was not set")
            return None

        if not self.client:
            self.logger.error("Llama client was not initialized")
            return None

        temperature = (
            temperature if temperature else self.default_generation_temperature
        )

        try:
            response = self.client.generate(
                model=self.generation_model_id,
                prompt=self.process_text(prompt),
                temperature=temperature,
                num_predict=max_output_tokens or self.default_generation_max_output_tokens,
                stream=False
            )

            if not response or not response.get("response"):
                self.logger.error("Error while generating text with Llama")
                return None

            return response["response"]

        except Exception as e:
            self.logger.error(f"Error while generating text with Llama: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Llama was not set")
            return None

        if not self.client:
            self.logger.error("Llama client was not initialized")
            return None

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                input=self.process_text(text)
            )

            if not response or not response.get("embedding"):
                self.logger.error("Error while embedding text with Llama")
                return None

            return response["embedding"]

        except Exception as e:
            self.logger.error(f"Error while embedding text with Llama: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt
        }
