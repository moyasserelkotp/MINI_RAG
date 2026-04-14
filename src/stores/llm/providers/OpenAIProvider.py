from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        api_url: str = None,
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
            client_kwargs = {"api_key": self.api_key}
            if self.api_url:
                client_kwargs["base_url"] = self.api_url
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            logger.error("Failed to initialize OpenAI client: %s", e)
            self.client = None

    @property
    def enums(self):
        return OpenAIEnums

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
        if not self.client:
            logger.error("OpenAI client was not set")
            return None
        if not self.generation_model_id:
            logger.error("Generation model for OpenAI was not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        messages = list(chat_history)
        messages.append(self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
            if not response or not response.choices or not response.choices[0].message:
                logger.error("Empty response from OpenAI generate")
                return None
            # Bug fix: was response.choices[0].message["content"] (dict access on object)
            return response.choices[0].message.content
        except Exception as e:
            logger.error("OpenAI generate_text error: %s", e)
            return None

    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        if not self.client:
            logger.error("OpenAI client was not set")
            return None
        if not self.embedding_model_id:
            logger.error("Embedding model for OpenAI was not set")
            return None

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=self.process_text(text),
            )
            if not response or not response.data or not response.data[0].embedding:
                logger.error("Empty embedding response from OpenAI")
                return None
            return response.data[0].embedding
        except Exception as e:
            logger.error("OpenAI embed_text error: %s", e)
            return None

    def embed_batch(
        self, texts: List[str], document_type: str = None
    ) -> List[Optional[List[float]]]:
        """OpenAI supports array input for embeddings — single API call for all texts."""
        if not self.client or not self.embedding_model_id:
            return [None] * len(texts)

        cleaned = [self.process_text(t) for t in texts]
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=cleaned,
            )
            if not response or not response.data:
                return [None] * len(texts)
            # API returns embeddings in the same order
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error("OpenAI embed_batch error: %s", e)
            return [None] * len(texts)

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
