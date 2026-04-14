from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging
import time


class CoHereProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):

        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)

        # Rate limiting for API calls (100 calls/minute = 1 call every 0.6 seconds)
        self.last_embed_time = 0
        self.min_embed_interval = 0.65  # seconds between embed calls

    @property
    def enums(self):
        return CoHereEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens
            else self.default_generation_max_output_tokens
        )
        temperature = (
            temperature if temperature else self.default_generation_temperature
        )

        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        if not response or not response.text:
            self.logger.error("Error while generating text with CoHere")
            return None

        return response.text

    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None

        # Rate limiting: ensure minimum interval between API calls
        elapsed = time.time() - self.last_embed_time
        if elapsed < self.min_embed_interval:
            time.sleep(self.min_embed_interval - elapsed)

        self.last_embed_time = time.time()

        input_type = CoHereEnums.DOCUMENT
        if document_type == DocumentTypeEnum.QUERY:
            input_type = CoHereEnums.QUERY

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                texts=[self.process_text(text)],
                input_type=input_type,
                embedding_types=["float"],
            )

            if not response or not response.embeddings or not response.embeddings.float:
                self.logger.error("Error while embedding text with CoHere")
                return None

            return response.embeddings.float[0]
        except Exception as e:
            self.logger.error(f"Error while embedding text with CoHere: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "text": self.process_text(prompt)}
