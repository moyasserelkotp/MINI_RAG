from ..LLMInterface import LLMInterface
import google.generativeai as genai
import logging


class GeminiProvider(LLMInterface):

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

        # Initialize Gemini client
        try:
            genai.configure(api_key=api_key)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Gemini client configured successfully")
        except Exception as e:
            self.logger = logging.getLogger(__name__)
            self.logger.error(f"Failed to configure Gemini client: {e}")

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Set Gemini generation model: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Set Gemini embedding model: {model_id}")

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
            self.logger.error("Generation model for Gemini was not set")
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens
            else self.default_generation_max_output_tokens
        )
        temperature = (
            temperature if temperature else self.default_generation_temperature
        )

        try:
            model = genai.GenerativeModel(self.generation_model_id)

            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )

            response = model.generate_content(
                self.process_text(prompt),
                generation_config=generation_config
            )

            if not response or not response.text:
                self.logger.error("Error while generating text with Gemini")
                return None

            return response.text

        except Exception as e:
            self.logger.error(f"Error while generating text with Gemini: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        try:
            response = genai.embed_content(
                model=self.embedding_model_id,
                content=self.process_text(text),
            )

            if not response or not response.get("embedding"):
                self.logger.error("Error while embedding text with Gemini")
                return None

            return response["embedding"]

        except Exception as e:
            self.logger.error(f"Error while embedding text with Gemini: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt
        }
