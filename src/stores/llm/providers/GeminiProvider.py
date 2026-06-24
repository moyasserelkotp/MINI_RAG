from ..LLMInterface import LLMInterface
from ..LLMEnums import GenericLLMEnums, DocumentTypeEnum
import google.generativeai as genai
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GeminiProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1024,
        default_generation_max_output_tokens: int = 512,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # Cached model instances — avoid re-instantiation on every call
        self._generation_model: Optional[genai.GenerativeModel] = None

        try:
            genai.configure(api_key=api_key)
            logger.info("Gemini client configured successfully")
        except Exception as e:
            logger.error("Failed to configure Gemini client: %s", e)

    @property
    def enums(self):
        return GenericLLMEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        # Cache the GenerativeModel so it is not re-created on every request
        self._generation_model = genai.GenerativeModel(model_id)
        logger.info("Set Gemini generation model: %s", model_id)

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        logger.info("Set Gemini embedding model: %s", model_id)

    def process_text(self, text: str) -> str:
        """Truncate text to the configured limit (used for generation prompts only)."""
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,   # FIX: was `= []` (mutable default)
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.generation_model_id or self._generation_model is None:
            logger.error("Generation model for Gemini was not set")
            return None

        chat_history = chat_history or []  # FIX: safe default
        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )


            system_text_parts = []
            gemini_history = []

            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_text_parts.append(content)
                elif role == "assistant":
                    gemini_history.append({"role": "model", "parts": [content]})
                else:
                    gemini_history.append({"role": "user", "parts": [content]})

            # If we have a system prompt, recreate the model with it injected.
            # Otherwise use the cached instance to avoid overhead.
            if system_text_parts:
                system_instruction = "\n\n".join(system_text_parts)
                model = genai.GenerativeModel(
                    self.generation_model_id,
                    system_instruction=system_instruction,
                )
            else:
                model = self._generation_model

            # Use start_chat so history is correctly threaded through the call
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(
                self.process_text(prompt),
                generation_config=generation_config,
            )

            if not response or not response.text:
                logger.error("Empty response from Gemini generate")
                return None
            return response.text
        except Exception as e:
            logger.error("Gemini generate_text error: %s", e)
            return None

    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        if not self.embedding_model_id:
            logger.error("Embedding model for Gemini was not set")
            return None

        # Map document_type to Gemini task type
        task_type = "retrieval_document"
        if document_type == DocumentTypeEnum.QUERY.value:
            task_type = "retrieval_query"

        try:
            response = genai.embed_content(
                model=self.embedding_model_id,
                content=text,
                task_type=task_type,
            )
            if not response or not response.get("embedding"):
                logger.error("Empty embedding response from Gemini")
                return None
            return response["embedding"]
        except Exception as e:
            logger.error("Gemini embed_text error: %s", e)
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
