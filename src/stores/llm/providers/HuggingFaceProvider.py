from ..LLMInterface import LLMInterface
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import logging


class HuggingFaceProvider(LLMInterface):

    def __init__(
        self,
        api_key: str = None,
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

        self.generation_pipeline = None
        self.embedding_model = None
        self.embedding_tokenizer = None

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        try:
            self.generation_pipeline = pipeline(
                "text-generation",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info(f"Loaded HuggingFace generation model: {model_id}")
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace generation model: {e}")
            self.generation_pipeline = None

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.embedding_model = AutoModel.from_pretrained(
                model_id,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.embedding_model.eval()
            self.logger.info(f"Loaded HuggingFace embedding model: {model_id}")
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace embedding model: {e}")
            self.embedding_model = None

    def process_text(self, text: str):
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):

        if not self.generation_pipeline:
            self.logger.error("HuggingFace generation model was not set")
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
            response = self.generation_pipeline(
                self.process_text(prompt),
                max_length=max_output_tokens,
                temperature=temperature,
                do_sample=True,
            )

            if not response or len(response) == 0:
                self.logger.error("Error while generating text with HuggingFace")
                return None

            return response[0]["generated_text"]

        except Exception as e:
            self.logger.error(f"Error while generating text with HuggingFace: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):

        if not self.embedding_model:
            self.logger.error("HuggingFace embedding model was not set")
            return None

        try:
            inputs = self.embedding_tokenizer(
                self.process_text(text),
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy().tolist()[0]

        except Exception as e:
            self.logger.error(f"Error while embedding text with HuggingFace: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt
        }
