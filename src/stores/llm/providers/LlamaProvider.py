from ..LLMInterface import LLMInterface
from ..LLMEnums import GenericLLMEnums
import ollama
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

def _get_default_ollama_host() -> str:
    """Intelligently detects if running in WSL to route localhost to Windows."""
    # 1. Respect explicitly set environment variable if user added one
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        return env_host

    # 2. Detect WSL Linux subsystem
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                with open("/etc/resolv.conf", "r") as r:
                    for line in r:
                        if line.startswith("nameserver"):
                            ip = line.split()[1].strip()
                            logger.info(f"Detected WSL environment. Routing Ollama host to {ip}")
                            return f"http://{ip}:11434"
    except Exception:
        pass

    # 3. Default local IPv4 (better than 'localhost' on Windows for httpx to avoid IPv6 issues)
    return "http://127.0.0.1:11434"



class LlamaProvider(LLMInterface):

    def __init__(
        self,
        api_key: str = None,
        api_url: str = "http://localhost:11434/",
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
            # Dynamically determine the best host string and pass it to Client
            host_url = _get_default_ollama_host()
            self.client = ollama.Client(host=host_url)
        except Exception as e:
            msg = "Llama client was not initialized"
            logger.error(f"{msg}: {e}")
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
            msg = "Generation model for Llama was not set"
            logger.error(msg)
            return msg
        if not self.client:
            msg = "Llama client was not initialized"
            logger.error(msg)
            return msg

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
                msg = "Empty response from Llama generate"
                logger.error(msg)
                return msg
            return text_out
        except Exception as e:
            msg = f"Llama generate_text error: {str(e)}"
            logger.error(msg)
            return msg

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
