from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
# pyrefly: ignore [missing-import]
import cohere
import logging
import time
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


class CoHereProvider(LLMInterface):

    # Max texts per Cohere embed call (API limit: 96)
    _BATCH_LIMIT = 96

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

        self.client = cohere.Client(api_key=self.api_key)

        # Rate limiting: Cohere trial keys allow ~100 calls/min
        self._last_embed_time = 0.0
        self._min_embed_interval = 0.65  # seconds between single-call embeds
        self._rate_limit_lock = threading.Lock()

    @property
    def enums(self):
        return CoHereEnums

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str) -> str:
        if len(text) > self.default_input_max_characters:
            logger.warning("Text truncated from %d to %d characters.", len(text), self.default_input_max_characters)
        return text[: self.default_input_max_characters].strip()

    # Role mapping: normalise any incoming role string to CoHere's required format
    _ROLE_MAP = {
        "user": "USER",
        "USER": "USER",
        "assistant": "CHATBOT",
        "ASSISTANT": "CHATBOT",
        "chatbot": "CHATBOT",
        "CHATBOT": "CHATBOT",
        "system": "SYSTEM",
        "SYSTEM": "SYSTEM",
    }

    def _to_cohere_history(self, chat_history: list) -> tuple[str, list]:
        """
        Split chat_history into:
          - preamble: concatenated text of all SYSTEM messages (CoHere v1 parameter)
          - history:  list of {role, message} dicts for USER/CHATBOT turns
        """
        preamble_parts = []
        history = []
        for msg in chat_history:
            role_raw = msg.get("role", "user")
            cohere_role = self._ROLE_MAP.get(role_raw, "USER")
            # CoHere v1 /chat uses "message" key, not "text" or "content"
            text = msg.get("message") or msg.get("text") or msg.get("content") or ""
            if cohere_role == "SYSTEM":
                preamble_parts.append(text)
            else:
                history.append({"role": cohere_role, "message": text})
        return "\n\n".join(preamble_parts), history

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.client:
            logger.error("CoHere client was not set")
            return None
        if not self.generation_model_id:
            logger.error("Generation model for CoHere was not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        preamble, cohere_history = self._to_cohere_history(chat_history or [])

        try:
            kwargs = dict(
                model=self.generation_model_id,
                chat_history=cohere_history,
                message=self.process_text(prompt),
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            if preamble:
                kwargs["preamble"] = preamble

            response = self.client.chat(**kwargs)
            if not response or not response.text:
                logger.error("Empty response from CoHere generate")
                return None
            return response.text
        except Exception as e:
            logger.error("CoHere generate_text error: %s", e)
            return None

    # ── Single embed ─────────────────────────────────────────────────────────

    def _cohere_input_type(self, document_type: str):
        """Map DocumentTypeEnum string value to CoHere input_type string."""
        # Bug fix: compare .value string, not enum member
        if document_type == DocumentTypeEnum.QUERY.value:
            return CoHereEnums.QUERY.value
        return CoHereEnums.DOCUMENT.value

    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        if not self.client:
            logger.error("CoHere client was not set")
            return None
        if not self.embedding_model_id:
            logger.error("Embedding model for CoHere was not set")
            return None

        # Rate limiting
        with self._rate_limit_lock:
            elapsed = time.monotonic() - self._last_embed_time
            if elapsed < self._min_embed_interval:
                time.sleep(self._min_embed_interval - elapsed)
            self._last_embed_time = time.monotonic()

        input_type = self._cohere_input_type(document_type)

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                texts=[self.process_text(text)],
                input_type=input_type,
                embedding_types=["float"],
            )
            if not response or not response.embeddings or not response.embeddings.float:
                logger.error("Empty embedding response from CoHere")
                return None
            return response.embeddings.float[0]
        except Exception as e:
            logger.error("CoHere embed_text error: %s", e)
            return None

    # ── Batch embed (native, up to 96 texts per call) ────────────────────────

    def embed_batch(
        self, texts: List[str], document_type: str = None
    ) -> List[Optional[List[float]]]:
        if not self.client or not self.embedding_model_id:
            return [None] * len(texts)

        input_type = self._cohere_input_type(document_type)
        results: List[Optional[List[float]]] = []

        for i in range(0, len(texts), self._BATCH_LIMIT):
            batch = [self.process_text(t) for t in texts[i : i + self._BATCH_LIMIT]]

            # Mild rate-limiting between batch calls
            with self._rate_limit_lock:
                elapsed = time.monotonic() - self._last_embed_time
                if elapsed < self._min_embed_interval:
                    time.sleep(self._min_embed_interval - elapsed)
                self._last_embed_time = time.monotonic()

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.embed(
                        model=self.embedding_model_id,
                        texts=batch,
                        input_type=input_type,
                        embedding_types=["float"],
                    )
                    if response and response.embeddings and response.embeddings.float:
                        results.extend(response.embeddings.float)
                        break
                    else:
                        results.extend([None] * len(batch))
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            "CoHere embed_batch attempt %d failed: %s — retrying in %ss",
                            attempt + 1, e, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("CoHere embed_batch failed after %d retries: %s", max_retries, e)
                        results.extend([None] * len(batch))

        return results

    def construct_prompt(self, prompt: str, role: str):
        # Always store with the "message" key; role is normalised at send time
        cohere_role = self._ROLE_MAP.get(role, "USER")
        return {"role": cohere_role, "message": self.process_text(prompt)}
