from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser):
        super().__init__()
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    # ── Helpers ──────────────────────────────────────────────────────────────

    def create_collection_name(self, project_id: str) -> str:
        prefix = getattr(self.app_settings, "VECTOR_DB_COLLECTION_PREFIX", "collection")
        return f"{prefix}_{project_id}".strip()

    # ── Collection management ─────────────────────────────────────────────────

    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _embed_texts_batch(self, texts: List[str], document_type: str) -> List[list]:
        """Embed a list of texts using batch API calls where available.

        Falls back to one-by-one embedding if `embed_batch` is not supported.
        Returns a list of embedding vectors (same order as input). Texts that
        fail to embed get ``None`` in the output list.
        """
        batch_size = self.app_settings.MAX_EMBEDDING_BATCH_SIZE

        # Use batch method if available (CoHere, OpenAI)
        if hasattr(self.embedding_client, "embed_batch"):
            vectors: List[Optional[list]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_vectors = self.embedding_client.embed_batch(
                    texts=batch, document_type=document_type
                )
                if batch_vectors:
                    vectors.extend(batch_vectors)
                else:
                    # Partial failure — fill with None
                    vectors.extend([None] * len(batch))
            return vectors

        # Fallback: sequential
        return [
            self.embedding_client.embed_text(text=t, document_type=document_type)
            for t in texts
        ]

    def index_into_vector_db(
        self,
        project: Project,
        chunks: List[DataChunk],
        chunks_ids: List[int],
        do_reset: bool = False,
    ) -> bool:
        collection_name = self.create_collection_name(project_id=project.project_id)

        # Texts and metadata
        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]

        # Batch-embed all texts at once
        vectors = self._embed_texts_batch(texts, document_type=DocumentTypeEnum.DOCUMENT.value)

        # Filter out any failed embeddings
        valid = [
            (t, v, m, rid)
            for t, v, m, rid in zip(texts, vectors, metadata, chunks_ids)
            if v is not None
        ]
        if not valid:
            logger.error("All embeddings failed for project %s", project.project_id)
            return False

        f_texts, f_vectors, f_metadata, f_ids = zip(*valid)

        # Create collection (do_reset handled here)
        self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # Batch insert
        self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=list(f_texts),
            metadata=list(f_metadata),
            vectors=list(f_vectors),
            record_ids=list(f_ids),
        )

        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search_vector_db_collection(
        self,
        project: Project,
        text: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
    ):
        """Search the vector DB collection.

        Returns:
            list  — results (may be empty if threshold filtered everything out)
            None  — infrastructure failure (embed failed or search threw an error)
        """
        collection_name = self.create_collection_name(project_id=project.project_id)

        # Embed the query
        vector = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.QUERY.value
        )
        if not vector or len(vector) == 0:
            logger.error("Failed to embed query: %s", text)
            return None  # real failure — caller should surface as error

        # Resolve score threshold (None or 0.0 means no filtering)
        threshold = score_threshold
        if threshold is None:
            threshold = getattr(self.app_settings, "SEARCH_SCORE_THRESHOLD", 0.0) or 0.0

        # Perform vector search
        try:
            if use_hybrid:
                results = self.vectordb_client.hybrid_search(
                    collection_name=collection_name,
                    query_text=text,
                    vector=vector,
                    limit=limit,
                    semantic_weight=0.6,
                )
            else:
                # Pass threshold to Qdrant directly when doing pure semantic search
                results = self.vectordb_client.search_by_vector(
                    collection_name=collection_name,
                    vector=vector,
                    limit=limit,
                    score_threshold=threshold if threshold > 0 else None,
                )
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return None  # real failure

        # results may be None or empty if the collection is empty
        if results is None:
            return []

        # Apply threshold post-filter for hybrid mode
        # (pure semantic already has it applied by Qdrant)
        if use_hybrid and threshold and threshold > 0:
            results = [r for r in results if getattr(r, "score", 0.0) >= threshold]

        # Always return a list — empty list means "no results above threshold",
        # which is NOT an error; callers must check for None to detect failures.
        return list(results)

    # ── Answer ────────────────────────────────────────────────────────────────

    def answer_rag_question(
        self,
        project: Project,
        query: str,
        limit: int = 10,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
    ):
        answer, full_prompt, chat_history = None, None, None

        # Step 1: Retrieve relevant documents
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
            use_hybrid=use_hybrid,
            score_threshold=score_threshold,
        )

        # None = infrastructure failure; [] = no results (threshold too strict or empty index)
        if retrieved_documents is None:
            logger.error("Search failed (infrastructure error) for query: %s", query)
            return False, None, None  # False denotes infrastructure error

        if len(retrieved_documents) == 0:
            logger.warning("No documents above threshold for query: %s", query)
            return None, None, None  # None denotes 0 documents found

        # Step 2: Build document prompts with source metadata
        system_prompt = self.template_parser.get("rag", "system_prompt")

        document_parts = []
        for idx, doc in enumerate(retrieved_documents):
            chunk_text = doc.payload.get("text", "")
            doc_metadata = doc.payload.get("metadata", {})
            source = doc_metadata.get("source", "unknown") if doc_metadata else "unknown"
            score = round(getattr(doc, "score", 0.0), 4)

            doc_prompt = self.template_parser.get(
                "rag",
                "document_prompt",
                {
                    "doc_num": idx + 1,
                    "chunk_text": chunk_text,
                    "source": source,
                    "score": score,
                },
            )
            document_parts.append(doc_prompt)

        documents_prompts = "\n\n".join(document_parts)
        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})

        # Step 3: Construct chat history
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]
        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        # Step 4: Generate answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt, chat_history=chat_history
        )

        return answer, full_prompt, chat_history
