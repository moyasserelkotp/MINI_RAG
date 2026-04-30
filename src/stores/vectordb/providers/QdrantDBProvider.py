from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List, Optional
import hashlib
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class QdrantDBProvider(VectorDBInterface):

    def __init__(
        self, db_path: str = None, db_url: str = None, distance_method: str = None
    ):
        self.client: Optional[QdrantClient] = None
        self.db_path = db_path
        self.db_url = db_url
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT
        else:
            # Default to cosine if unrecognised
            self.distance_method = models.Distance.COSINE

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        try:
            if self.db_url:
                # Connect to remote Qdrant server
                self.client = QdrantClient(url=self.db_url)
                logger.info("Connected to remote Qdrant at %s", self.db_url)
            elif self.db_path:
                # Connect to local Qdrant
                try:
                    self.client = QdrantClient(path=self.db_path)
                except RuntimeError as re:
                    # Common failure when the local storage folder is locked by another Qdrant instance
                    logger.error(
                        "Failed to connect to local Qdrant at %s: %s. "
                        "This usually means the storage folder is already used by another Qdrant process. "
                        "Run only one local Qdrant instance or switch to a Qdrant server (set db_url).",
                        self.db_path,
                        re,
                    )
                    raise
                logger.info("Connected to local Qdrant at %s", self.db_path)
            else:
                raise ValueError("Either db_url or db_path must be provided")
        except Exception as e:
            logger.error("Failed to connect to Qdrant: %s", e)
            raise

    def disconnect(self):
        # FIX: explicitly close the underlying HTTP session to release sockets
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    # ── Collection helpers ────────────────────────────────────────────────────

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def list_all_collections(self) -> List:
        return self.client.get_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)

    def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        # If requested, remove any existing collection first
        if do_reset:
            self.delete_collection(collection_name=collection_name)

        # If collection does not exist, create it with the provided embedding size
        if not self.is_collection_existed(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method,
                ),
            )
            return True

        # Collection exists — verify vector size matches requested embedding_size.
        try:
            info = self.client.get_collection(collection_name=collection_name)
            existing_size = None

            # Try several possible shapes of the returned object to extract vector size
            try:
                # qdrant_client may return a model with .result.vectors or .vectors
                if hasattr(info, "result") and getattr(info, "result"):
                    vectors = getattr(info.result, "vectors", None)
                else:
                    vectors = getattr(info, "vectors", None)

                if isinstance(vectors, dict):
                    # vectors is a mapping of named vector configs; take the first
                    for v in vectors.values():
                        if hasattr(v, "size"):
                            existing_size = int(getattr(v, "size"))
                            break
                        if isinstance(v, dict) and "size" in v:
                            existing_size = int(v.get("size"))
                            break
            except Exception:
                existing_size = None

            # If we could determine an existing size and it differs, handle accordingly
            if existing_size is not None and existing_size != embedding_size:
                msg = (
                    f"Collection '{collection_name}' exists with embedding size {existing_size}, "
                    f"but requested embedding size is {embedding_size}."
                )
                if do_reset:
                    # Already removed above when do_reset True, but keep for safety
                    self.delete_collection(collection_name=collection_name)
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=embedding_size,
                            distance=self.distance_method,
                        ),
                    )
                    logger.warning(
                        "%s Recreated collection with new embedding size.", msg
                    )
                    return True
                else:
                    # Do not implicitly alter an existing collection with a different vector size
                    logger.error(
                        "%s To recreate with the new size set do_reset=True or delete the collection manually.",
                        msg,
                    )
                    raise RuntimeError(msg)

        except Exception as e:
            # If we couldn't inspect the collection for some reason, log and raise
            logger.error(
                "Failed to verify existing collection '%s': %s", collection_name, e
            )
            raise

        # Collection exists and matches the requested size (or we couldn't determine size) — nothing to do
        return False

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id=None,
    ):
        if not self.is_collection_existed(collection_name):
            logger.error("Collection does not exist: %s", collection_name)
            return False

        if record_id is None:
            record_id = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**63 - 1)

        try:
            # Use upsert (idempotent) instead of deprecated upload_records
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=record_id,
                        vector=vector,
                        payload={"text": text, "metadata": metadata},
                    )
                ],
            )
            return True
        except Exception as e:
            logger.error("Qdrant insert_one error: %s", e)
            return False

    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list = None,
        record_ids: list = None,
        batch_size: int = 50,
    ):
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = [
                int(hashlib.md5(f"{i}_{t}".encode()).hexdigest(), 16) % (2**63 - 1)
                for i, t in enumerate(texts)
            ]

        for i in range(0, len(texts), batch_size):
            b_end = i + batch_size
            points = [
                models.PointStruct(
                    id=record_ids[j],
                    vector=vectors[j],
                    payload={"text": texts[j], "metadata": metadata[j]},
                )
                for j in range(i, min(b_end, len(texts)))
            ]

            try:
                # upsert is idempotent and replaces deprecated upload_records
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
            except Exception as e:
                logger.error("Qdrant insert_many batch [%d:%d] error: %s", i, b_end, e)
                return False

        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 5,
        score_threshold: float = None,
    ):
        kwargs = dict(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
        )
        if score_threshold is not None and score_threshold > 0:
            kwargs["score_threshold"] = score_threshold

        return self.client.search(**kwargs)

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        vector: list,
        limit: int = 5,
        semantic_weight: float = 0.6,
    ):
        """Hybrid search using a re-ranking pattern.

        1. Semantic search retrieves top-(limit × 4) candidates.
        2. BM25 re-ranks *only those candidates* (not the full collection).
        3. Results are combined with a weighted score and top-k returned.

        This avoids the catastrophic O(N) scroll of the old implementation.
        """
        try:
            fetch_limit = limit * 4

            # Stage 1: semantic candidates
            semantic_results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=fetch_limit,
            )

            if not semantic_results:
                return []

            # Stage 2: BM25 re-rank over candidates only
            candidate_texts = [r.payload.get("text", "") for r in semantic_results]
            tokenized_query = query_text.lower().split()
            bm25 = BM25Okapi([t.lower().split() for t in candidate_texts])
            bm25_scores = bm25.get_scores(tokenized_query)

            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            keyword_weight = 1.0 - semantic_weight

            combined = []
            for result, bm25_score in zip(semantic_results, bm25_scores):
                norm_bm25 = bm25_score / max_bm25
                norm_semantic = float(result.score)
                combined_score = keyword_weight * norm_bm25 + semantic_weight * norm_semantic
                combined.append((combined_score, result))

            # Sort descending and take top-k
            combined.sort(key=lambda x: x[0], reverse=True)
            final = []
            for score, point in combined[:limit]:
                point.score = score
                final.append(point)

            return final

        except Exception as e:
            logger.error("Hybrid search error: %s — falling back to semantic", e)
            return self.search_by_vector(
                collection_name=collection_name, vector=vector, limit=limit
            )
