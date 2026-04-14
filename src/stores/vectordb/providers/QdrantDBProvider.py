from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
import hashlib
from rank_bm25 import BM25Okapi


class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

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
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)

        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size, distance=self.distance_method
                ),
            )

            return True

        return False

    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id: str = None,
    ):

        if not self.is_collection_existed(collection_name):
            self.logger.error(
                f"Can not insert new record to non-existed collection: {collection_name}"
            )
            return False

        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        vector=vector, payload={"text": text, "metadata": metadata}
                    )
                ],
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True

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
                int(hashlib.md5(f"{i}_{text}".encode()).hexdigest(), 16) % (2**63 - 1)
                for i, text in enumerate(texts)
            ]

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={"text": batch_texts[x], "metadata": batch_metadata[x]},
                )
                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records,
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True

    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 5,
        score_threshold: float = None,
    ):

        return self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
        )

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        vector: list,
        limit: int = 5,
        semantic_weight: float = 0.7,
    ):
        """
        Hybrid search combining BM25 (keyword) and semantic search.

        Args:
            collection_name: Name of the collection
            query_text: Query text for BM25 matching
            vector: Query vector for semantic search
            limit: Number of results to return
            semantic_weight: Weight for semantic search (0-1), keyword gets (1-semantic_weight)
        """
        try:
            # Get all points from collection for BM25 ranking
            all_points = self.client.scroll(
                collection_name=collection_name,
                limit=10000,
            )[0]

            if not all_points:
                return []

            # Extract texts for BM25
            texts = []
            point_map = {}

            for point in all_points:
                text = point.payload.get("text", "")
                texts.append(text)
                point_map[len(texts) - 1] = point

            # BM25 ranking
            tokenized_query = query_text.lower().split()
            bm25 = BM25Okapi([text.lower().split() for text in texts])
            bm25_scores = bm25.get_scores(tokenized_query)

            # Semantic search
            semantic_results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit * 3,  # Get more results to combine
            )

            # Create semantic score map
            semantic_map = {}
            for idx, result in enumerate(semantic_results):
                semantic_map[result.id] = result.score

            # Combine scores
            combined_results = {}
            for idx, (point, bm25_score) in enumerate(zip(all_points, bm25_scores)):
                semantic_score = semantic_map.get(point.id, 0)
                # Normalize both scores to 0-1 range
                norm_bm25 = bm25_score / (max(bm25_scores) + 1e-10)
                norm_semantic = semantic_score

                # Weighted combination
                combined_score = (
                    1 - semantic_weight
                ) * norm_bm25 + semantic_weight * norm_semantic

                combined_results[point.id] = {
                    "point": point,
                    "combined_score": combined_score,
                    "bm25_score": norm_bm25,
                    "semantic_score": norm_semantic,
                }

            # Sort by combined score
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True,
            )

            # Return top results with updated scores
            final_results = []
            for item in sorted_results[:limit]:
                # Update the score to combined score
                item["point"].score = item["combined_score"]
                final_results.append(item["point"])

            return final_results

        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}")
            # Fallback to semantic search
            return self.search_by_vector(
                collection_name=collection_name,
                vector=vector,
                limit=limit,
            )
