import hashlib
import logging
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums

logger = logging.getLogger(__name__)


class DummyPoint:
    def __init__(self, id: str, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = payload


class PineconeDBProvider(VectorDBInterface):

    def __init__(self, api_key: str, environment: str, distance_method: str):
        self.api_key = api_key
        self.environment = environment or "us-east-1"  # Default fallback if not provided
        self._distance_method = distance_method
        self.client: Optional[Pinecone] = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = "cosine"
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = "dotproduct"
        else:
            self.distance_method = "euclidean"

    def connect(self):
        try:
            self.client = Pinecone(api_key=self.api_key)
        except Exception as e:
            logger.error("Failed to connect to Pinecone: %s", e)
            raise

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        if not self.client:
            self.connect()
        # Pinecone indexes correspond to collections
        return collection_name in self.client.list_indexes().names()

    def list_all_collections(self) -> List:
        if not self.client:
            self.connect()
        return self.client.list_indexes().names()

    def get_collection_info(self, collection_name: str) -> dict:
        if not self.client:
            self.connect()
        try:
            index = self.client.Index(collection_name)
            stats = index.describe_index_stats()
            return {
                "name": collection_name,
                "vectors_count": stats.total_vector_count,
                "embedding_size": stats.dimension,
            }
        except Exception:
            return {}

    def delete_collection(self, collection_name: str):
        if not self.client:
            self.connect()
        if self.is_collection_existed(collection_name):
            self.client.delete_index(collection_name)

    def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        if not self.client:
            self.connect()

        if do_reset:
            self.delete_collection(collection_name)

        if not self.is_collection_existed(collection_name):
            self.client.create_index(
                name=collection_name,
                dimension=embedding_size,
                metric=self.distance_method,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
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
        record_id=None,
    ):
        return self.insert_many(
            collection_name, [text], [vector], [metadata], [record_id]
        )

    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list = None,
        record_ids: list = None,
        batch_size: int = 50,
    ):
        if not self.client:
            self.connect()

        try:
            index = self.client.Index(collection_name)
        except Exception:
            logger.error("Collection does not exist: %s", collection_name)
            return False

        if metadata is None:
            metadata = [None] * len(texts)

        # Pinecone requires string IDs
        if record_ids is None:
            record_ids = [
                hashlib.md5(f"{i}_{t}".encode()).hexdigest()
                for i, t in enumerate(texts)
            ]
        else:
            record_ids = [str(rid) for rid in record_ids]

        # Flatten metadata for pinecone requirement + add text
        for i in range(len(texts)):
            if metadata[i] is None:
                metadata[i] = {}
            metadata[i]["text"] = texts[i]
            # Ensure metadata values are strings, ints, floats, bools, or lists of strings
            for k, v in list(metadata[i].items()):
                 if not isinstance(v, (str, int, float, bool, list)):
                     import json
                     metadata[i][k] = json.dumps(v)

        for i in range(0, len(texts), batch_size):
            b_end = i + batch_size
            b_vectors = vectors[i:b_end]
            b_metadata = metadata[i:b_end]
            b_ids = record_ids[i:b_end]

            records = []
            for j in range(len(b_ids)):
                records.append({
                    "id": b_ids[j],
                    "values": b_vectors[j],
                    "metadata": b_metadata[j]
                })

            try:
                index.upsert(vectors=records)
            except Exception as e:
                logger.error("Pinecone insert_many error: %s", e)
                return False

        return True

    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 5,
        score_threshold: float = None,
    ):
        if not self.client:
            self.connect()

        try:
            index = self.client.Index(collection_name)
        except Exception:
            return []

        results = index.query(
            vector=vector,
            top_k=limit,
            include_metadata=True,
        )

        final_results = []
        if results and hasattr(results, "matches"):
            for match in results.matches:
                score = match.score
                if score_threshold is not None:
                    if score < score_threshold:
                        continue

                text = match.metadata.pop("text", "") if match.metadata else ""
                payload = {"text": text, "metadata": match.metadata or {}}
                
                final_results.append(DummyPoint(id=match.id, score=score, payload=payload))

        return final_results

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        vector: list,
        limit: int = 5,
        semantic_weight: float = 0.6,
    ):
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        try:
            fetch_limit = limit * 4
            semantic_results = self.search_by_vector(collection_name, vector, fetch_limit)

            if not semantic_results:
                return []

            import re as _re
            _tok = lambda s: _re.findall(r"(?u)\b\w+\b", s.lower())

            candidate_texts = [r.payload.get("text", "") for r in semantic_results]
            tokenized_query = _tok(query_text)
            bm25 = BM25Okapi([_tok(t) for t in candidate_texts])
            bm25_scores = bm25.get_scores(tokenized_query)

            k = 60
            bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked_indices)}
            
            keyword_weight = 1.0 - semantic_weight
            
            combined = []
            for i, result in enumerate(semantic_results):
                semantic_rank = i + 1
                bm25_rank = bm25_ranks[i]
                
                rrf_score = (semantic_weight * (1.0 / (k + semantic_rank))) + (keyword_weight * (1.0 / (k + bm25_rank)))
                combined.append((rrf_score, result))

            combined.sort(key=lambda x: x[0], reverse=True)
            
            # Normalize to [0, 1] so top result = 1.0 (keeps SEARCH_SCORE_THRESHOLD compatible)
            max_score = combined[0][0] if combined else 1.0
            final = []
            for score, point in combined[:limit]:
                point.score = round(score / max_score, 4)
                final.append(point)

            return final

        except Exception as e:
            logger.error("Hybrid search error: %s — falling back to semantic", e)
            return self.search_by_vector(
                collection_name=collection_name, vector=vector, limit=limit
            )
