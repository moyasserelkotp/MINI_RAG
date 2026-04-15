import os
import hashlib
import logging
from typing import List, Optional
import chromadb
from rank_bm25 import BM25Okapi

from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums

logger = logging.getLogger(__name__)


class DummyPoint:
    def __init__(self, id: str, score: float, payload: dict):
        self.id = id
        self.score = score
        self.payload = payload


class ChromaDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):
        self.db_path = db_path
        self._distance_method = distance_method
        self.client: Optional[chromadb.PersistentClient] = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = "cosine"
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = "ip"
        else:
            self.distance_method = "l2"

    def connect(self):
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
        except Exception as e:
            logger.error("Failed to connect to ChromaDB at %s: %s", self.db_path, e)
            raise

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        if not self.client:
            self.connect()
        try:
            self.client.get_collection(name=collection_name)
            return True
        except ValueError:
            return False
        except Exception:
            return False

    def list_all_collections(self) -> List:
        if not self.client:
            self.connect()
        return [c.name for c in self.client.list_collections()]

    def get_collection_info(self, collection_name: str) -> dict:
        if not self.client:
            self.connect()
        try:
            collection = self.client.get_collection(name=collection_name)
            return {
                "name": collection.name,
                "vectors_count": collection.count(),
                "embedding_size": collection.metadata.get("embedding_size", 0),
            }
        except Exception:
            return {}

    def delete_collection(self, collection_name: str):
        if not self.client:
            self.connect()
        if self.is_collection_existed(collection_name):
            self.client.delete_collection(name=collection_name)

    def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        if not self.client:
            self.connect()

        if do_reset:
            self.delete_collection(collection_name)

        if not self.is_collection_existed(collection_name):
            self.client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": self.distance_method,
                    "embedding_size": embedding_size,
                },
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
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            logger.error("Collection does not exist: %s", collection_name)
            return False

        if metadata is None:
            metadata = [None] * len(texts)

        # Chroma requires string IDs
        if record_ids is None:
            record_ids = [
                hashlib.md5(f"{i}_{t}".encode()).hexdigest()
                for i, t in enumerate(texts)
            ]
        else:
            record_ids = [str(rid) for rid in record_ids]

        for i in range(0, len(texts), batch_size):
            b_end = i + batch_size
            b_texts = texts[i:b_end]
            b_vectors = vectors[i:b_end]
            b_metadata = metadata[i:b_end]
            b_ids = record_ids[i:b_end]

            clean_metadata = []
            for meta in b_metadata:
                if meta is None:
                    clean_metadata.append({})
                else:
                    # Chroma only allows str, int, float, bool in metadata maps. We flatten or stringify.
                    flat_m = {}
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool)):
                            flat_m[k] = v
                        else:
                            import json
                            flat_m[k] = json.dumps(v)
                    clean_metadata.append(flat_m)

            try:
                collection.upsert(
                    embeddings=b_vectors,
                    documents=b_texts,
                    metadatas=clean_metadata,
                    ids=b_ids,
                )
            except Exception as e:
                logger.error("Chroma insert_many error: %s", e)
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
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            return []

        results = collection.query(
            query_embeddings=[vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        final_results = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                id = results["ids"][0][i]
                distance = results["distances"][0][i]
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]

                if score_threshold is not None:
                    # Chroma returns distance. L2 goes up, cosine goes up depending on space. 
                    # If cosine distance is returned, smaller is better. If dot product is returned, it might be reversed depending on internal hnsw:space. 
                    if self.distance_method == "cosine":
                        if distance > score_threshold: # Assuming cosine distance
                            continue
                    elif self.distance_method == "l2":
                        if distance > score_threshold:
                            continue

                # Reconstruct original structure
                payload = {"text": document, "metadata": metadata}
                final_results.append(DummyPoint(id=id, score=float(distance), payload=payload))

        return final_results

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        vector: list,
        limit: int = 5,
        semantic_weight: float = 0.6,
    ):
        try:
            fetch_limit = limit * 4
            semantic_results = self.search_by_vector(collection_name, vector, fetch_limit)

            if not semantic_results:
                return []

            candidate_texts = [r.payload.get("text", "") for r in semantic_results]
            tokenized_query = query_text.lower().split()
            bm25 = BM25Okapi([t.lower().split() for t in candidate_texts])
            bm25_scores = bm25.get_scores(tokenized_query)

            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            keyword_weight = 1.0 - semantic_weight

            combined = []
            for result, bm25_score in zip(semantic_results, bm25_scores):
                norm_bm25 = bm25_score / max_bm25
                
                # Chroma distance: lower is better for cosine distance. 
                # To combine with bm25 (higher is better), we need to invert chroma distance
                # Cosine distance ranges [0, 2], so similarity = 1 - (dist / 2)
                sim = 1.0 - min(result.score / 2.0, 1.0) if self.distance_method == "cosine" else 1.0 / (1.0 + result.score)
                norm_semantic = sim

                combined_score = keyword_weight * norm_bm25 + semantic_weight * norm_semantic
                combined.append((combined_score, result))

            combined.sort(key=lambda x: x[0], reverse=True)
            final = []
            for score, point in combined[:limit]:
                # replace score with the combined one, as other providers do
                point.score = score
                final.append(point)

            return final

        except Exception as e:
            logger.error("Hybrid search error: %s — falling back to semantic", e)
            return self.search_by_vector(
                collection_name=collection_name, vector=vector, limit=limit
            )
