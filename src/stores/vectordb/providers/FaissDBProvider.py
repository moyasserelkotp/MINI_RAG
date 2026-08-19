import os
import json
import logging
import hashlib
from typing import List, Optional
import numpy as np
import faiss

from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class DummyPoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


class FaissDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.distance_method = distance_method
        self.collections = {}

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        pass

    def disconnect(self):
        self.collections.clear()

    # ── File Storage Helpers ──────────────────────────────────────────────────

    def _get_collection_paths(self, collection_name: str):
        index_path = os.path.join(self.db_path, f"{collection_name}.index")
        metadata_path = os.path.join(self.db_path, f"{collection_name}_meta.json")
        info_path = os.path.join(self.db_path, f"{collection_name}_info.json")
        return index_path, metadata_path, info_path

    def _load_collection(self, collection_name: str) -> bool:
        if collection_name in self.collections:
            return True

        index_path, metadata_path, info_path = self._get_collection_paths(
            collection_name
        )
        if (
            not os.path.exists(index_path)
            or not os.path.exists(metadata_path)
            or not os.path.exists(info_path)
        ):
            return False

        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)

            self.collections[collection_name] = {
                "index": index,
                "metadata": metadata,
                "info": info,
            }
            return True
        except Exception as e:
            logger.error("Failed to load FAISS collection %s: %s", collection_name, e)
            return False

    def _save_collection(self, collection_name: str):
        if collection_name not in self.collections:
            return

        index_path, metadata_path, info_path = self._get_collection_paths(
            collection_name
        )
        col = self.collections[collection_name]

        try:
            faiss.write_index(col["index"], index_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(col["metadata"], f)

            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(col["info"], f)
        except Exception as e:
            logger.error("Failed to save FAISS collection %s: %s", collection_name, e)

    # ── Collection helpers ────────────────────────────────────────────────────

    def is_collection_existed(self, collection_name: str) -> bool:
        if collection_name in self.collections:
            return True
        index_path, metadata_path, info_path = self._get_collection_paths(
            collection_name
        )
        return (
            os.path.exists(index_path)
            and os.path.exists(metadata_path)
            and os.path.exists(info_path)
        )

    def list_all_collections(self) -> List:
        collections = []
        if os.path.exists(self.db_path):
            for file in os.listdir(self.db_path):
                if file.endswith(".index"):
                    collections.append(file[: -len(".index")])
        return list(set(collections))

    def get_collection_info(self, collection_name: str) -> dict:
        if self._load_collection(collection_name):
            col = self.collections[collection_name]
            return {
                "name": collection_name,
                "vectors_count": col["index"].ntotal,
                "embedding_size": col["info"].get("embedding_size", 0),
            }
        return {}

    def delete_collection(self, collection_name: str):
        if collection_name in self.collections:
            del self.collections[collection_name]

        index_path, metadata_path, info_path = self._get_collection_paths(
            collection_name
        )
        for p in [index_path, metadata_path, info_path]:
            if os.path.exists(p):
                os.remove(p)

    def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        if do_reset:
            self.delete_collection(collection_name=collection_name)

        if not self.is_collection_existed(collection_name):
            if self.distance_method == DistanceMethodEnums.COSINE.value:
                # Cosine requires normalized L2 vectors with Inner Product index
                base_index = faiss.IndexFlatIP(embedding_size)
            elif self.distance_method == DistanceMethodEnums.DOT.value:
                base_index = faiss.IndexFlatIP(embedding_size)
            else:
                base_index = faiss.IndexFlatL2(embedding_size)

            # Map index required for custom IDs
            index = faiss.IndexIDMap(base_index)

            self.collections[collection_name] = {
                "index": index,
                "metadata": {},
                "info": {"embedding_size": embedding_size},
            }
            self._save_collection(collection_name)
            return True

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
        batch_size: int = 50,  # Batching parameter not strictly needed for FAISS ID map, but supported.
    ):
        if not self._load_collection(collection_name):
            logger.error("Collection does not exist: %s", collection_name)
            return False

        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = [
                int(hashlib.md5(f"{i}_{t}".encode()).hexdigest(), 16) % (2**63 - 1)
                for i, t in enumerate(texts)
            ]

        col = self.collections[collection_name]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            b_end = i + batch_size
            b_texts = texts[i:b_end]
            b_vectors = vectors[i:b_end]
            b_metadata = metadata[i:b_end]
            b_record_ids = record_ids[i:b_end]

            vecs_np = np.array(b_vectors, dtype=np.float32)

            if self.distance_method == DistanceMethodEnums.COSINE.value:
                faiss.normalize_L2(vecs_np)

            ids_np = np.array(b_record_ids, dtype=np.int64)

            try:
                # remove existing points if present (to act like upsert)
                try:
                    col["index"].remove_ids(ids_np)
                except Exception:
                    pass

                col["index"].add_with_ids(vecs_np, ids_np)

                for j, rec_id in enumerate(b_record_ids):
                    col["metadata"][str(rec_id)] = {
                        "text": b_texts[j],
                        "metadata": b_metadata[j],
                    }

            except Exception as e:
                logger.error("Faiss insert_many batch error: %s", e)
                return False

        self._save_collection(collection_name)
        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 5,
        score_threshold: float = None,
    ):
        if not self._load_collection(collection_name):
            return []

        col = self.collections[collection_name]

        vec_np = np.array([vector], dtype=np.float32)

        if self.distance_method == DistanceMethodEnums.COSINE.value:
            faiss.normalize_L2(vec_np)

        distances, indices = col["index"].search(vec_np, limit)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:
                break

            score = float(distances[0][i])
            if score_threshold is not None:
                if self.distance_method in (
                    DistanceMethodEnums.COSINE.value,
                    DistanceMethodEnums.DOT.value,
                ):
                    # Higher is better
                    if score < score_threshold:
                        continue
                else:
                    # L2 distance: lower is better distance, so score < threshold means better? 
                    # threshold behaves differently depending on method. Usually Qdrant returns similarity.
                    # We will assume similarity context. 
                    if score < score_threshold:
                        continue

            payload = col["metadata"].get(str(idx), {})
            results.append(DummyPoint(id=idx, score=score, payload=payload))

        return results

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

            semantic_results = self.search_by_vector(
                collection_name=collection_name,
                vector=vector,
                limit=fetch_limit,
            )

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
