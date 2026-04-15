from enum import Enum


class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    FAISS = "FAISS"
    CHROMA = "CHROMA"
    PINECONE = "PINECONE"


class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"
