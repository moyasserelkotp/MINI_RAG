from enum import Enum

class ChunkingStrategyEnum(Enum):
    FIXED = "fixed"
    OVERLAPPING = "overlapping"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    DOCUMENT_STRUCTURE = "document_structure"
    SENTENCE_BASED = "sentence_based"
