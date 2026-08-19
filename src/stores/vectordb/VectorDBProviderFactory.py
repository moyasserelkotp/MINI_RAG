from .providers import QdrantDBProvider, FaissDBProvider, ChromaDBProvider, PineconeDBProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController


class VectorDBProviderFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            # Use remote URL if available, otherwise use local path
            db_url = getattr(self.config, "VECTOR_DB_URL", None)
            db_path = None

            if not db_url:
                db_path = self.base_controller.get_database_path(
                    db_name=self.config.VECTOR_DB_PATH
                )

            return QdrantDBProvider(
                db_path=db_path,
                db_url=db_url,
                db_api_key=getattr(self.config, "VECTOR_DB_API_KEY", None),
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )
        elif provider == VectorDBEnums.FAISS.value:
            db_path = self.base_controller.get_database_path(
                db_name=self.config.VECTOR_DB_PATH + "_faiss"
            )

            return FaissDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )
        elif provider == VectorDBEnums.CHROMA.value:
            db_path = self.base_controller.get_database_path(
                db_name=self.config.VECTOR_DB_PATH + "_chroma"
            )

            return ChromaDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )
        elif provider == VectorDBEnums.PINECONE.value:
            return PineconeDBProvider(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENV,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )

        return None
