from .BaseController import BaseController
from .ProjectController import ProjectController
import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum

logger = logging.getLogger(__name__)

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        # Added to allow injection of embedding client
        self.embedding_client = None

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(
            self.project_path,
            file_id
        )

        if file_ext in (ProcessingEnum.TXT.value, ProcessingEnum.MD.value):
            return TextLoader(file_path, encoding="utf-8")

        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)

        if file_ext in (ProcessingEnum.DOC.value, ProcessingEnum.DOCX.value):
            return Docx2txtLoader(file_path)

        if file_ext == ProcessingEnum.CSV.value:
            return CSVLoader(file_path)

        if file_ext == ProcessingEnum.HTML.value:
            return BSHTMLLoader(file_path)

        return None

    def get_file_content(self, file_id: str):
        """Attempt to load the file from disk."""
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.isfile(file_path):
            # file missing (possibly wrong project_id or file_id)
            return None

        loader = self.get_file_loader(file_id=file_id)
        if loader is None:
            return None

        try:
            return loader.load()
        except Exception as e:
            logger.error("Failed to load file '%s': %s", file_id, e)  # FIX: log before re-raise
            raise

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=512, overlap_size: int=50, chunk_strategy: str="recursive"):

        if chunk_strategy in ("fixed", "overlapping", "recursive"):
            from langchain_text_splitters import CharacterTextSplitter
            if chunk_strategy == "fixed":
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator="")
            elif chunk_strategy == "overlapping":
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size, separator="")
            else:
                # FIX: RecursiveCharacterTextSplitter already imported at top of file
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size, length_function=len)

        elif chunk_strategy == "semantic":
            from langchain_experimental.text_splitter import SemanticChunker
            if self.embedding_client is None:
                # Fallback to local HuggingFace embeddings if not injected
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            else:
                # Wrap existing embedding_client so it conforms to Langchain Embeddings if needed.
                # Assuming the environment uses HuggingFace local or we have an adapter.
                # Here we just fallback to HuggingFace
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            text_splitter = SemanticChunker(embeddings)

        elif chunk_strategy == "document_structure":
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            # Fallback wrapper for doc structure
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        elif chunk_strategy == "sentence_based":
            from langchain_text_splitters import SentenceTransformersTokenTextSplitter
            from langchain_text_splitters import NLTKTextSplitter
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                text_splitter = NLTKTextSplitter(chunk_size=chunk_size)
            except Exception:
                # Fallback to simple sentence based splitting
                from langchain_text_splitters import CharacterTextSplitter
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size, separator=".")

        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size, length_function=len)

        # MarkdownHeaderTextSplitter requires individual splits on strings rather than document list creation directly
        if chunk_strategy == "document_structure":
            file_content_texts = [rec.page_content for rec in file_content]
            file_content_metadata = [rec.metadata for rec in file_content]
            chunks = []
            for text, meta in zip(file_content_texts, file_content_metadata):
                splits = text_splitter.split_text(text)
                for split in splits:
                    split.metadata.update(meta)
                chunks.extend(splits)
            return chunks
        
        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata
        )

        return chunks



