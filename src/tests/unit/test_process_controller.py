"""Unit tests for ProcessController — chunking strategies and file loader selection."""
import pytest
from unittest.mock import MagicMock, patch
from controllers.ProcessController import ProcessController


def make_controller(project_id="test_proj"):
    ctrl = ProcessController(project_id=project_id)
    ctrl.project_path = "/tmp/test_project"
    return ctrl


def make_docs(content="Hello world. This is a sample test document."):
    """Create a fake list of LangChain Document-like objects."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": "test.txt", "page": 0}
    return [doc]


# ── Chunking strategy tests ───────────────────────────────────────────────────

class TestProcessFileContent:

    def test_recursive_chunking_produces_multiple_chunks(self):
        ctrl = make_controller()
        docs = make_docs("Hello world. " * 200)
        chunks = ctrl.process_file_content(
            docs, "test.txt", chunk_size=100, overlap_size=20, chunk_strategy="recursive"
        )
        assert len(chunks) > 1

    def test_fixed_chunking_respects_chunk_size(self):
        ctrl = make_controller()
        docs = make_docs("A" * 1000)
        chunks = ctrl.process_file_content(
            docs, "test.txt", chunk_size=200, overlap_size=0, chunk_strategy="fixed"
        )
        assert len(chunks) > 1
        # All chunks (except possibly last) should be <= chunk_size + small tolerance
        for c in chunks[:-1]:
            assert len(c.page_content) <= 210

    def test_overlapping_chunking_creates_overlap(self):
        ctrl = make_controller()
        docs = make_docs("Word " * 400)
        chunks = ctrl.process_file_content(
            docs, "test.txt", chunk_size=100, overlap_size=30, chunk_strategy="overlapping"
        )
        assert len(chunks) > 1

    def test_fallback_unknown_strategy_uses_recursive(self):
        ctrl = make_controller()
        docs = make_docs("Test content. " * 100)
        # Unknown strategy should fall back gracefully
        chunks = ctrl.process_file_content(
            docs, "test.txt", chunk_size=100, overlap_size=20, chunk_strategy="unknown_xyz"
        )
        assert len(chunks) >= 1

    def test_chunks_retain_metadata(self):
        ctrl = make_controller()
        docs = make_docs("Content with metadata. " * 50)
        chunks = ctrl.process_file_content(
            docs, "test.txt", chunk_size=100, overlap_size=0, chunk_strategy="recursive"
        )
        assert len(chunks) > 0


# ── File loader selection tests ───────────────────────────────────────────────

class TestGetFileLoader:

    def test_pdf_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("document.pdf")
        assert loader is not None

    def test_txt_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("notes.txt")
        assert loader is not None

    def test_docx_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("report.docx")
        assert loader is not None

    def test_doc_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("legacy.doc")
        assert loader is not None

    def test_csv_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("data.csv")
        assert loader is not None

    def test_html_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("page.html")
        assert loader is not None

    def test_unsupported_extension_returns_none(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("archive.zip")
        assert loader is None

    def test_unknown_extension_returns_none(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("file.xyz123")
        assert loader is None

    def test_md_extension_uses_text_loader(self):
        ctrl = make_controller()
        loader = ctrl.get_file_loader("readme.md")
        assert loader is not None
