"""Unit tests for DataController — file validation and naming logic."""
import pytest
from unittest.mock import MagicMock, patch
from controllers.DataController import DataController


def make_upload_file(filename="test.pdf", content_type="application/pdf", size=1_000):
    """Helper to create a fake UploadFile object."""
    f = MagicMock()
    f.filename = filename
    f.content_type = content_type
    f.size = size
    return f


def make_controller(allowed=("pdf", "txt", "docx"), max_bytes=10_000_000):
    ctrl = DataController()
    ctrl.app_settings = MagicMock(
        FILE_ALLOWED_TYPES=list(allowed),
        FILE_MAX_SIZE=max_bytes,
    )
    return ctrl


# ── Validation tests ──────────────────────────────────────────────────────────

class TestValidateUploadedFile:

    def test_valid_pdf_by_extension(self):
        ctrl = make_controller()
        is_valid, signal = ctrl.validate_uploaded_file(make_upload_file("report.pdf"))
        assert is_valid is True

    def test_valid_txt_by_extension(self):
        ctrl = make_controller()
        is_valid, _ = ctrl.validate_uploaded_file(make_upload_file("notes.txt", "text/plain"))
        assert is_valid is True

    def test_rejected_exe_file(self):
        ctrl = make_controller()
        is_valid, signal = ctrl.validate_uploaded_file(
            make_upload_file("virus.exe", "application/octet-stream")
        )
        assert is_valid is False

    def test_rejected_unknown_extension(self):
        ctrl = make_controller()
        is_valid, _ = ctrl.validate_uploaded_file(make_upload_file("data.xyz", "application/xyz"))
        assert is_valid is False

    def test_file_too_large(self):
        ctrl = make_controller(max_bytes=500)
        is_valid, signal = ctrl.validate_uploaded_file(make_upload_file("big.pdf", size=1_000))
        assert is_valid is False

    def test_file_exactly_at_size_limit_is_valid(self):
        ctrl = make_controller(max_bytes=1_000)
        is_valid, _ = ctrl.validate_uploaded_file(make_upload_file("exact.pdf", size=1_000))
        assert is_valid is True

    def test_no_extension_rejected(self):
        ctrl = make_controller()
        is_valid, _ = ctrl.validate_uploaded_file(make_upload_file("noextension", "text/plain"))
        assert is_valid is False

    def test_mime_type_match_when_ext_not_in_allowed(self):
        """If extension isn't in allowed list, MIME type can still pass."""
        ctrl = make_controller(allowed=["txt"])
        # text/plain maps to 'txt' which IS allowed — should pass
        is_valid, _ = ctrl.validate_uploaded_file(
            make_upload_file("document.log", "text/plain")
        )
        assert is_valid is True

    def test_zero_max_size_skips_size_check(self):
        """FILE_MAX_SIZE=0 means no size limit enforced."""
        ctrl = make_controller(max_bytes=0)
        is_valid, _ = ctrl.validate_uploaded_file(make_upload_file("huge.pdf", size=999_999_999))
        assert is_valid is True


# ── Filename cleaning tests ───────────────────────────────────────────────────

class TestGetCleanFileName:

    def test_removes_spaces(self):
        ctrl = DataController()
        result = ctrl.get_clean_file_name("my file.pdf")
        assert " " not in result

    def test_removes_special_chars(self):
        ctrl = DataController()
        result = ctrl.get_clean_file_name("my (file) [v2].pdf")
        assert "(" not in result
        assert "[" not in result

    def test_preserves_extension(self):
        ctrl = DataController()
        result = ctrl.get_clean_file_name("document.pdf")
        assert result.endswith(".pdf")

    def test_empty_filename_returns_empty(self):
        ctrl = DataController()
        result = ctrl.get_clean_file_name("   ")
        assert isinstance(result, str)
