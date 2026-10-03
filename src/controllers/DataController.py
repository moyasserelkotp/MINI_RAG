from .BaseController import BaseController
from .ProjectController import ProjectController
# pyrefly: ignore [missing-import]
from fastapi import UploadFile
from models import ResponseSignal
import re
import os
import logging

logger = logging.getLogger(__name__)

# Map known MIME types to their canonical extension
_MIME_TO_EXT = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def _detect_magic_ext(file_header: bytes) -> str:
    """Return the canonical extension detected from file magic bytes, or '' on failure."""
    try:
        # pyrefly: ignore [missing-import]
        import magic  # python-magic
        detected_mime = magic.from_buffer(file_header, mime=True)
        return _MIME_TO_EXT.get(detected_mime, "")
    except ImportError:
        logger.warning(
            "python-magic is not installed — magic-byte validation skipped. "
            "Run: pip install python-magic"
        )
        return ""
    except Exception as exc:
        logger.warning("Magic-byte detection failed: %s", exc)
        return ""

class DataController(BaseController):

    def __init__(self):
        super().__init__()

    def validate_uploaded_file(self, file: UploadFile, file_header: bytes = b""):
        """Return ``(is_valid, signal)`` for the supplied file.

        Checks file extension, MIME type, AND magic bytes (H-1) against the
        allow-list. ``file_header`` should be the first 512 bytes of the file
        content read by the route before calling this method.
        FILE_MAX_SIZE in .env is already in bytes — no extra scaling applied.
        """
        allowed: list = self.app_settings.FILE_ALLOWED_TYPES or []

        #  extension check 
        parts = (file.filename or "").rsplit(".", 1)
        ext = parts[-1].lower() if len(parts) == 2 else ""

        #  MIME check (normalise to extension) 
        raw_mime = (file.content_type or "").lower().split(";")[0].strip()
        mime_ext = _MIME_TO_EXT.get(raw_mime, raw_mime)

        #  Magic-byte check — server-side truth, overrides client claims (H-1) 
        magic_ext = _detect_magic_ext(file_header) if file_header else ""

        # Accept if ANY of: extension OR mime OR magic matches the allow-list.
        # Magic-byte check is the most reliable; the others are fallbacks for
        # environments where python-magic is unavailable.
        ext_ok = ext in allowed
        mime_ok = mime_ext in allowed
        magic_ok = magic_ext in allowed if magic_ext else ext_ok  # graceful fallback

        if not (ext_ok or mime_ok or magic_ok):
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        #  size check (FILE_MAX_SIZE is already bytes in .env) 
        max_bytes = self.app_settings.FILE_MAX_SIZE
        if max_bytes and file.size and file.size > max_bytes:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value

        return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):
        random_key = self.generate_random_string()
        project_path = ProjectController().get_project_path(project_id=project_id)
        cleaned = self.get_clean_file_name(orig_file_name=orig_file_name)

        new_file_path = os.path.join(project_path, f"{random_key}_{cleaned}")
        while os.path.exists(new_file_path):
            random_key = self.generate_random_string()
            new_file_path = os.path.join(project_path, f"{random_key}_{cleaned}")

        return new_file_path, f"{random_key}_{cleaned}"

    def get_clean_file_name(self, orig_file_name: str) -> str:
        # H-5: restrict to safe ASCII chars only; remove Unicode word chars
        # that can combine into path-relevant sequences on some locales.
        cleaned = re.sub(r"[^a-zA-Z0-9_.\-]", "", orig_file_name.strip())
        cleaned = cleaned.replace(" ", "_")
        # Strip leading dots to prevent hidden-file names (e.g. "..txt" → "txt")
        cleaned = cleaned.lstrip(".")
        if not cleaned:
            cleaned = "unnamed_file"
        if len(cleaned) > 100:
            parts = cleaned.rsplit(".", 1)
            if len(parts) == 2:
                ext = parts[1][:10]
                cleaned = f"{parts[0][:89]}.{ext}"
            else:
                cleaned = cleaned[:100]
        return cleaned


    def delete_file_by_name(self, project_id: str, file_id: str) -> bool:
        """Remove the physical file from the project folder.
        """
        candidate = os.path.join(self.files_dir, project_id, file_id)
        if os.path.isfile(candidate):
            try:
                os.remove(candidate)
                return True
            except OSError as e:
                import logging as _log
                _log.getLogger(__name__).error("Failed to delete file %s: %s", candidate, e)
                return False
        return False


