from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models import ResponseSignal
import re
import os

# Map known MIME types to their canonical extension
_MIME_TO_EXT = {
    "text/plain": "txt",
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


class DataController(BaseController):

    def __init__(self):
        super().__init__()

    def validate_uploaded_file(self, file: UploadFile):
        """Return ``(is_valid, signal)`` for the supplied file.

        Checks both the file extension AND MIME type against the allow-list.
        FILE_MAX_SIZE in .env is already in bytes — no extra scaling applied.
        """
        allowed: list = self.app_settings.FILE_ALLOWED_TYPES or []

        # --- extension check ------------------------------------------------
        parts = (file.filename or "").rsplit(".", 1)
        ext = parts[-1].lower() if len(parts) == 2 else ""

        # --- MIME check (normalise to extension) ----------------------------
        raw_mime = (file.content_type or "").lower().split(";")[0].strip()
        mime_ext = _MIME_TO_EXT.get(raw_mime, raw_mime)

        if ext not in allowed and mime_ext not in allowed:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        # --- size check (FILE_MAX_SIZE is already bytes in .env) ------------
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
        # Keep only word characters and dots; replace spaces with underscore
        cleaned = re.sub(r"[^\w.]", "", orig_file_name.strip())
        return cleaned.replace(" ", "_")
