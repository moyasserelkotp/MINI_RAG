from helpers.config import get_settings
import os
import random
import string


class BaseController:

    def __init__(self):
        # Re-use cached singleton — no repeated env parsing
        self.app_settings = get_settings()

        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.files_dir = os.path.join(self.base_dir, "assets", "files")
        self.database_dir = os.path.join(self.base_dir, "assets", "databases")

    def generate_random_string(self, length: int = 12) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def get_database_path(self, db_name: str) -> str:
        database_path = os.path.join(self.database_dir, db_name)
        os.makedirs(database_path, exist_ok=True)
        return database_path
