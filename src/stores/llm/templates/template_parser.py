import os
import importlib
from functools import lru_cache


class TemplateParser:

    def __init__(self, language: str = None, default_language: str = "en"):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_language = default_language
        self.language = None
        self.set_language(language)

    def set_language(self, language: str):
        if not language:
            self.language = self.default_language
            return

        language_path = os.path.join(self.current_path, "locales", language)
        if os.path.exists(language_path):
            self.language = language
        else:
            self.language = self.default_language

    def get(self, group: str, key: str, vars: dict = {}):
        if not group or not key:
            return None

        template_text = self._load_template(group, key)
        if template_text is None:
            return None

        # Substitute variables safely — unknown keys left as-is
        try:
            return template_text.safe_substitute(vars)
        except Exception:
            return template_text.substitute(vars)

    def clear_cache(self):
        """Clear the template LRU cache (call after editing template files at runtime)."""
        self._load_template.cache_clear()

    @lru_cache(maxsize=128)
    def _load_template(self, group: str, key: str):
        """Load and cache a template attribute from disk.

        Results are cached so each group/key pair is only imported once.
        Call clear_cache() to invalidate after file changes.
        """
        for lang in (self.language, self.default_language):
            group_path = os.path.join(
                self.current_path, "locales", lang, f"{group}.py"
            )
            if not os.path.exists(group_path):
                continue

            module_name = f"stores.llm.templates.locales.{lang}.{group}"
            try:
                # import_module + reload ensures file changes are picked up
                module = importlib.import_module(module_name)
                importlib.reload(module)
                attr = getattr(module, key, None)
                if attr is not None:
                    return attr
            except Exception:
                continue

        return None
