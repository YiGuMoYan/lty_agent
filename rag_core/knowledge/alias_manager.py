import json
import os
import re
from typing import Dict, Optional

class AliasManager:
    def __init__(self, alias_path: str = None):
        if alias_path is None:
            # Default to dataset/knowledge_base/aliases.json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alias_path = os.path.join(base_dir, "dataset", "knowledge_base", "aliases.json")

        self.alias_path = alias_path
        self.aliases: Dict[str, str] = {}
        self.load_aliases()

    def load_aliases(self):
        """Load aliases from JSON file"""
        if os.path.exists(self.alias_path):
            try:
                with open(self.alias_path, 'r', encoding='utf-8') as f:
                    self.aliases = json.load(f)
                print(f"[AliasManager] Loaded {len(self.aliases)} aliases")
            except Exception as e:
                print(f"[AliasManager] Error loading aliases: {e}")
        else:
            print(f"[AliasManager] Alias file not found at {self.alias_path}")

    def normalize(self, text: str) -> str:
        """
        Replace aliases in text with canonical names.
        Case-insensitive replacement.
        """
        if not text or not self.aliases:
            return text

        normalized_text = text
        # sort by length descending to replace longer phrases first (e.g. avoid partial match issues)
        sorted_keys = sorted(self.aliases.keys(), key=len, reverse=True)

        for alias in sorted_keys:
            canonical = self.aliases[alias]
            # Use regex for case-insensitive replacement with word boundaries if needed
            # For Chinese/mixed, word boundaries \b might not work well, so simple replace usually suffices
            # or custom regex escaping.

            # Simple case-insensitive replace
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            normalized_text = pattern.sub(canonical, normalized_text)

        return normalized_text
