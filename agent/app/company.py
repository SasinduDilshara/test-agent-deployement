"""Company metadata (non-HR) loaded from the company profile JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CompanyDirectory:
    """Reads company metadata (offices, leadership, departments, ...) from the company profile JSON."""

    def __init__(self, profile_path: Path) -> None:
        """Store the path to the company profile JSON file."""
        self.profile_path = profile_path

    def load(self) -> dict[str, Any]:
        """Load and return the whole company profile as a dict."""
        # Read on every call so edits to the JSON are picked up without a restart.
        with self.profile_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{self.profile_path} must contain a JSON object.")
        return data

    def sections(self) -> list[str]:
        """Return the top-level section names of the company profile."""
        return list(self.load().keys())

    def get(self, section: str | None = None) -> dict[str, Any]:
        """Return the whole profile, or only one top-level section; raises KeyError for unknown sections."""
        data = self.load()
        if not section:
            return data
        key = section.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in data:
            raise KeyError(f"Unknown section {section!r}. Available sections: {', '.join(data)}")
        return {key: data[key]}
