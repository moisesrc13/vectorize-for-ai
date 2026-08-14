import json
import logging
from pathlib import Path
from typing import Optional

from vectorize_for_ai.config import gdrive_settings

logger = logging.getLogger(__name__)


class GdriveStateManager:
    """Tracks last processed timestamp so we only fetch new documents."""

    def __init__(self, state_file: Path = gdrive_settings.drive_state_file):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {"last_createdTime": None, "processed_ids": []}

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_since_date(self, override: Optional[str] = None) -> Optional[str]:
        """Returns the date to filter from. Priority: override > saved state > None."""
        if override:
            return override
        return self.state.get("last_createdTime")

    def update_timestamp(self, timestamp: str):
        """Update the last processed timestamp."""
        self.state["last_createdTime"] = timestamp
        self.save()

    def is_processed(self, file_id: str) -> bool:
        """Check if we already processed this file (optional dedup)."""
        return file_id in self.state.get("processed_ids", [])

    def mark_processed(self, file_id: str):
        """Mark file as processed. Keeps last 1000 IDs to prevent bloat."""
        processed = self.state.get("processed_ids", [])
        processed.append(file_id)
        self.state["processed_ids"] = processed[-1000:]  # Keep last 1000
        self.save()
