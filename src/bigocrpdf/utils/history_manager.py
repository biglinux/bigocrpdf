"""
BigOcrPdf - File History Manager

This module provides functionality to track and manage the history of processed files.
It stores metadata like file path, processing time, size before/after, etc.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from bigocrpdf.config import CONFIG_DIR
from bigocrpdf.utils.logger import logger

# History configuration
HISTORY_FILE = os.path.join(CONFIG_DIR, "processing_history.json")
MAX_HISTORY_ENTRIES = 100  # Maximum number of entries to keep


@dataclass
class HistoryEntry:
    """Represents a single processed file entry in history."""

    input_path: str
    output_path: str
    timestamp: float = field(default_factory=time.time)
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    pages_processed: int = 0
    processing_time_seconds: float = 0.0
    language: str = "latin"
    success: bool = True
    error_message: str = ""

    @property
    def input_filename(self) -> str:
        """Get the input filename."""
        return os.path.basename(self.input_path)

    @property
    def input_size_mb(self) -> float:
        """Get the input size in MB."""
        return round(self.input_size_bytes / (1024 * 1024), 2)

    @property
    def output_size_mb(self) -> float:
        """Get the output size in MB."""
        return round(self.output_size_bytes / (1024 * 1024), 2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoryEntry":
        """Create an entry from a dictionary."""
        return cls(**data)


class HistoryManager:
    """Manages the file processing history."""

    def __init__(self) -> None:
        """Initialize the history manager."""
        self._entries: list[HistoryEntry] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load history from disk."""
        if not os.path.exists(HISTORY_FILE):
            self._entries = []
            return

        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
                self._entries = [HistoryEntry.from_dict(entry) for entry in data.get("entries", [])]
            logger.debug(f"Loaded {len(self._entries)} history entries")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load history: {e}")
            self._entries = []

    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            data = {"entries": [entry.to_dict() for entry in self._entries]}
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._entries)} history entries")
        except OSError as e:
            logger.error(f"Failed to save history: {e}")

    def add_entry(
        self,
        input_path: str,
        output_path: str,
        pages_processed: int = 0,
        processing_time_seconds: float = 0.0,
        language: str = "latin",
        success: bool = True,
        error_message: str = "",
    ) -> HistoryEntry:
        """Add a new entry to the history.

        Args:
            input_path: Path to the input file
            output_path: Path to the output file
            pages_processed: Number of pages processed
            processing_time_seconds: Total processing time
            language: OCR language used
            success: Whether processing was successful
            error_message: Error message if failed

        Returns:
            The created HistoryEntry
        """
        # Get file sizes
        input_size = 0
        output_size = 0
        try:
            if os.path.exists(input_path):
                input_size = os.path.getsize(input_path)
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
        except OSError:
            pass

        entry = HistoryEntry(
            input_path=input_path,
            output_path=output_path,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            pages_processed=pages_processed,
            processing_time_seconds=processing_time_seconds,
            language=language,
            success=success,
            error_message=error_message,
        )

        # Add to the beginning (most recent first)
        self._entries.insert(0, entry)

        # Trim to max size
        if len(self._entries) > MAX_HISTORY_ENTRIES:
            self._entries = self._entries[:MAX_HISTORY_ENTRIES]

        self._save_history()
        return entry

    @property
    def count(self) -> int:
        """Get the number of entries in history."""
        return len(self._entries)


# Global history manager instance
_history_manager: HistoryManager | None = None


def get_history_manager() -> HistoryManager:
    """Get the global history manager instance.

    Returns:
        HistoryManager instance
    """
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager
