"""
BigOcrPdf - File History Manager

This module provides functionality to track and manage the history of processed files.
It stores metadata like file path, processing time, size before/after, etc.
"""

import json
import math
import os
import stat
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from bigocrpdf.config import CONFIG_DIR
from bigocrpdf.utils.durable_writes import write_text_file_atomically
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
    def from_dict(cls, data: object) -> "HistoryEntry":
        """Create an entry from a validated legacy or current dictionary."""
        if not isinstance(data, dict):
            raise TypeError("history entry must be an object")

        input_path = data.get("input_path")
        output_path = data.get("output_path")
        if not isinstance(input_path, str) or not isinstance(output_path, str):
            raise ValueError("history paths must be strings")

        timestamp = _history_number(data.get("timestamp", time.time()), "timestamp")
        input_size = _history_integer(data.get("input_size_bytes", 0), "input_size_bytes")
        output_size = _history_integer(data.get("output_size_bytes", 0), "output_size_bytes")
        pages = _history_integer(data.get("pages_processed", 0), "pages_processed")
        processing_time = _history_number(
            data.get("processing_time_seconds", 0.0), "processing_time_seconds"
        )
        language = data.get("language", "latin")
        success = data.get("success", True)
        error_message = data.get("error_message", "")
        if not isinstance(language, str) or not isinstance(error_message, str):
            raise ValueError("history language and error_message must be strings")
        if type(success) is not bool:
            raise ValueError("history success must be a boolean")

        return cls(
            input_path=input_path,
            output_path=output_path,
            timestamp=timestamp,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            pages_processed=pages,
            processing_time_seconds=processing_time,
            language=language,
            success=success,
            error_message=error_message,
        )


class HistoryManager:
    """Manages the file processing history."""

    def __init__(self) -> None:
        """Initialize the history manager."""
        self._entries: list[HistoryEntry] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load history from disk."""
        try:
            with open(
                HISTORY_FILE,
                encoding="utf-8",
                opener=lambda path, flags: os.open(
                    path,
                    flags
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                ),
            ) as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    raise OSError("history path is not a regular file")
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("history root must be an object")
            raw_entries = data.get("entries", [])
            if not isinstance(raw_entries, list):
                raise ValueError("history entries must be a list")

            entries: list[HistoryEntry] = []
            for index, raw_entry in enumerate(raw_entries):
                try:
                    entries.append(HistoryEntry.from_dict(raw_entry))
                except (TypeError, ValueError) as e:
                    logger.warning(f"Ignoring invalid history entry {index}: {e}")
            self._entries = entries[:MAX_HISTORY_ENTRIES]
            logger.debug(f"Loaded {len(self._entries)} history entries")
        except FileNotFoundError:
            self._entries = []
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.warning(f"Failed to load history: {e}")
            self._entries = []

    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            data = {"entries": [entry.to_dict() for entry in self._entries]}
            write_text_file_atomically(
                HISTORY_FILE,
                lambda history_file: json.dump(data, history_file, indent=2),
            )
            logger.debug(f"Saved {len(self._entries)} history entries")
        except (OSError, TypeError, ValueError) as e:
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
            input_size = os.path.getsize(input_path)
        except OSError:
            pass
        try:
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


def _history_integer(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"history {field_name} must be a non-negative integer")
    return value


def _history_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"history {field_name} must be a non-negative number")
    return float(value)


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
