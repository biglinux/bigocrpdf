"""File queue and document state management."""

from __future__ import annotations

from typing import Any


class FileQueueManager:
    """File queue and document state management."""

    def __init__(self) -> None:
        self.selected_files: list[str] = []
        self.page_ranges: dict[str, tuple[int, int] | None] = {}
        self.processed_files: list[str] = []
        self.original_file_paths: dict[str, str] = {}
        self.file_modifications: dict[str, list[dict[str, Any]]] = {}
        self.pages_count: int = 0
