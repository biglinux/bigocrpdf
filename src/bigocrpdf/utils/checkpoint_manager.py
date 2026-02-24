"""
BigOcrPdf - Checkpoint Manager Module

Provides checkpoint/resume functionality for OCR processing.
Saves progress after each file is processed and allows resuming
interrupted batch processing sessions.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bigocrpdf.utils.logger import logger

# Default checkpoint directory (XDG compliant)
CHECKPOINT_DIR = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "bigocrpdf"
)
CHECKPOINT_FILE = "checkpoint.json"


@dataclass
class CheckpointData:
    """Data structure for checkpoint state.

    Attributes:
        session_id: Unique identifier for this processing session
        files_to_process: Original list of files queued for processing
        files_completed: List of files that have been successfully processed
        files_failed: List of files that failed processing
        output_files: Mapping of input files to their output files
        settings_snapshot: Copy of relevant settings at session start
        start_time: Unix timestamp when processing started
        last_update: Unix timestamp of last checkpoint update
        is_complete: Whether the session finished normally
    """

    session_id: str = ""
    files_to_process: list[str] = field(default_factory=list)
    files_completed: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
    file_modifications: dict[str, Any] = field(default_factory=dict)
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    last_update: float = 0.0
    is_complete: bool = False


class CheckpointManager:
    """Manages checkpoint state for OCR processing sessions.

    The checkpoint system allows users to resume processing after
    crashes, power failures, or intentional interruptions.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Custom directory for checkpoint files.
                           Uses XDG_STATE_HOME/bigocrpdf by default.
        """
        self._checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self._checkpoint_path = self._checkpoint_dir / CHECKPOINT_FILE
        self._current_checkpoint: CheckpointData | None = None

        # Ensure directory exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def start_session(self, files: list[str], settings: dict[str, Any] | None = None) -> str:
        """Start a new processing session and create initial checkpoint.

        Args:
            files: List of file paths to process
            settings: Optional settings snapshot to preserve

        Returns:
            Session ID for tracking
        """
        # Generate unique session ID
        session_id = f"session_{int(time.time() * 1000)}"

        self._current_checkpoint = CheckpointData(
            session_id=session_id,
            files_to_process=list(files),
            files_completed=[],
            files_failed=[],
            output_files={},
            settings_snapshot=settings or {},
            start_time=time.time(),
            last_update=time.time(),
            is_complete=False,
        )

        self._save_checkpoint()
        logger.info(f"Started checkpoint session: {session_id}")

        return session_id

    def mark_file_completed(self, input_file: str, output_file: str) -> None:
        """Mark a file as successfully processed.

        Args:
            input_file: Path to the input file
            output_file: Path to the generated output file
        """
        if not self._current_checkpoint:
            logger.warning("No active checkpoint session")
            return

        if input_file not in self._current_checkpoint.files_completed:
            self._current_checkpoint.files_completed.append(input_file)

        self._current_checkpoint.output_files[input_file] = output_file
        self._current_checkpoint.last_update = time.time()

        self._save_checkpoint()
        logger.debug(f"Checkpoint: marked completed - {os.path.basename(input_file)}")

    def mark_file_failed(self, input_file: str, error: str = "") -> None:
        """Mark a file as failed processing.

        Args:
            input_file: Path to the input file
            error: Optional error message
        """
        if not self._current_checkpoint:
            logger.warning("No active checkpoint session")
            return

        if input_file not in self._current_checkpoint.files_failed:
            self._current_checkpoint.files_failed.append(input_file)

        self._current_checkpoint.last_update = time.time()

        self._save_checkpoint()
        logger.debug(f"Checkpoint: marked failed - {os.path.basename(input_file)}: {error}")

    def save_file_modifications(self, input_file: str, modifications: dict[str, Any]) -> None:
        """Save editor modifications for a file.

        Args:
            input_file: Path to the input file
            modifications: Serializable dict of editor state (e.g. from PDFDocument.to_dict())
        """
        if not self._current_checkpoint:
            return
        self._current_checkpoint.file_modifications[input_file] = modifications
        self._current_checkpoint.last_update = time.time()
        self._save_checkpoint()

    def complete_session(self) -> None:
        """Mark the current session as successfully completed."""
        if not self._current_checkpoint:
            return

        self._current_checkpoint.is_complete = True
        self._current_checkpoint.last_update = time.time()

        self._save_checkpoint()
        logger.info("Checkpoint session completed successfully")

    def has_incomplete_session(self) -> bool:
        """Check if there's an incomplete session that can be resumed.

        Returns:
            True if an incomplete session exists
        """
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            return False

        # Session is incomplete if not marked complete and has pending files
        if checkpoint.is_complete:
            return False

        pending = self._get_pending_files(checkpoint)
        return len(pending) > 0

    def get_incomplete_session_info(self) -> dict[str, Any] | None:
        """Get information about an incomplete session.

        Returns:
            Dictionary with session info, or None if no incomplete session
        """
        checkpoint = self._load_checkpoint()
        if not checkpoint or checkpoint.is_complete:
            return None

        pending = self._get_pending_files(checkpoint)
        if not pending:
            return None

        return {
            "session_id": checkpoint.session_id,
            "total_files": len(checkpoint.files_to_process),
            "completed_files": len(checkpoint.files_completed),
            "failed_files": len(checkpoint.files_failed),
            "pending_files": len(pending),
            "pending_file_list": pending,
            "start_time": checkpoint.start_time,
            "last_update": checkpoint.last_update,
            "settings": checkpoint.settings_snapshot,
        }

    def resume_session(self) -> tuple[list[str], dict[str, Any]] | None:
        """Resume an incomplete session.

        Returns:
            Tuple of (files_to_process, settings) or None if nothing to resume
        """
        checkpoint = self._load_checkpoint()
        if not checkpoint or checkpoint.is_complete:
            logger.info("No incomplete session to resume")
            return None

        pending = self._get_pending_files(checkpoint)
        if not pending:
            logger.info("All files already processed, nothing to resume")
            return None

        # Restore checkpoint as current
        self._current_checkpoint = checkpoint

        logger.info(f"Resuming session {checkpoint.session_id}: {len(pending)} files remaining")

        return pending, checkpoint.settings_snapshot

    def discard_session(self) -> bool:
        """Discard the incomplete session checkpoint.

        Returns:
            True if a session was discarded
        """
        if self._checkpoint_path.exists():
            try:
                self._checkpoint_path.unlink()
                self._current_checkpoint = None
                logger.info("Checkpoint session discarded")
                return True
            except OSError as e:
                logger.error(f"Failed to discard checkpoint: {e}")
                return False
        return False

    def _get_pending_files(self, checkpoint: CheckpointData) -> list[str]:
        """Get list of files that still need processing.

        Args:
            checkpoint: Checkpoint data to analyze

        Returns:
            List of file paths that haven't been processed
        """
        processed = set(checkpoint.files_completed) | set(checkpoint.files_failed)
        return [f for f in checkpoint.files_to_process if f not in processed]

    def _save_checkpoint(self) -> None:
        """Persist current checkpoint to disk."""
        if not self._current_checkpoint:
            return

        try:
            checkpoint_dict = asdict(self._current_checkpoint)

            # Atomic write: write to temp file then rename
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_dict, f, indent=2, ensure_ascii=False)

            temp_path.replace(self._checkpoint_path)

        except OSError as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> CheckpointData | None:
        """Load checkpoint from disk.

        Returns:
            CheckpointData if valid checkpoint exists, None otherwise
        """
        if not self._checkpoint_path.exists():
            return None

        try:
            with open(self._checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)

            # Validate required fields
            if "session_id" not in data or "files_to_process" not in data:
                logger.warning("Invalid checkpoint file, missing required fields")
                return None

            return CheckpointData(
                session_id=data.get("session_id", ""),
                files_to_process=data.get("files_to_process", []),
                files_completed=data.get("files_completed", []),
                files_failed=data.get("files_failed", []),
                output_files=data.get("output_files", {}),
                settings_snapshot=data.get("settings_snapshot", {}),
                start_time=data.get("start_time", 0.0),
                last_update=data.get("last_update", 0.0),
                is_complete=data.get("is_complete", False),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Checkpoint file corrupted: {e}")
            return None
        except OSError as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


# Module-level singleton instance
_checkpoint_manager_instance: CheckpointManager | None = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance.

    Returns:
        The singleton CheckpointManager instance
    """
    global _checkpoint_manager_instance
    if _checkpoint_manager_instance is None:
        _checkpoint_manager_instance = CheckpointManager()
    return _checkpoint_manager_instance
