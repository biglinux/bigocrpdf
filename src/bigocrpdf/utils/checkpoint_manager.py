"""
BigOcrPdf - Checkpoint Manager Module

Provides checkpoint/resume functionality for OCR processing.
Saves progress after each file is processed and allows resuming
interrupted batch processing sessions.
"""

import copy
import json
import math
import os
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bigocrpdf.utils.durable_writes import write_text_file_atomically
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
        file_errors: Mapping of failed input files to their last error
        file_modifications: Serialized editor state keyed by input file
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
    file_errors: dict[str, str] = field(default_factory=dict)
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

    def start_session(
        self,
        files: list[str],
        settings: dict[str, Any] | None = None,
        file_modifications: dict[str, Any] | None = None,
    ) -> str:
        """Start a new processing session and create initial checkpoint.

        Args:
            files: List of file paths to process
            settings: Optional settings snapshot to preserve

        Returns:
            Session ID for tracking
        """
        # Generate unique session ID
        session_id = f"session_{int(time.time() * 1000)}"

        unique_files = list(dict.fromkeys(files))
        self._current_checkpoint = CheckpointData(
            session_id=session_id,
            files_to_process=unique_files,
            files_completed=[],
            files_failed=[],
            output_files={},
            file_errors={},
            file_modifications=copy.deepcopy(file_modifications) if file_modifications else {},
            settings_snapshot=copy.deepcopy(settings) if settings else {},
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

        checkpoint = self._current_checkpoint
        if checkpoint.is_complete:
            logger.warning("Cannot update a completed checkpoint session")
            return
        if input_file not in checkpoint.files_to_process:
            logger.warning(f"Cannot complete unqueued checkpoint file: {input_file}")
            return
        if (
            input_file in checkpoint.files_completed
            and input_file not in checkpoint.files_failed
            and checkpoint.output_files.get(input_file) == output_file
            and input_file not in checkpoint.file_errors
        ):
            return

        checkpoint.files_failed = [path for path in checkpoint.files_failed if path != input_file]
        checkpoint.file_errors.pop(input_file, None)
        if input_file not in checkpoint.files_completed:
            checkpoint.files_completed.append(input_file)
        checkpoint.output_files[input_file] = output_file
        checkpoint.last_update = time.time()

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

        checkpoint = self._current_checkpoint
        if checkpoint.is_complete:
            logger.warning("Cannot update a completed checkpoint session")
            return
        if input_file not in checkpoint.files_to_process:
            logger.warning(f"Cannot fail unqueued checkpoint file: {input_file}")
            return
        current_error = checkpoint.file_errors.get(input_file, "")
        if (
            input_file in checkpoint.files_failed
            and input_file not in checkpoint.files_completed
            and input_file not in checkpoint.output_files
            and current_error == error
        ):
            return

        checkpoint.files_completed = [
            path for path in checkpoint.files_completed if path != input_file
        ]
        checkpoint.output_files.pop(input_file, None)
        if input_file not in checkpoint.files_failed:
            checkpoint.files_failed.append(input_file)
        if error:
            checkpoint.file_errors[input_file] = error
        else:
            checkpoint.file_errors.pop(input_file, None)
        checkpoint.last_update = time.time()

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
        if input_file not in self._current_checkpoint.files_to_process:
            logger.warning(f"Cannot save modifications for unqueued file: {input_file}")
            return
        if self._current_checkpoint.file_modifications.get(input_file) == modifications:
            return
        self._current_checkpoint.file_modifications[input_file] = copy.deepcopy(modifications)
        self._current_checkpoint.last_update = time.time()
        self._save_checkpoint()

    def complete_session(self) -> None:
        """Mark the current session as successfully completed."""
        if not self._current_checkpoint:
            return

        if self._current_checkpoint.is_complete:
            return
        pending = self._get_pending_files(self._current_checkpoint)
        if pending:
            logger.warning(f"Cannot complete checkpoint with {len(pending)} pending files")
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
            "file_errors": dict(checkpoint.file_errors),
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

        return pending, copy.deepcopy(checkpoint.settings_snapshot)

    def get_file_modifications(self) -> dict[str, Any]:
        """Return a defensive copy of editor state from the active checkpoint."""
        if not self._current_checkpoint:
            return {}
        return copy.deepcopy(self._current_checkpoint.file_modifications)

    def discard_session(self) -> bool:
        """Discard the incomplete session checkpoint.

        Returns:
            True if a session was discarded
        """
        if (
            self._current_checkpoint is None
            and not self._checkpoint_path.exists()
            and not self._checkpoint_path.is_symlink()
        ):
            return False

        try:
            self._checkpoint_path.unlink(missing_ok=True)
        except OSError as e:
            logger.error(f"Failed to discard checkpoint: {e}")
            return False

        self._current_checkpoint = None
        logger.info("Checkpoint session discarded")
        return True

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

            write_text_file_atomically(
                self._checkpoint_path,
                lambda checkpoint_file: json.dump(
                    checkpoint_dict,
                    checkpoint_file,
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> CheckpointData | None:
        """Load checkpoint from disk.

        Returns:
            CheckpointData if valid checkpoint exists, None otherwise
        """
        descriptor = -1
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = os.open(self._checkpoint_path, flags)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("checkpoint must be a regular file")
            with os.fdopen(descriptor, encoding="utf-8") as checkpoint_file:
                descriptor = -1
                data = json.load(checkpoint_file)

            return _checkpoint_from_document(data)

        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.error(f"Invalid checkpoint file: {e}")
            return None
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _checkpoint_from_document(document: object) -> CheckpointData:
    """Validate and migrate a checkpoint JSON document."""
    if not isinstance(document, dict):
        raise ValueError("checkpoint root must be an object")

    session_id = document.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("checkpoint session_id must be a non-empty string")

    files_to_process = _string_list(document.get("files_to_process"), "files_to_process")
    files_completed = _string_list(document.get("files_completed", []), "files_completed")
    files_failed = _string_list(document.get("files_failed", []), "files_failed")
    queued_files = set(files_to_process)
    if not set(files_completed).issubset(queued_files):
        raise ValueError("completed checkpoint files must belong to the queue")
    if not set(files_failed).issubset(queued_files):
        raise ValueError("failed checkpoint files must belong to the queue")

    output_files = _string_mapping(document.get("output_files", {}), "output_files")
    file_errors = _string_mapping(document.get("file_errors", {}), "file_errors")
    file_modifications = _object_mapping(
        document.get("file_modifications", {}), "file_modifications"
    )
    settings_snapshot = _object(document.get("settings_snapshot", {}), "settings_snapshot")

    # Legacy writers could leave a file in both lists; an output mapping is
    # the only durable evidence that the successful transition happened last.
    conflicts = set(files_completed) & set(files_failed)
    files_completed = [
        path for path in files_completed if path not in conflicts or path in output_files
    ]
    files_failed = [
        path for path in files_failed if path not in conflicts or path not in output_files
    ]
    completed_set = set(files_completed)
    failed_set = set(files_failed)
    output_files = {path: output for path, output in output_files.items() if path in completed_set}
    file_errors = {path: error for path, error in file_errors.items() if path in failed_set}
    file_modifications = {
        path: modifications
        for path, modifications in file_modifications.items()
        if path in queued_files
    }

    start_time = _number(document.get("start_time", 0.0), "start_time")
    last_update = _number(document.get("last_update", 0.0), "last_update")
    is_complete = document.get("is_complete", False)
    if type(is_complete) is not bool:
        raise ValueError("checkpoint is_complete must be a boolean")
    processed = completed_set | failed_set
    if is_complete and any(path not in processed for path in files_to_process):
        is_complete = False

    return CheckpointData(
        session_id=session_id,
        files_to_process=files_to_process,
        files_completed=files_completed,
        files_failed=files_failed,
        output_files=output_files,
        file_errors=file_errors,
        file_modifications=file_modifications,
        settings_snapshot=settings_snapshot,
        start_time=start_time,
        last_update=last_update,
        is_complete=is_complete,
    )


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"checkpoint {field_name} must be a list of strings")
    return list(dict.fromkeys(value))


def _string_mapping(value: object, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"checkpoint {field_name} must map strings to strings")
    return dict(value)


def _object_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"checkpoint {field_name} must be an object with string keys")
    return dict(value)


def _object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {field_name} must be an object")
    return dict(value)


def _number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"checkpoint {field_name} must be a non-negative number")
    return float(value)


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
