"""
BigOcrPdf - OCR Processor Module

This module handles PDF OCR using RapidOCR with the unified PP-OCRv6 model.
"""

import atexit
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from bigocrpdf.services.rapidocr_service import (
    ModelDiscovery,
    OCRConfig,
    ProcessingStats,
    RapidOCREngine,
)
from bigocrpdf.services.rapidocr_service.ocr_document_io import publish_ocr_pdfs
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.utils.checkpoint_manager import CheckpointManager, get_checkpoint_manager
from bigocrpdf.utils.history_manager import HistoryManager, get_history_manager
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger


class OcrProcessor:
    """Class to handle OCR processing tasks using RapidOCR.

    This processor uses RapidOCR with the unified PP-OCRv6 model.
    """

    def __init__(
        self,
        settings: OcrSettings,
        checkpoint_manager: CheckpointManager | None = None,
        history_manager: HistoryManager | None = None,
    ) -> None:
        """Initialize the OCR processor.

        Args:
            settings: The OcrSettings object containing processing settings
        """
        self.settings = settings
        self._checkpoint_manager = checkpoint_manager
        self._history_manager = history_manager
        self.on_file_complete: Callable[[str, str, str, list], None] | None = None
        self.on_all_complete: Callable[[], None] | None = None
        self.on_progress: Callable[[int, int, str], None] | None = None

        # Processing state
        self._is_processing = False
        self._processing_started = False
        self._stop_requested = False
        self._processing_thread: threading.Thread | None = None

        self._total_pages = 0
        self._total_files_at_start = 0
        self._completed_input_count = 0
        self._successful_input_count = 0

        # Progress tracking (simplified)
        self._file_progress = 0.0  # Progress within current file (0.0-1.0)
        self._current_status = ""  # Status message from backend
        self._current_filename = ""  # Current file being processed
        self._current_engine: RapidOCREngine | None = None
        self._engine_lock = threading.Lock()  # Protects _current_engine
        self._state_lock = threading.Lock()  # Protects progress/status fields
        self._run_lock = threading.Lock()  # Protects worker creation and ownership
        self._idle_callbacks: list[Callable[[], None]] = []

        # Model discovery
        self._discovery = ModelDiscovery()

        # Register cleanup for abnormal termination
        atexit.register(self._atexit_cleanup)

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Resolve the shared checkpoint store only when first needed."""
        if self._checkpoint_manager is None:
            self._checkpoint_manager = get_checkpoint_manager()
        return self._checkpoint_manager

    @property
    def history_manager(self) -> HistoryManager:
        """Resolve the shared history store only when first needed."""
        if self._history_manager is None:
            self._history_manager = get_history_manager()
        return self._history_manager

    def process_with_api(self) -> bool:
        """Process selected files using RapidOCR.

        Returns:
            True if processing started successfully, False otherwise
        """
        session_started = False
        try:
            with self._run_lock:
                if self._processing_thread is not None and self._processing_thread.is_alive():
                    logger.warning("OCR processing is already active")
                    return False
                if not self._validate_input_files():
                    return False

                self._setup_processing()
                session_started = True

                # Start processing in a background thread (non-daemon so
                # cleanup runs on exit instead of abrupt kill)
                self._processing_thread = threading.Thread(
                    target=self._process_all_files,
                )
                self._processing_thread.start()

            file_count = len(self.settings.selected_files)
            logger.info(
                ngettext(
                    "Started OCR processing for {count} file using RapidOCR",
                    "Started OCR processing for {count} files using RapidOCR",
                    file_count,
                ).format(count=file_count)
            )
            return True

        except MemoryError:
            if session_started:
                self._rollback_failed_start()
            raise

        except Exception as e:
            if session_started:
                self._rollback_failed_start()
            logger.error(_("Error starting OCR processing: {0}").format(str(e)))
            return False

    def _rollback_failed_start(self) -> None:
        """Undo state published before a processing thread failed to start."""
        self._is_processing = False
        self._processing_started = False
        self._processing_thread = None
        self.checkpoint_manager.discard_session()

    def _validate_input_files(self) -> bool:
        """Validate that we have files to process."""
        if not self.settings.selected_files:
            logger.error(_("No files to process"))
            return False
        return True

    def _setup_processing(self) -> None:
        """Reset state and prepare for a new processing run."""
        self._is_processing = True
        self._processing_started = True
        self._stop_requested = False
        with self._engine_lock:
            self._current_engine = None
        self._total_pages = 0
        with self._state_lock:
            self._file_progress = 0.0
            self._current_status = ""
            self._current_filename = ""
            self._completed_input_count = 0
            self._successful_input_count = 0
        # Store original file count before processing removes files from queue
        self._total_files_at_start = len(self.settings.selected_files)
        self.settings.processed_files = []

        # Start checkpoint session for crash recovery
        checkpoint = self.checkpoint_manager
        settings_snapshot = {
            "ocr_language": self.settings.ocr_language,
            "dpi": self.settings.dpi,
            "destination_folder": self.settings.destination_folder,
            "save_in_same_folder": self.settings.save_in_same_folder,
        }
        checkpoint.start_session(
            self.settings.selected_files,
            settings_snapshot,
            self.settings.file_modifications,
        )

    def _process_all_files(self) -> None:
        """Process all selected files in sequence."""
        try:
            files_to_process = list(self.settings.selected_files)
            checkpoint = self.checkpoint_manager

            for i, file_path in enumerate(files_to_process):
                if self._stop_requested:
                    logger.info(_("Processing stopped by user"))
                    break

                if not self._process_file_with_checkpoint(file_path, i, checkpoint):
                    break

            self._finish_processing_run(checkpoint)

        except Exception as e:
            logger.error(_("Processing thread error: {0}").format(e))
            self._is_processing = False
        finally:
            current_thread = threading.current_thread()
            with self._run_lock:
                if self._processing_thread is current_thread:
                    self._processing_thread = None
                idle_callbacks = self._idle_callbacks
                self._idle_callbacks = []
            for callback in idle_callbacks:
                callback()

    def _process_file_with_checkpoint(self, file_path: str, index: int, checkpoint) -> bool:
        self._prepare_current_file(file_path)

        try:
            success, extracted_text, ocr_boxes, output_file = self._process_single_file(
                file_path,
                index,
            )
        except InterruptedError:
            logger.info(_("Processing cancelled by user during file: {0}").format(file_path))
            return False
        except MemoryError:
            logger.critical(_("Out of memory processing {0}").format(file_path))
            raise
        except Exception as e:
            self._record_terminal_input(success=False)
            self._record_file_processing_error(file_path, checkpoint, e)
            return True

        self._reset_file_progress()
        self._record_terminal_input(success=success)
        self._record_file_processing_result(
            file_path,
            output_file,
            success,
            extracted_text,
            ocr_boxes,
            checkpoint,
        )
        return True

    def _prepare_current_file(self, file_path: str) -> None:
        with self._state_lock:
            self._current_filename = self.settings.display_name(file_path)
            self._file_progress = 0.0

    def _reset_file_progress(self) -> None:
        with self._state_lock:
            self._file_progress = 0.0

    def _record_terminal_input(self, *, success: bool) -> None:
        """Record one non-cancelled input after it reaches a terminal state."""
        with self._state_lock:
            self._completed_input_count += 1
            if success:
                self._successful_input_count += 1

    def _record_file_processing_result(
        self,
        file_path: str,
        output_file: str,
        success: bool,
        extracted_text: str,
        ocr_boxes: list,
        checkpoint,
    ) -> None:
        if success:
            checkpoint.mark_file_completed(file_path, output_file)
            if self.on_file_complete:
                self.on_file_complete(file_path, output_file, extracted_text, ocr_boxes)
        else:
            checkpoint.mark_file_failed(file_path, "Processing failed")

    def _record_file_processing_error(self, file_path: str, checkpoint, error: Exception) -> None:
        logger.error(_("Error processing {0}: {1}").format(file_path, error))
        checkpoint.mark_file_failed(file_path, str(error))
        if self.on_file_complete:
            self.on_file_complete(file_path, "", f"error: {error}", [])

    def _finish_processing_run(self, checkpoint) -> None:
        self._is_processing = False

        if not self._stop_requested:
            checkpoint.complete_session()

        if self.on_all_complete:
            self.on_all_complete()

    def _process_single_file(
        self,
        file_path: str,
        index: int,
    ) -> tuple[bool, str, list, str]:
        """Process a single file with OCR.

        Returns:
            Tuple of (success, extracted_text, ocr_boxes, primary_output)
        """
        if not file_path or not os.path.exists(file_path):
            logger.error(_("Error: File not found or invalid: {0}").format(file_path))
            return False, "", [], ""

        requested_output = self._get_output_file_path(file_path, index)
        if not requested_output:
            return False, "", [], ""

        # Create OCR config from settings, including file-specific page range
        config = self._create_ocr_config(file_path)

        # Create engine and process
        engine = RapidOCREngine(config)
        with self._engine_lock:
            self._current_engine = engine

        requested_output_path = Path(requested_output)
        with tempfile.TemporaryDirectory(
            prefix=".bigocr_stage_",
            dir=requested_output_path.parent,
        ) as staging_dir:
            staged_output = Path(staging_dir) / requested_output_path.name
            stats = engine.process(
                Path(file_path),
                staged_output,
                self._make_progress_callback(),
            )
            success = stats.pages_processed > 0
            output_files = self._publish_successful_outputs(
                success,
                staged_output,
                requested_output_path,
                stats,
            )

        self._total_pages += stats.pages_total

        logger.info(
            _("Processed {filename}: Pages: {pages} · Words: {words}").format(
                filename=os.path.basename(file_path),
                pages=stats.pages_processed,
                words=stats.total_words,
            )
        )

        extracted_text = stats.full_text or ""
        ocr_boxes = stats.ocr_boxes

        output_file = self._track_successful_outputs(success, output_files)
        self._record_processing_history(file_path, output_file, stats, success)

        return success, extracted_text, ocr_boxes, output_file

    def _publish_successful_outputs(
        self,
        success: bool,
        staged_output: Path,
        requested_output: Path,
        stats: ProcessingStats,
    ) -> list[str]:
        if not success:
            return []

        overwrite = self.settings.overwrite_existing
        if stats.split_output_files:
            published_parts = publish_ocr_pdfs(
                [
                    (Path(part), requested_output.parent / Path(part).name)
                    for part in stats.split_output_files
                ],
                overwrite=overwrite,
                family_root=requested_output,
            )
            stats.split_output_files = [str(part) for part in published_parts]
            return stats.split_output_files

        published_output = publish_ocr_pdfs(
            [(staged_output, requested_output)],
            overwrite=overwrite,
            family_root=requested_output,
        )[0]
        return [str(published_output)]

    def _make_progress_callback(self) -> Callable[[int, int, str], None]:
        def progress_callback(current: int, total: int, message: str) -> None:
            with self._state_lock:
                self._file_progress = current / 100.0 if total > 0 else 0.0
                self._current_status = message
            if self.on_progress:
                self.on_progress(current, total, message)

        return progress_callback

    def _track_successful_outputs(
        self,
        success: bool,
        output_files: list[str],
    ) -> str:
        if not success:
            return ""

        for output_file in output_files:
            if output_file not in self.settings.processed_files:
                self.settings.processed_files.append(output_file)
        return output_files[0]

    def _record_processing_history(
        self,
        file_path: str,
        output_file: str,
        stats: ProcessingStats,
        success: bool,
    ) -> None:
        self.history_manager.add_entry(
            input_path=file_path,
            output_path=output_file,
            pages_processed=stats.pages_processed,
            processing_time_seconds=stats.processing_time_seconds,
            language=self.settings.ocr_language,
            success=success,
        )

    def _create_ocr_config(self, file_path: str | None = None) -> OCRConfig:
        """Create OCR configuration from settings.

        Args:
            file_path: Optional file path to look up file-specific settings (e.g., page range)
        """
        # Map settings to OCRConfig
        language = self.settings.ocr_language
        dpi = self.settings.dpi

        # Get file-specific page range if available
        page_range = None
        if file_path:
            page_range = self.settings.page_ranges.get(file_path)

        # Get file-specific modifications
        page_modifications = None
        if file_path:
            resolved = os.path.realpath(file_path)
            # Try resolved path first, then original path
            state_dict = self.settings.file_modifications.get(
                resolved
            ) or self.settings.file_modifications.get(file_path)
            if not state_dict:
                # Try matching by resolved path against stored keys
                found_key = next(
                    (
                        k
                        for k in self.settings.file_modifications
                        if os.path.realpath(k) == resolved
                    ),
                    None,
                )
                if found_key:
                    state_dict = self.settings.file_modifications[found_key]

            if state_dict and "pages" in state_dict:
                page_modifications = state_dict["pages"]

        config = self.settings._snapshot_ocr_config()
        return replace(
            config,
            language=language,
            dpi=dpi,
            page_range=page_range,
            page_modifications=page_modifications,
            force_full_ocr=bool(file_path and file_path in self.settings.original_file_paths),
        )

    def _get_output_file_path(self, file_path: str, index: int) -> str | None:
        """Determine the output file path for a processed file.

        If the file was edited (merged by editor), uses the original file's
        name and directory for the output path.
        """
        try:
            # Resolve original path for edited files (editor creates temp files in /tmp)
            original_path = self.settings.original_file_paths.get(file_path, file_path)
            input_filename = os.path.basename(original_path)
            base_name = os.path.splitext(input_filename)[0]

            output_dir = self._get_output_directory(original_path)
            if not output_dir:
                logger.error(_("Could not determine output directory for {0}").format(file_path))
                return None

            output_file = self._create_output_file_path(output_dir, base_name, index)

            return output_file
        except Exception as e:
            logger.error(_("Error creating output path for {0}: {1}").format(file_path, e))
            return None

    def _get_output_directory(self, file_path: str) -> str | None:
        """Determine the output directory for a processed file."""
        if self.settings.save_in_same_folder:
            return os.path.dirname(file_path)
        elif self.settings.destination_folder:
            os.makedirs(self.settings.destination_folder, exist_ok=True)
            return self.settings.destination_folder
        else:
            return os.path.dirname(file_path)

    def _create_output_file_path(self, output_dir: str, base_name: str, index: int) -> str:
        """Create the output file path based on settings."""
        use_original = self.settings.use_original_filename

        if use_original:
            return os.path.join(output_dir, f"{base_name}.pdf")
        else:
            suffix = self.settings.get_pdf_suffix() or "ocr"
            if index == 0:
                return os.path.join(output_dir, f"{base_name}-{suffix}.pdf")
            else:
                return os.path.join(output_dir, f"{base_name}-{suffix}-{index + 1}.pdf")

    def get_available_ocr_languages(self) -> list[tuple[str, str]]:
        """Report availability of the unified PP-OCRv6 model.

        Returns:
            List of tuples (language_code, display_name)
        """
        try:
            return self._discovery.get_available_languages()

        except Exception as e:
            logger.error(_("Error getting OCR languages: {0}").format(e))
            return []

    def get_progress(self) -> float:
        """Get the current OCR processing progress.

        Returns:
            Float between 0.0 and 1.0 representing completion percentage
        """
        if not self._processing_started:
            return 0.0
        elif not self._is_processing:
            return 1.0

        total_files = self._total_files_at_start
        if total_files == 0:
            return 0.0

        with self._state_lock:
            completed_files = self._completed_input_count
            current_file_contribution = self._file_progress / total_files

        # Calculate progress: completed inputs + current input progress.
        base_progress = completed_files / total_files
        return min(1.0, base_progress + current_file_contribution)

    def get_processed_count(self) -> int:
        """Get the successful input count retained for API compatibility."""
        return self.get_successful_input_count()

    def get_completed_input_count(self) -> int:
        """Get the number of non-cancelled inputs that reached a terminal state."""
        with self._state_lock:
            return self._completed_input_count

    def get_successful_input_count(self) -> int:
        """Get the number of inputs that produced at least one output PDF."""
        with self._state_lock:
            return self._successful_input_count

    def get_total_count(self) -> int:
        """Get the total number of files to process.

        During processing, this returns the count from start to avoid
        incorrect counts as files are removed from the queue.
        """
        if self._total_files_at_start > 0:
            return self._total_files_at_start
        return len(self.settings.selected_files) if self.settings.selected_files else 0

    def get_total_pages(self) -> int:
        """Get the total number of pages processed."""
        return self._total_pages

    def get_current_file_info(self) -> dict[str, Any]:
        """Get information about the currently processing file."""
        if not self._is_processing:
            return {}

        with self._state_lock:
            return {
                "filename": self._current_filename,
                "file_number": self._completed_input_count + 1,
                "total_files": self._total_files_at_start,
                "status_message": self._current_status,
                "file_progress": self._file_progress,
            }

    def register_callbacks(
        self,
        on_file_complete: Callable[[str, str, str, list], None] | None = None,
        on_all_complete: Callable[[], None] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Register callbacks for OCR processing events."""
        self.on_file_complete = on_file_complete
        self.on_all_complete = on_all_complete
        self.on_progress = on_progress

    def remove_processed_file(self, input_file: str) -> None:
        """Remove a processed file from the selected files list."""
        if self.settings._remove_file(input_file):
            logger.info(
                _("Removed processed file from queue: {0}").format(os.path.basename(input_file))
            )

    def force_cleanup(self) -> None:
        """Force cleanup of all resources and stop processing.

        This method is NON-BLOCKING — it signals the backend to stop and
        returns immediately so the GTK main thread remains responsive.
        The processing thread will exit on its own after finishing the
        current page.
        """
        try:
            self._stop_requested = True

            # Signal the backend to stop between pages
            with self._engine_lock:
                if self._current_engine is not None:
                    self._current_engine.cancel_event.set()
                    logger.info("Cancel event set on current OCR engine")

            self._is_processing = False
            self._processing_started = False

            # CRITICAL: Clear callbacks to prevent stale callbacks being called
            # after the user has already navigated away
            self.on_file_complete = None
            self.on_all_complete = None
            self.on_progress = None

            logger.info("OCR processor cleanup completed (non-blocking)")

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def _atexit_cleanup(self) -> None:
        """Cleanup handler for abnormal termination (atexit)."""
        self._stop_requested = True
        with self._engine_lock:
            if self._current_engine is not None:
                self._current_engine.cancel_event.set()
        t = self._processing_thread
        if t is not None and t.is_alive():
            t.join(timeout=5)

    def is_processing(self) -> bool:
        """Check if processing is currently active."""
        return self._is_processing

    def has_active_worker(self) -> bool:
        """Return whether a previous processing thread still owns the processor."""
        with self._run_lock:
            return self._processing_thread is not None and self._processing_thread.is_alive()

    def run_when_idle(self, callback: Callable[[], None]) -> None:
        """Run a callback after the current worker releases shared state."""
        with self._run_lock:
            if self._processing_thread is not None and self._processing_thread.is_alive():
                self._idle_callbacks.append(callback)
                return
        callback()

    def has_resumable_session(self) -> bool:
        """Check if there's a previous incomplete session that can be resumed.

        Returns:
            True if an incomplete session exists with pending files
        """
        return self.checkpoint_manager.has_incomplete_session()

    def get_resumable_session_info(self) -> dict[str, Any] | None:
        """Get information about a resumable session.

        Returns:
            Dictionary with session info, or None if no session to resume
        """
        return self.checkpoint_manager.get_incomplete_session_info()

    def resume_previous_session(self) -> bool:
        """Resume processing from an incomplete session.

        This restores the list of pending files from the checkpoint and
        optionally restores relevant settings.

        Returns:
            True if session was resumed, False if nothing to resume
        """
        result = self.checkpoint_manager.resume_session()

        if not result:
            return False

        pending_files, settings_snapshot = result

        # Set pending files as selected files
        self.settings.selected_files = list(pending_files)
        restored_modifications = self.checkpoint_manager.get_file_modifications()
        self.settings.file_modifications = {
            path: modifications
            for path, modifications in restored_modifications.items()
            if path in pending_files
        }

        # Restore settings from snapshot if available
        if settings_snapshot:
            if "ocr_language" in settings_snapshot:
                self.settings.ocr_language = settings_snapshot["ocr_language"]
            if "dpi" in settings_snapshot:
                self.settings.dpi = settings_snapshot["dpi"]
            if "destination_folder" in settings_snapshot:
                self.settings.destination_folder = settings_snapshot["destination_folder"]
            if "save_in_same_folder" in settings_snapshot:
                self.settings.save_in_same_folder = settings_snapshot["save_in_same_folder"]

        logger.info(f"Resumed session with {len(pending_files)} pending files")
        return True

    def discard_previous_session(self) -> bool:
        """Discard a previous incomplete session.

        Returns:
            True if session was discarded
        """
        return self.checkpoint_manager.discard_session()
