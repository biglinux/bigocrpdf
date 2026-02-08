"""
BigOcrPdf - OCR Processor Module

This module handles the OCR processing of PDF files using RapidOCR with PP-OCRv5 models.
"""

import os
import threading
from collections.abc import Callable
from typing import Any

from bigocrpdf.services.rapidocr_service import (
    ModelDiscovery,
    OCRConfig,
    ProcessingStats,
    RapidOCREngine,
)
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.utils.checkpoint_manager import get_checkpoint_manager
from bigocrpdf.utils.history_manager import get_history_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class OcrProcessor:
    """Class to handle OCR processing tasks using RapidOCR.

    This processor uses RapidOCR with PP-OCRv5 models for high-quality OCR
    with support for multiple languages and automatic model detection.
    """

    def __init__(self, settings: OcrSettings) -> None:
        """Initialize the OCR processor.

        Args:
            settings: The OcrSettings object containing processing settings
        """
        self.settings = settings
        self.on_file_complete: Callable[[str, str, str, list], None] | None = None
        self.on_all_complete: Callable[[], None] | None = None
        self.on_progress: Callable[[int, int, str], None] | None = None

        # Processing state
        self._is_processing = False
        self._processing_started = False
        self._stop_requested = False
        self._processing_thread: threading.Thread | None = None

        # Stats tracking
        self._stats = ProcessingStats()
        self._total_files_at_start = 0

        # Progress tracking (simplified)
        self._file_progress = 0.0  # Progress within current file (0.0-1.0)
        self._current_status = ""  # Status message from backend
        self._current_filename = ""  # Current file being processed

        # Model discovery
        self._discovery = ModelDiscovery()

    def process_with_api(self) -> bool:
        """Process selected files using RapidOCR.

        Returns:
            True if processing started successfully, False otherwise
        """
        try:
            if not self._validate_input_files():
                return False

            self._setup_processing()

            # Set processing flags BEFORE starting thread to avoid race condition
            self._is_processing = True
            self._processing_started = True

            # Start processing in a background thread
            self._processing_thread = threading.Thread(
                target=self._process_all_files,
                daemon=True,
            )
            self._processing_thread.start()

            logger.info(
                _("Started OCR processing for {0} files using RapidOCR").format(
                    len(self.settings.selected_files)
                )
            )
            return True

        except Exception as e:
            logger.error(_("Error starting OCR processing: {0}").format(str(e)))
            return False

    def _validate_input_files(self) -> bool:
        """Validate that we have files to process."""
        if not self.settings.selected_files:
            logger.error(_("No files to process"))
            return False
        return True

    def _setup_processing(self) -> None:
        """Set up the OCR processing environment."""
        self._is_processing = False
        self._processing_started = False
        self._stop_requested = False
        self._current_engine = None  # Reference to current engine for cancellation
        self._stats = ProcessingStats()
        self._file_progress = 0.0
        self._current_status = ""
        self._current_filename = ""
        # Store original file count before processing removes files from queue
        self._total_files_at_start = len(self.settings.selected_files)
        self.settings.processed_files = []

        # Start checkpoint session for crash recovery
        checkpoint = get_checkpoint_manager()
        settings_snapshot = {
            "ocr_language": getattr(self.settings, "ocr_language", "latin"),
            "dpi": getattr(self.settings, "dpi", 300),
            "destination_folder": self.settings.destination_folder,
            "save_in_same_folder": self.settings.save_in_same_folder,
        }
        checkpoint.start_session(self.settings.selected_files, settings_snapshot)

    def _process_all_files(self) -> None:
        """Process all selected files in sequence."""
        try:
            # Copy list to avoid modification during iteration
            files_to_process = list(self.settings.selected_files)
            checkpoint = get_checkpoint_manager()

            for i, file_path in enumerate(files_to_process):
                if self._stop_requested:
                    logger.info(_("Processing stopped by user"))
                    break

                self._current_filename = os.path.basename(file_path)
                self._file_progress = 0.0

                try:
                    success, extracted_text, ocr_boxes = self._process_single_file(file_path, i)

                    if success and self.on_file_complete:
                        output_file = (
                            self.settings.processed_files[-1]
                            if self.settings.processed_files
                            else ""
                        )
                        # Update checkpoint with completed file
                        checkpoint.mark_file_completed(file_path, output_file)
                        self.on_file_complete(file_path, output_file, extracted_text, ocr_boxes)
                    elif not success:
                        checkpoint.mark_file_failed(file_path, "Processing failed")

                except InterruptedError:
                    logger.info(
                        _("Processing cancelled by user during file: {0}").format(file_path)
                    )
                    break

                except Exception as e:
                    logger.error(_("Error processing {0}: {1}").format(file_path, e))
                    checkpoint.mark_file_failed(file_path, str(e))
                    if self.on_file_complete:
                        self.on_file_complete(file_path, "", f"error: {e}", [])

            self._is_processing = False

            # Mark session complete if all files processed (not stopped by user)
            if not self._stop_requested:
                checkpoint.complete_session()

            if self.on_all_complete:
                self.on_all_complete()

        except Exception as e:
            logger.error(_("Processing thread error: {0}").format(e))
            self._is_processing = False

    def _process_single_file(self, file_path: str, index: int) -> tuple[bool, str, list]:
        """Process a single file with OCR.

        Returns:
            Tuple of (success, extracted_text, ocr_boxes)
        """
        if not file_path or not os.path.exists(file_path):
            logger.error(_("Error: File not found or invalid: {0}").format(file_path))
            return False, "", []

        output_file = self._get_output_file_path(file_path, index)
        if not output_file:
            return False, "", []

        # Reset processed_files list on first file of a new batch
        if index == 0:
            self.settings.processed_files = []

        # Create OCR config from settings, including file-specific page range
        config = self._create_ocr_config(file_path)

        # Create engine and process
        engine = RapidOCREngine(config)
        self._current_engine = engine  # Store reference for cancellation

        def progress_callback(current: int, total: int, message: str) -> None:
            # Backend reports progress as percentage (0-100)
            self._file_progress = current / 100.0 if total > 0 else 0.0
            self._current_status = message
            if self.on_progress:
                self.on_progress(current, total, message)

        stats = engine.process(file_path, output_file, progress_callback)

        # Update global stats
        self._stats.pages_processed += stats.pages_processed
        self._stats.pages_total += stats.pages_total
        self._stats.total_words += stats.total_words
        self._stats.total_chars += stats.total_chars
        self._stats.total_time += stats.total_time

        logger.info(
            _("Processed {0}: {1} pages, {2} words").format(
                os.path.basename(file_path),
                stats.pages_processed,
                stats.total_words,
            )
        )

        extracted_text = getattr(stats, "extracted_text", "")
        ocr_boxes = getattr(stats, "ocr_boxes", [])

        # Track output file only AFTER successful processing
        success = stats.pages_processed > 0
        split_files = getattr(stats, "split_output_files", [])

        if success:
            if split_files:
                # PDF was split into parts — track all parts
                for part_path in split_files:
                    if part_path not in self.settings.processed_files:
                        self.settings.processed_files.append(part_path)
                # Use first split part as representative output
                output_file = split_files[0]
            elif output_file not in self.settings.processed_files:
                self.settings.processed_files.append(output_file)

        # Record in processing history
        history = get_history_manager()
        history.add_entry(
            input_path=file_path,
            output_path=output_file,
            pages_processed=stats.pages_processed,
            processing_time_seconds=stats.total_time,
            language=getattr(self.settings, "ocr_language", "latin"),
            success=success,
        )

        return success, extracted_text, ocr_boxes

    def _create_ocr_config(self, file_path: str | None = None) -> OCRConfig:
        """Create OCR configuration from settings.

        Args:
            file_path: Optional file path to look up file-specific settings (e.g., page range)
        """
        # Map settings to OCRConfig
        language = getattr(self.settings, "ocr_language", "latin")
        dpi = getattr(self.settings, "dpi", 300)

        # Get file-specific page range if available
        page_range = None
        if file_path and hasattr(self.settings, "page_ranges"):
            page_range = self.settings.page_ranges.get(file_path)

        # Get file-specific modifications
        page_modifications = None
        if file_path and hasattr(self.settings, "file_modifications"):
            # Try exact match first
            state_dict = self.settings.file_modifications.get(file_path)
            if not state_dict:
                # Try basename match if full path fails (sometimes paths vary slightly)
                found_key = next(
                    (
                        k
                        for k in self.settings.file_modifications.keys()
                        if os.path.basename(k) == os.path.basename(file_path)
                    ),
                    None,
                )
                if found_key:
                    state_dict = self.settings.file_modifications[found_key]

            if state_dict and "pages" in state_dict:
                page_modifications = state_dict["pages"]

        return OCRConfig(
            language=language,
            dpi=dpi,
            # Preprocessing options from settings (reference defaults)
            # Color/Enhancement: OFF by default (PP-OCRv5 works best without)
            enable_preprocessing=getattr(self.settings, "enable_preprocessing", False),
            # Auto-detect controls whether detection gates corrections
            enable_auto_detect=getattr(self.settings, "enable_auto_detect", True),
            # Geometric corrections: ON by default (reference CLI behavior)
            enable_perspective_correction=getattr(
                self.settings, "enable_perspective_correction", False
            ),
            enable_deskew=getattr(self.settings, "enable_deskew", True),
            enable_orientation_detection=getattr(
                self.settings, "enable_orientation_detection", True
            ),
            # These only take effect if enable_preprocessing=True
            enable_auto_contrast=getattr(self.settings, "enable_auto_contrast", False),
            enable_auto_brightness=getattr(self.settings, "enable_auto_brightness", False),
            enable_denoise=getattr(self.settings, "enable_denoise", False),
            enable_scanner_effect=getattr(self.settings, "enable_scanner_effect", False),
            scanner_effect_strength=getattr(self.settings, "scanner_effect_strength", 1.0),
            # OCR options
            text_score_threshold=getattr(self.settings, "text_score_threshold", 0.3),
            box_thresh=getattr(self.settings, "box_thresh", 0.5),
            # Output options
            convert_to_pdfa=getattr(self.settings, "convert_to_pdfa", False),
            max_file_size_mb=getattr(self.settings, "max_file_size_mb", 0),
            # Image export options
            image_export_format=getattr(self.settings, "image_export_format", "original"),
            image_export_quality=getattr(self.settings, "image_export_quality", 85),
            auto_detect_quality=getattr(self.settings, "auto_detect_quality", True),
            # Parallel processing
            workers=getattr(self.settings, "parallel_workers", 0),
            # Page range (file-specific)
            page_range=page_range,
            page_modifications=page_modifications,
            # Force full OCR for editor-merged files (skip mixed content detection)
            force_full_ocr=bool(
                file_path
                and hasattr(self.settings, "original_file_paths")
                and file_path in self.settings.original_file_paths
            ),
            replace_existing_ocr=getattr(self.settings, "replace_existing_ocr", False),
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

            if os.path.exists(output_file) and not self.settings.overwrite_existing:
                output_file = self._generate_unique_filename(output_file)

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
        use_original = getattr(self.settings, "use_original_filename", False)

        if use_original:
            return os.path.join(output_dir, f"{base_name}.pdf")
        else:
            suffix = self.settings.get_pdf_suffix() or "ocr"
            if index == 0:
                return os.path.join(output_dir, f"{base_name}-{suffix}.pdf")
            else:
                return os.path.join(output_dir, f"{base_name}-{suffix}-{index + 1}.pdf")

    def get_available_ocr_languages(self) -> list[tuple[str, str]]:
        """Get a list of available OCR languages based on installed models.

        Returns:
            List of tuples (language_code, display_name)
        """
        try:
            # get_available_languages already returns list[tuple[str, str]]
            available = self._discovery.get_available_languages()
            return available

        except Exception as e:
            logger.error(_("Error getting OCR languages: {0}").format(e))
            return [("latin", "Latin")]

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

        # Calculate progress: completed files + current file progress
        completed_files = len(self.settings.processed_files) if self.settings.processed_files else 0
        base_progress = completed_files / total_files
        current_file_contribution = self._file_progress / total_files

        return min(1.0, base_progress + current_file_contribution)

    def get_processed_count(self) -> int:
        """Get the number of files that have been processed so far."""
        return len(self.settings.processed_files) if self.settings.processed_files else 0

    def get_total_count(self) -> int:
        """Get the total number of files to process.

        During processing, this returns the count from start to avoid
        incorrect counts as files are removed from the queue.
        """
        if hasattr(self, "_total_files_at_start") and self._total_files_at_start > 0:
            return self._total_files_at_start
        return len(self.settings.selected_files) if self.settings.selected_files else 0

    def get_total_pages(self) -> int:
        """Get the total number of pages processed."""
        return self._stats.pages_total

    def get_current_file_info(self) -> dict[str, Any]:
        """Get information about the currently processing file."""
        if not self._is_processing:
            return {}

        completed_files = len(self.settings.processed_files) if self.settings.processed_files else 0
        return {
            "filename": self._current_filename,
            "file_number": completed_files + 1,  # 1-based, current file being processed
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
        if input_file in self.settings.selected_files:
            self.settings.selected_files.remove(input_file)

            # Clear editor modifications for this file (transient state)
            if input_file in self.settings.file_modifications:
                del self.settings.file_modifications[input_file]
                logger.info(_("Cleared editor state for: {0}").format(os.path.basename(input_file)))

            # Clear original path mapping for edited files
            if input_file in self.settings.original_file_paths:
                del self.settings.original_file_paths[input_file]

            logger.info(
                _("Removed processed file from queue: {0}").format(os.path.basename(input_file))
            )

    def _generate_unique_filename(self, file_path: str) -> str:
        """Generate a unique filename by appending a counter."""
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        counter = 1
        new_path = file_path

        while os.path.exists(new_path):
            new_path = os.path.join(dir_name, f"{name}-{counter}{ext}")
            counter += 1

        logger.info(
            _("Generated unique filename to avoid overwriting: {0}").format(
                os.path.basename(new_path)
            )
        )
        return new_path

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
            if hasattr(self, "_current_engine") and self._current_engine is not None:
                self._current_engine.cancel_event.set()
                logger.info("Cancel event set on current OCR engine")

            self._is_processing = False
            self._processing_started = False

            # CRITICAL: Clear callbacks to prevent stale callbacks being called
            # after the user has already navigated away
            self.on_file_complete = None
            self.on_all_complete = None
            self.on_progress = None

            # Don't join() the thread — it blocks the UI. The daemon thread
            # will exit on its own after the current page finishes processing.
            self._processing_thread = None

            logger.info("OCR processor cleanup completed (non-blocking)")

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def is_processing(self) -> bool:
        """Check if processing is currently active."""
        return self._is_processing

    def has_resumable_session(self) -> bool:
        """Check if there's a previous incomplete session that can be resumed.

        Returns:
            True if an incomplete session exists with pending files
        """
        checkpoint = get_checkpoint_manager()
        return checkpoint.has_incomplete_session()

    def get_resumable_session_info(self) -> dict[str, Any] | None:
        """Get information about a resumable session.

        Returns:
            Dictionary with session info, or None if no session to resume
        """
        checkpoint = get_checkpoint_manager()
        return checkpoint.get_incomplete_session_info()

    def resume_previous_session(self) -> bool:
        """Resume processing from an incomplete session.

        This restores the list of pending files from the checkpoint and
        optionally restores relevant settings.

        Returns:
            True if session was resumed, False if nothing to resume
        """
        checkpoint = get_checkpoint_manager()
        result = checkpoint.resume_session()

        if not result:
            return False

        pending_files, settings_snapshot = result

        # Set pending files as selected files
        self.settings.selected_files = list(pending_files)

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
        checkpoint = get_checkpoint_manager()
        return checkpoint.discard_session()
