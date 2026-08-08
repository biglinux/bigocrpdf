"""OCR processing workflow controller for the main window."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk

from bigocrpdf.services.export_service import save_odf_file, save_text_file
from bigocrpdf.services.processor import OcrProcessor
from bigocrpdf.utils.comparison import compare_pdfs
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.temp_manager import check_disk_space, check_writable
from bigocrpdf.utils.timer import safe_remove_source

if TYPE_CHECKING:
    from bigocrpdf import OcrDependencyState
    from bigocrpdf.services.settings import OcrSettings
    from bigocrpdf.ui.navigation_manager import NavigationManager
    from bigocrpdf.ui.window_ui import BigOcrPdfUI


class _BatchCompletionState(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class _BatchCompletionAction(Enum):
    SHOW_RESULTS = "show-results"
    REVIEW_PARTIAL = "review-partial"
    RETRY_FAILED = "retry-failed"


@dataclass(frozen=True)
class _BatchCompletion:
    state: _BatchCompletionState
    title: str
    message: str
    action: _BatchCompletionAction
    successful_files: int
    failed_files: int


def _build_batch_completion(total_files: int, successful_files: int) -> _BatchCompletion:
    """Build a truthful user-visible outcome from confirmed input-file counts."""
    if total_files <= 0:
        raise ValueError("total_files must be positive")
    if successful_files < 0 or successful_files > total_files:
        raise ValueError("successful_files must be between zero and total_files")

    failed_files = total_files - successful_files
    if successful_files == total_files:
        return _BatchCompletion(
            state=_BatchCompletionState.SUCCESS,
            title=_("OCR processing complete"),
            message=ngettext(
                "OCR processing completed successfully for {count} file",
                "OCR processing completed successfully for {count} files",
                successful_files,
            ).format(count=successful_files),
            action=_BatchCompletionAction.SHOW_RESULTS,
            successful_files=successful_files,
            failed_files=0,
        )

    message = _("Saved {ok}; {n} failed").format(ok=successful_files, n=failed_files)
    if successful_files:
        return _BatchCompletion(
            state=_BatchCompletionState.PARTIAL,
            title=_("Results"),
            message=message,
            action=_BatchCompletionAction.REVIEW_PARTIAL,
            successful_files=successful_files,
            failed_files=failed_files,
        )

    return _BatchCompletion(
        state=_BatchCompletionState.FAILURE,
        title=_("OCR processing failed."),
        message=message,
        action=_BatchCompletionAction.RETRY_FAILED,
        successful_files=0,
        failed_files=failed_files,
    )


def _estimate_output_bytes(selected_files: list[str]) -> int:
    total_input_bytes = 0
    for selected_file in selected_files:
        try:
            total_input_bytes += Path(selected_file).stat().st_size
        except OSError:
            pass

    return int(total_input_bytes * 1.5) if total_input_bytes else 0


class ProcessingController:
    """Own the main-window OCR workflow and its transient batch state."""

    def __init__(
        self,
        *,
        parent: Adw.ApplicationWindow,
        settings: OcrSettings,
        ui: BigOcrPdfUI,
        nav_manager: NavigationManager,
        ocr_dependency: OcrDependencyState,
        show_ocr_unavailable: Callable[[], bool],
        announce_status: Callable[[str], None],
    ) -> None:
        self.parent = parent
        self.settings = settings
        self.ocr_processor = OcrProcessor(settings)
        self.ui = ui
        self.nav_manager = nav_manager
        self.ocr_dependency = ocr_dependency
        self._show_ocr_unavailable = show_ocr_unavailable
        self._announce_status = announce_status
        self.process_start_time = 0.0
        self.conclusion_timer_id: int | None = None
        self._closed = False

    def _validate_ocr_settings(self) -> bool:
        """Validate OCR settings before processing.

        Checks: files selected, destination folder, disk space, write permissions.

        Returns:
            True if settings are valid
        """
        if not self.settings.selected_files:
            logger.warning(_("No files selected for processing"))
            self.ui.show_toast(_("No files selected for processing"))
            return False

        save_in_same_folder = self.get_save_in_same_folder()

        if not save_in_same_folder and not self.settings.destination_folder:
            logger.warning(_("No destination folder selected"))
            self.ui.show_toast(_("Please select a destination folder"))
            return False

        needed_bytes = _estimate_output_bytes(self.settings.selected_files)

        if save_in_same_folder:
            dest_dir = os.path.dirname(self.settings.selected_files[0])
            return self._validate_output_destination(dest_dir, needed_bytes)

        return self._validate_output_destination(
            self.settings.destination_folder,
            needed_bytes,
        )

    def _validate_output_destination(self, dest_dir: str, needed_bytes: int) -> bool:
        ok, msg = check_writable(dest_dir)
        if not ok:
            self.ui.show_toast(msg)
            return False

        ok, msg = check_disk_space(dest_dir, needed_bytes)
        if not ok:
            self.ui.show_toast(msg)
            return False

        return True

    def get_save_in_same_folder(self) -> bool:
        """Get the value of the save in same folder switch.

        Returns:
            True if files should be saved in the same folder
        """
        folder_combo = self.ui.settings_page_manager.folder_combo
        if folder_combo:
            # folder_combo: 0 = same folder, 1 = custom folder
            return folder_combo.get_selected() == 0
        return True

    def _get_settings_from_ui(self) -> None:
        """Get settings from UI components."""
        settings_page = self.ui.settings_page_manager
        if settings_page.lang_dropdown is not None:
            lang_index = settings_page.lang_dropdown.get_selected()
            languages = self.ocr_processor.get_available_ocr_languages()
            if 0 <= lang_index < len(languages):
                self.settings.lang = languages[lang_index][0]

        save_in_same_folder = self.get_save_in_same_folder()

        if settings_page.dest_entry is not None:
            self.settings.destination_folder = settings_page.dest_entry.get_text()

        self.settings.save_settings(
            self.settings.lang,
            self.settings.destination_folder,
            save_in_same_folder,
        )

    def start(self, _button: Gtk.Button | None = None) -> bool:
        """Process the selected files with OCR.

        Args:
            _button: The button that triggered processing, when applicable.

        Returns:
            True when OCR processing started successfully.
        """
        if not self.ocr_dependency.is_available:
            self._show_ocr_unavailable()
            return False

        if self.ocr_processor.has_active_worker():
            self.ui.show_toast(_("Failed to start OCR processing"))
            return False

        self._get_settings_from_ui()

        if not self._validate_ocr_settings():
            return False

        # Immediate visual feedback (Doherty Threshold: respond within same frame)
        start_btn = self.ui.custom_header_bar.start_button
        start_btn.set_sensitive(False)
        start_btn.set_label(_("Starting…"))

        # Clean up any previous processing state
        self._cleanup_ocr_processor()
        self.process_start_time = time.time()

        # Register callbacks for OCR processing events
        self.ocr_processor.register_callbacks(
            on_file_complete=self._on_file_processed,
            on_all_complete=self._on_processing_complete,
        )

        # Start OCR processing using Python API
        success = self.ocr_processor.process_with_api()
        if not success:
            logger.error(_("Failed to start OCR processing"))
            self.ui.show_toast(_("Failed to start OCR processing"))
            start_btn.set_label(_("Start OCR"))
            start_btn.set_sensitive(self.ocr_dependency.is_available)
            return False

        # Switch to terminal page (in main_stack) and update UI
        self.nav_manager.navigate_to_terminal()

        self.ui.terminal_page_manager.start_progress_monitor()

        self._announce_status(_("OCR processing started"))
        logger.info(_("OCR processing started using Python API"))
        return True

    def reset_to_settings(self) -> None:
        """Reset the application state and return to the settings page."""
        # Stop all timers first
        self.ui.terminal_page_manager.stop_progress_monitor()

        self._clear_conclusion_timer()

        # Clean up OCR processor
        self._cleanup_ocr_processor()

        # Reset processing state but keep remaining files in queue
        self.settings.reset_processing_state(full=False)
        self.ocr_processor.run_when_idle(self._schedule_processing_state_reset)

        # Navigate back to settings page
        self.nav_manager.navigate_to_settings()

        # Restore Start OCR button state
        start_btn = self.ui.custom_header_bar.start_button
        start_btn.set_label(_("Start OCR"))
        start_btn.set_sensitive(self.ocr_dependency.is_available)

        self.ui.update_file_info()
        logger.info("Application state reset - ready for new files")

    def _schedule_processing_state_reset(self) -> None:
        """Move the terminal worker reset back onto the GTK main loop."""
        if not self._closed:
            GLib.idle_add(self._finalize_processing_state_reset)

    def _finalize_processing_state_reset(self) -> bool:
        """Clear mutations completed by a worker after non-blocking cancellation."""
        if self._closed:
            return GLib.SOURCE_REMOVE

        self.settings.reset_processing_state(full=False)
        self.ui.update_file_info()
        return GLib.SOURCE_REMOVE

    def _cleanup_ocr_processor(self) -> None:
        """Stop callbacks and request non-blocking cleanup of the current worker."""
        try:
            self.ocr_processor.force_cleanup()
        except Exception as error:
            logger.error("Error during OCR processor cleanup: %s", error)
        else:
            logger.info("Cleaned OCR processor state")

    def _on_file_processed(
        self,
        input_file: str,
        output_file: str,
        extracted_text: str = "",
        ocr_boxes: list | None = None,
    ) -> None:
        """Prepare a completed OCR result and publish it on the GTK main loop."""
        if self._closed:
            return

        if not output_file:
            logger.error("OCR processing failed for input file: %s", input_file)
            return

        boxes = ocr_boxes or []
        if self.settings.save_txt and extracted_text:
            separate = self.settings.txt_folder if self.settings.separate_txt_folder else None
            save_text_file(output_file, extracted_text, separate, boxes)

        if self.settings.save_odf and extracted_text:
            save_odf_file(
                output_file,
                extracted_text,
                boxes,
                input_file,
                include_images=self.settings.odf_include_images,
            )

        comparison = compare_pdfs(
            input_path=input_file,
            output_path=output_file,
            extracted_text=extracted_text,
            include_thumbnails=False,
        )

        def publish_result() -> bool:
            if self._closed:
                return GLib.SOURCE_REMOVE

            self.settings.extracted_text[output_file] = extracted_text
            self.settings.ocr_boxes[output_file] = boxes
            self.settings.comparison_results.append(comparison)
            logger.debug(
                "Comparison: %sMB -> %sMB (%+.1f%%)",
                comparison.input_size_mb,
                comparison.output_size_mb,
                comparison.size_change_percent,
            )
            logger.info(
                _("Processed file {current}/{total}: {filename}").format(
                    current=self.ocr_processor.get_completed_input_count(),
                    total=self.ocr_processor.get_total_count(),
                    filename=self.settings.display_name(input_file),
                )
            )
            self.ui.terminal_page_manager.update_processing_status(input_file)

            # Generated inputs can be released after every result consumer finishes.
            self.ocr_processor.remove_processed_file(input_file)
            return GLib.SOURCE_REMOVE

        GLib.idle_add(publish_result)

    def _on_processing_complete(self) -> None:
        """Callback when all files are processed with OCR."""

        def complete_in_main_thread():
            if self._closed:
                return GLib.SOURCE_REMOVE

            logger.info(_("OCR processing complete callback triggered"))

            # Terminal page is in main_stack, not stack
            if self.ui.main_stack.get_visible_child_name() != "terminal":
                logger.info(
                    _("Processing complete but no longer on terminal page, likely cancelled")
                )
                return GLib.SOURCE_REMOVE

            outcome = _build_batch_completion(
                total_files=self.ocr_processor.get_total_count(),
                successful_files=self.ocr_processor.get_successful_input_count(),
            )

            # Publish the outcome derived from canonical input-level counters.
            terminal_page = self.ui.terminal_page_manager
            terminal_page.update_terminal_progress(1.0, "100%")
            terminal_page.stop_progress_monitor()

            if outcome.successful_files:
                self.ui.conclusion_page_manager.update_conclusion_page()

            logger.info("%s: %s", outcome.title, outcome.message)
            self._announce_status(outcome.message)

            if outcome.action is _BatchCompletionAction.SHOW_RESULTS:
                self.ui.show_toast(outcome.message)
                self._schedule_conclusion_page()
            else:
                self._present_batch_completion(outcome)

            return GLib.SOURCE_REMOVE

        GLib.idle_add(complete_in_main_thread)

    def _schedule_conclusion_page(self) -> None:
        """Schedule the results page after the terminal completion feedback."""
        self._clear_conclusion_timer()
        self.conclusion_timer_id = GLib.timeout_add(
            2000,
            self._show_scheduled_conclusion_page,
        )

    def _show_scheduled_conclusion_page(self) -> bool:
        """Show results if processing completion is still the active page."""
        self.conclusion_timer_id = None
        if not self._closed and self.ui.main_stack.get_visible_child_name() == "terminal":
            self.nav_manager.navigate_to_conclusion()
        return GLib.SOURCE_REMOVE

    def _clear_conclusion_timer(self) -> None:
        if self.conclusion_timer_id is not None:
            safe_remove_source(self.conclusion_timer_id)
            self.conclusion_timer_id = None

    def _present_batch_completion(self, outcome: _BatchCompletion) -> None:
        """Present partial or failed outcomes with an honest next action."""
        dialog = Adw.AlertDialog(heading=outcome.title, body=outcome.message)
        dialog.add_response("back", _("Back"))
        dialog.set_close_response("back")

        if outcome.action is _BatchCompletionAction.REVIEW_PARTIAL:
            dialog.add_response("results", _("Results"))
            dialog.set_response_appearance("results", Adw.ResponseAppearance.SUGGESTED)
            dialog.set_default_response("results")
        else:
            dialog.set_default_response("back")

        dialog.connect("response", self._on_batch_completion_response)
        dialog.present(cast(Gtk.Widget, self.parent))

    def _on_batch_completion_response(self, _dialog: Adw.AlertDialog, response: str) -> None:
        """Follow the action selected for a partial or failed batch."""
        if response == "results":
            self._schedule_conclusion_page()
        else:
            self.reset_to_settings()

    def cancel(self) -> None:
        """Handle cancel button click during OCR processing."""
        logger.info(_("OCR processing cancelled by user"))
        self.ui.show_toast(_("OCR processing cancelled"))
        self.reset_to_settings()

    def cleanup(self) -> None:
        """Stop controller timers and release processor resources."""
        self._closed = True
        self._clear_conclusion_timer()
        self._cleanup_ocr_processor()
