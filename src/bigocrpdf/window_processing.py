"""Window Navigation and Processing Callbacks Mixin."""

from __future__ import annotations

import gc
import os
import time
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk

from bigocrpdf.services.processor import OcrProcessor
from bigocrpdf.utils.comparison import compare_pdfs
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.timer import safe_remove_source

if TYPE_CHECKING:
    from bigocrpdf.services.settings import OcrSettings


class WindowProcessingMixin:
    """Mixin providing navigation callbacks and OCR processing management."""

    # Type stubs for attributes provided by the main window class
    settings: OcrSettings
    ocr_processor: OcrProcessor
    processed_files: list[str]
    process_start_time: float
    conclusion_timer_id: int | None

    def on_back_clicked(self, button: Gtk.Button) -> None:
        """Handle back button navigation - delegates to NavigationManager."""
        self.nav_manager.handle_back_clicked(button)

    def on_add_file_clicked(self, button: Gtk.Button) -> None:
        """Handle add file button click - delegates to FileSelectionManager."""
        self.file_manager.show_open_files_dialog()

    def show_toast(self, message: str, timeout: int = 3) -> None:
        """Show a toast notification.

        Args:
            message: The message to display
            timeout: The timeout in seconds
        """
        toast = Adw.Toast.new(message)
        toast.set_timeout(timeout)
        self.toast_overlay.add_toast(toast)

    def on_browse_clicked(self, button: Gtk.Button) -> None:
        """Handle the browse button click - delegates to FileSelectionManager."""
        self.file_manager.show_folder_selection_dialog()

    def _validate_ocr_settings(self) -> bool:
        """Validate OCR settings before processing.

        Checks: files selected, destination folder, disk space, write permissions.

        Returns:
            True if settings are valid
        """
        if not self.settings.selected_files:
            logger.warning(_("No files selected for processing"))
            self.show_toast(_("No files selected for processing"))
            return False

        save_in_same_folder = self.get_save_in_same_folder()

        if not save_in_same_folder and not self.settings.destination_folder:
            logger.warning(_("No destination folder selected"))
            self.show_toast(_("Please select a destination folder"))
            return False

        # Estimate output size (input size × 1.5 as safety margin)
        from bigocrpdf.utils.temp_manager import check_disk_space, check_writable

        total_input_bytes = 0
        for f in self.settings.selected_files:
            try:
                total_input_bytes += os.path.getsize(f)
            except OSError:
                pass
        needed_bytes = int(total_input_bytes * 1.5) if total_input_bytes else 0

        # Validate destination writable + has enough space
        if save_in_same_folder:
            # Check each input file's directory
            for f in self.settings.selected_files:
                dest_dir = os.path.dirname(f)
                ok, msg = check_writable(dest_dir)
                if not ok:
                    self.show_toast(msg)
                    return False
                ok, msg = check_disk_space(dest_dir, needed_bytes)
                if not ok:
                    self.show_toast(msg)
                    return False
                break  # All files likely on same filesystem
        else:
            dest_dir = self.settings.destination_folder
            ok, msg = check_writable(dest_dir)
            if not ok:
                self.show_toast(msg)
                return False
            ok, msg = check_disk_space(dest_dir, needed_bytes)
            if not ok:
                self.show_toast(msg)
                return False

        return True

    def get_save_in_same_folder(self) -> bool:
        """Get the value of the save in same folder switch.

        Returns:
            True if files should be saved in the same folder
        """
        if hasattr(self.ui, "folder_combo") and self.ui.folder_combo:
            # folder_combo: 0 = same folder, 1 = custom folder
            return self.ui.folder_combo.get_selected() == 0
        return True

    def _get_settings_from_ui(self) -> None:
        """Get settings from UI components."""
        if hasattr(self.ui, "lang_dropdown") and self.ui.lang_dropdown is not None:
            lang_index = self.ui.lang_dropdown.get_selected()
            languages = self.ocr_processor.get_available_ocr_languages()
            if 0 <= lang_index < len(languages):
                self.settings.lang = languages[lang_index][0]

        save_in_same_folder = self.get_save_in_same_folder()

        if hasattr(self.ui, "dest_entry") and self.ui.dest_entry is not None:
            dest_folder = self.ui.dest_entry.get_text()
            if dest_folder:
                self.settings.destination_folder = dest_folder

        self.settings.save_settings(
            self.settings.lang,
            self.settings.destination_folder,
            save_in_same_folder,
        )

    def on_apply_clicked(self, button: Gtk.Button | None = None) -> None:
        """Process the selected files with OCR.

        Args:
            button: The button that was clicked (optional, None when called from shortcut)
        """
        if not self._validate_ocr_settings():
            return

        # Immediate visual feedback (Doherty Threshold: respond within same frame)
        start_btn = self.custom_header_bar.start_button
        start_btn.set_sensitive(False)
        start_btn.set_label(_("Starting…"))

        # Clean up any previous processing state
        self._cleanup_ocr_processor()

        self._get_settings_from_ui()

        self.processed_files = []
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
            self.show_toast(_("Failed to start OCR processing"))
            self.custom_header_bar.start_button.set_label(_("Start OCR"))
            self.custom_header_bar.start_button.set_sensitive(True)
            return

        # Switch to terminal page (in main_stack) and update UI
        self.nav_manager.navigate_to_terminal()

        # Start progress updates - DELEGATE TO TERMINAL PAGE MANAGER
        self.ui.start_progress_monitor()

        self.announce_status(_("OCR processing started"))
        logger.info(_("OCR processing started using Python API"))

    def reset_and_go_to_settings(self) -> None:
        """Reset the application state and return to the settings page."""
        # Stop all timers first
        if hasattr(self.ui, "stop_progress_monitor"):
            self.ui.stop_progress_monitor()

        if self.conclusion_timer_id is not None:
            safe_remove_source(self.conclusion_timer_id)
            self.conclusion_timer_id = None

        # Clean up OCR processor
        self._cleanup_ocr_processor()
        self.processed_files = []

        # Clear file queue and all processing state
        self.settings.reset_processing_state(full=True)

        # Navigate back to settings page
        self.nav_manager.restore_next_button()
        self.nav_manager.navigate_to_settings()

        # Restore Start OCR button state
        start_btn = self.custom_header_bar.start_button
        start_btn.set_label(_("Start OCR"))
        start_btn.set_sensitive(True)

        self.update_file_info()
        logger.info("Application state reset - ready for new files")

    def _cleanup_ocr_processor(self) -> None:
        """Clean up OCR processor resources for a new processing run.

        Creates a fresh OcrProcessor instance to eliminate any race conditions
        with callbacks from previous sessions.
        """
        if self.ocr_processor:
            try:
                # Stop any running threads and clear callbacks
                self.ocr_processor.force_cleanup()
            except Exception as e:
                logger.error(f"Error during OCR processor cleanup: {e}")

        # Create a fresh processor instance - eliminates race conditions
        self.ocr_processor = OcrProcessor(self.settings)

        # Force garbage collection to release old resources
        gc.collect()

        logger.info("Created fresh OCR processor instance")

    def show_conclusion_page(self) -> None:
        """Show the conclusion page after OCR processing completes."""
        self.nav_manager.navigate_to_conclusion()

    def _safe_show_conclusion_page(self) -> bool:
        """Safely show conclusion page with allocation check.

        Called via GLib.timeout_add(100, ...). Returns True to keep
        polling until the widget is allocated (up to 5 s), then shows
        the conclusion page and returns False to stop the timer.
        """
        # Initialise the retry counter on first call
        if not hasattr(self, "_conclusion_retries"):
            self._conclusion_retries = 0

        try:
            # Terminal page is in main_stack, not stack
            if self.main_stack.get_visible_child_name() != "terminal":
                del self._conclusion_retries
                return False

            if not self.get_allocated_width() or not self.get_allocated_height():
                self._conclusion_retries += 1
                if self._conclusion_retries < 50:
                    return True  # GLib re-invokes after 100 ms
                logger.warning("Gave up waiting for allocation, showing conclusion anyway")

            self.show_conclusion_page()

        except Exception as e:
            logger.error(f"Error showing conclusion page safely: {e}")
            self.show_conclusion_page()

        # Cleanup and stop the timer
        if hasattr(self, "_conclusion_retries"):
            del self._conclusion_retries
        return False

    def update_file_info(self) -> None:
        """Update the file information UI after files have been added or removed."""
        current_page = self.stack.get_visible_child_name()

        file_count = len(self.settings.selected_files)
        self.announce_status(_("{} files in queue").format(file_count))

        if current_page != "settings":
            return

        if hasattr(self.ui, "refresh_queue_status"):
            self.ui.refresh_queue_status()
            logger.info(f"Queue status refreshed with {len(self.settings.selected_files)} files")
            return

        old_page = self.stack.get_visible_child()
        new_page = self.ui.create_settings_page()
        self.stack.remove(old_page)
        self.stack.add_named(new_page, "settings")
        self.stack.set_visible_child_name("settings")
        logger.info(f"UI updated with {len(self.settings.selected_files)} files in queue")

    def _on_file_processed(
        self,
        input_file: str,
        output_file: str,
        extracted_text: str = "",
        ocr_boxes: list | None = None,
    ) -> None:
        """Callback when a file is processed with OCR.

        Args:
            input_file: Path to the input file
            output_file: Path to the output file
            extracted_text: The extracted text content
            ocr_boxes: Structured OCR data with position information
        """
        if ocr_boxes is None:
            ocr_boxes = []

        def process_in_main_thread():
            # Remove processed file from queue
            self.ocr_processor.remove_processed_file(input_file)

            # Track output file
            if output_file and output_file not in self.settings.processed_files:
                self.settings.processed_files.append(output_file)

            # Store extracted text
            if not hasattr(self.settings, "extracted_text"):
                self.settings.extracted_text = {}
            self.settings.extracted_text[output_file] = extracted_text

            # Auto-save TXT file if enabled
            if self.settings.save_txt and extracted_text:
                from bigocrpdf.services.export_service import save_text_file

                separate = self.settings.txt_folder if self.settings.separate_txt_folder else None
                save_text_file(output_file, extracted_text, separate)

            # Auto-save ODF file if enabled
            if getattr(self.settings, "save_odf", False) and extracted_text:
                from bigocrpdf.services.export_service import save_odf_file

                save_odf_file(
                    output_file,
                    extracted_text,
                    ocr_boxes,
                    input_file,
                    include_images=getattr(self.settings, "odf_include_images", True),
                    use_formatting=getattr(self.settings, "odf_use_formatting", True),
                )

            # Store OCR boxes
            if not hasattr(self.settings, "ocr_boxes"):
                self.settings.ocr_boxes = {}
            self.settings.ocr_boxes[output_file] = ocr_boxes

            # Generate and store before/after comparison
            if not hasattr(self.settings, "comparison_results"):
                self.settings.comparison_results = []
            comparison = compare_pdfs(
                input_path=input_file,
                output_path=output_file,
                extracted_text=extracted_text,
                include_thumbnails=False,  # Thumbnails generated on-demand in UI
            )
            self.settings.comparison_results.append(comparison)
            logger.debug(
                f"Comparison: {comparison.input_size_mb}MB -> {comparison.output_size_mb}MB "
                f"({comparison.size_change_percent:+.1f}%)"
            )

            logger.info(
                _("Processed file {current}/{total}: {filename}").format(
                    current=len(self.settings.processed_files),
                    total=self.ocr_processor.get_total_count(),
                    filename=os.path.basename(input_file),
                )
            )

            # Update the status bar
            if hasattr(self.ui, "update_processing_status"):
                self.ui.update_processing_status(input_file)

            return False

        GLib.idle_add(process_in_main_thread)

    def _on_processing_complete(self) -> None:
        """Callback when all files are processed with OCR."""

        def complete_in_main_thread():
            logger.info(_("OCR processing complete callback triggered"))

            # Terminal page is in main_stack, not stack
            if self.main_stack.get_visible_child_name() != "terminal":
                logger.info(
                    _("Processing complete but no longer on terminal page, likely cancelled")
                )
                return False

            # Update the progress display to show 100%
            if hasattr(self.ui, "show_completion_ui"):
                self.ui.show_completion_ui()

            # Clean up temporary files
            if hasattr(self.settings, "processed_files") and self.settings.processed_files:
                self.settings.cleanup_temp_files(self.settings.processed_files)

            # Update the conclusion page
            if hasattr(self.ui, "update_conclusion_page"):
                self.ui.update_conclusion_page()

            # Show conclusion page with a delay (cancel existing timer first)
            if self.conclusion_timer_id is not None:
                safe_remove_source(self.conclusion_timer_id)
                self.conclusion_timer_id = None
            self.conclusion_timer_id = GLib.timeout_add(
                2000, lambda: self._safe_show_conclusion_page()
            )

            logger.info(
                _("OCR processing completed successfully for {count} files").format(
                    count=self.ocr_processor.get_processed_count()
                )
            )

            self.show_toast(_("OCR processing complete"))
            self.announce_status(_("OCR processing complete"))

            return False

        GLib.idle_add(complete_in_main_thread)

    def on_cancel_clicked(self) -> None:
        """Handle cancel button click during OCR processing."""
        logger.info(_("OCR processing cancelled by user"))

        # Stop the progress monitor first
        if hasattr(self.ui, "stop_progress_monitor"):
            self.ui.stop_progress_monitor()

        # Force cleanup the processor (this sets _stop_requested and waits for thread)
        if self.ocr_processor:
            self.ocr_processor.force_cleanup()

        self.show_toast(_("OCR processing cancelled"))
        self.reset_and_go_to_settings()
