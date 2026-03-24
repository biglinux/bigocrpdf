"""
BigOcrPdf - UI Manager (Refactored)

This module manages the creation of UI components and coordinates between different page managers.
After refactoring, this module is now much smaller and acts as a coordinator.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from typing import TYPE_CHECKING

from gi.repository import Gtk

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from bigocrpdf.ui.conclusion_page import ConclusionPageManager
from bigocrpdf.ui.dialogs_manager import DialogsManager
from bigocrpdf.ui.settings_page import SettingsPageManager
from bigocrpdf.ui.terminal_page import TerminalPageManager
from bigocrpdf.utils.logger import logger


class BigOcrPdfUI:
    """Coordinates UI creation and interaction between different page managers"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize UI manager with all page managers

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # Initialize all page managers
        self.settings_page_manager = SettingsPageManager(window)
        self.terminal_page_manager = TerminalPageManager(window)
        self.conclusion_page_manager = ConclusionPageManager(window)
        self.dialogs_manager = DialogsManager(window)

        # Keep references to commonly accessed components for backward compatibility
        self._setup_component_references()

    def _setup_component_references(self) -> None:
        """Set up references to UI components for backward compatibility"""
        # These references allow the window to access UI components directly
        # without knowing which page manager they belong to

        # Settings page components
        self.lang_dropdown = None
        self.dest_entry = None
        self.folder_combo = None
        self.file_list_box = None

        # Terminal page components
        self.terminal_progress_bar = None
        self.terminal_status_bar = None

        # These will be populated when pages are created

    def create_settings_page(self) -> Gtk.Widget:
        """Create the settings page

        Returns:
            Widget containing the settings UI
        """
        page = self.settings_page_manager.create_settings_page()

        # Connect view toggle buttons in header bar to queue panel
        if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
            self.settings_page_manager._connect_view_toggles()

        # Update component references for backward compatibility
        self._update_settings_references()

        return page

    def create_terminal_page(self) -> Gtk.Widget:
        """Create the terminal/processing page

        Returns:
            Widget containing the terminal UI
        """
        page = self.terminal_page_manager.create_terminal_page()

        # Update component references for backward compatibility
        self._update_terminal_references()

        return page

    def create_conclusion_page(self) -> Gtk.Widget:
        """Create the conclusion/results page

        Returns:
            Widget containing the conclusion UI
        """
        page = self.conclusion_page_manager.create_conclusion_page()
        return page

    def _update_settings_references(self) -> None:
        """Update references to settings page components"""
        manager = self.settings_page_manager

        # Update references so window can access them directly
        self.lang_dropdown = manager.lang_dropdown
        self.dest_entry = manager.dest_entry
        self.folder_combo = manager.folder_combo
        self.file_list_box = manager.file_list_box

    def _update_terminal_references(self) -> None:
        """Update references to terminal page components"""
        manager = self.terminal_page_manager

        # Update references so window can access them directly
        self.terminal_progress_bar = manager.terminal_progress_bar
        self.terminal_status_bar = manager.terminal_status_bar

    # Public interface methods - these delegate to appropriate managers

    def refresh_queue_status(self) -> None:
        """Update the queue status without rebuilding the entire settings page"""
        self.settings_page_manager.refresh_queue_status()

    def update_conclusion_page(self) -> None:
        """Update the conclusion page with results from OCR processing"""
        self.conclusion_page_manager.update_conclusion_page()

    def show_pdf_options_dialog(self, callback) -> None:
        """Show dialog with PDF output options

        Args:
            callback: Function to call when dialog is confirmed
        """
        self.dialogs_manager.show_pdf_options_dialog(callback)

    def show_extracted_text(self, file_path: str) -> None:
        """Display extracted text from a PDF file in a dialog

        Args:
            file_path: Path to the PDF file
        """
        self.dialogs_manager.show_extracted_text(file_path)

    # Terminal page coordination methods

    def start_progress_monitor(self) -> None:
        """Start monitoring OCR progress"""
        self.terminal_page_manager.start_progress_monitor()

    def stop_progress_monitor(self) -> None:
        """Stop monitoring OCR progress"""
        self.terminal_page_manager.stop_progress_monitor()

    def update_processing_status(self, input_file: str | None = None) -> None:
        """Update processing status display

        Args:
            input_file: Currently processing file (optional)
        """
        self.terminal_page_manager.update_processing_status(input_file)

    def update_terminal_progress(self, fraction: float, text: str | None = None) -> None:
        """Update terminal progress bar

        Args:
            fraction: Progress fraction (0.0-1.0)
            text: Optional text to display
        """
        self.terminal_page_manager.update_terminal_progress(fraction, text)

    def update_terminal_status_complete(self) -> None:
        """Update terminal status to show completion"""
        self.terminal_page_manager.update_terminal_status_complete()

    def show_completion_ui(self) -> None:
        """Update UI to show processing completion"""
        self.terminal_page_manager.show_completion_ui()

    # Cleanup and resource management

    def cleanup(self) -> None:
        """Clean up resources from all managers"""
        try:
            # Clean up terminal page resources (timers, etc.)
            self.terminal_page_manager.cleanup()

            # Reset conclusion page
            self.conclusion_page_manager.reset_page()

            logger.info("UI managers cleanup completed")

        except Exception as e:
            logger.error(f"Error during UI cleanup: {e}")

    # Utility methods for window integration

    # Debug and logging methods
