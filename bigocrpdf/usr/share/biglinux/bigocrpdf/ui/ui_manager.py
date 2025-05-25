"""
BigOcrPdf - UI Manager (Refactored)

This module manages the creation of UI components and coordinates between different page managers.
After refactoring, this module is now much smaller and acts as a coordinator.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from ui.settings_page import SettingsPageManager
from ui.terminal_page import TerminalPageManager
from ui.conclusion_page import ConclusionPageManager
from ui.dialogs_manager import DialogsManager
from utils.logger import logger
from utils.i18n import _


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
        self.quality_dropdown = None
        self.alignment_dropdown = None
        self.dest_entry = None
        self.same_folder_switch_row = None
        self.file_list_box = None

        # Terminal page components  
        self.terminal_progress_bar = None
        self.terminal_status_bar = None
        self.terminal_spinner = None

        # These will be populated when pages are created
        
    def create_settings_page(self) -> Gtk.Widget:
        """Create the settings page

        Returns:
            Widget containing the settings UI
        """
        page = self.settings_page_manager.create_settings_page()
        
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
        self.quality_dropdown = manager.quality_dropdown
        self.alignment_dropdown = manager.alignment_dropdown
        self.dest_entry = manager.dest_entry
        self.same_folder_switch_row = manager.same_folder_switch_row
        self.file_list_box = manager.file_list_box

    def _update_terminal_references(self) -> None:
        """Update references to terminal page components"""
        manager = self.terminal_page_manager
        
        # Update references so window can access them directly
        self.terminal_progress_bar = manager.terminal_progress_bar
        self.terminal_status_bar = manager.terminal_status_bar
        self.terminal_spinner = manager.terminal_spinner

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

    def update_processing_status(self, input_file: str = None) -> None:
        """Update processing status display
        
        Args:
            input_file: Currently processing file (optional)
        """
        self.terminal_page_manager.update_processing_status(input_file)

    def update_terminal_progress(self, fraction: float, text: str = None) -> None:
        """Update terminal progress bar
        
        Args:
            fraction: Progress fraction (0.0-1.0)
            text: Optional text to display
        """
        self.terminal_page_manager.update_terminal_progress(fraction, text)

    def update_terminal_status_complete(self) -> None:
        """Update terminal status to show completion"""
        self.terminal_page_manager.update_terminal_status_complete()

    def stop_terminal_spinner(self) -> None:
        """Stop the terminal spinner"""
        self.terminal_page_manager.stop_terminal_spinner()

    def show_completion_ui(self) -> None:
        """Update UI to show processing completion"""
        self.terminal_page_manager.show_completion_ui()

    def reset_terminal_progress(self) -> None:
        """Reset terminal progress to initial state"""
        self.terminal_page_manager.reset_progress()

    # Backward compatibility methods - these maintain the old interface

    def _show_extracted_text(self, file_path: str) -> None:
        """Backward compatibility wrapper for show_extracted_text
        
        Args:
            file_path: Path to the PDF file
        """
        self.show_extracted_text(file_path)

    def _update_queue_status(self) -> None:
        """Backward compatibility wrapper for refresh_queue_status"""
        self.refresh_queue_status()

    def _populate_file_list(self) -> None:
        """Refresh the file list in settings page"""
        if hasattr(self.settings_page_manager, '_populate_file_list'):
            self.settings_page_manager._populate_file_list()

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

    def get_settings_components(self) -> dict:
        """Get references to settings page components
        
        Returns:
            Dictionary of component references
        """
        return {
            'lang_dropdown': self.lang_dropdown,
            'quality_dropdown': self.quality_dropdown,
            'alignment_dropdown': self.alignment_dropdown,
            'dest_entry': self.dest_entry,
            'same_folder_switch_row': self.same_folder_switch_row,
            'file_list_box': self.file_list_box,
        }

    def get_terminal_components(self) -> dict:
        """Get references to terminal page components
        
        Returns:
            Dictionary of component references
        """
        return {
            'terminal_progress_bar': self.terminal_progress_bar,
            'terminal_status_bar': self.terminal_status_bar,
            'terminal_spinner': self.terminal_spinner,
        }

    def is_settings_page_ready(self) -> bool:
        """Check if settings page components are initialized
        
        Returns:
            True if settings page is ready, False otherwise
        """
        return (self.lang_dropdown is not None and 
                self.quality_dropdown is not None and
                self.alignment_dropdown is not None)

    def is_terminal_page_ready(self) -> bool:
        """Check if terminal page components are initialized
        
        Returns:
            True if terminal page is ready, False otherwise
        """
        return (self.terminal_progress_bar is not None and
                self.terminal_status_bar is not None)

    # Debug and logging methods

    def log_manager_status(self) -> None:
        """Log the status of all page managers for debugging"""
        logger.debug("UI Manager Status:")
        logger.debug(f"  Settings page ready: {self.is_settings_page_ready()}")
        logger.debug(f"  Terminal page ready: {self.is_terminal_page_ready()}")
        logger.debug(f"  File list box exists: {self.file_list_box is not None}")
        logger.debug(f"  Progress bar exists: {self.terminal_progress_bar is not None}")

    def get_manager_info(self) -> dict:
        """Get information about all managers
        
        Returns:
            Dictionary with manager information
        """
        return {
            'settings_manager': type(self.settings_page_manager).__name__,
            'terminal_manager': type(self.terminal_page_manager).__name__,
            'conclusion_manager': type(self.conclusion_page_manager).__name__,
            'dialogs_manager': type(self.dialogs_manager).__name__,
            'components_initialized': {
                'settings': self.is_settings_page_ready(),
                'terminal': self.is_terminal_page_ready(),
            }
        }