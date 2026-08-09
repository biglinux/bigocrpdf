"""
BigOcrPdf - Window Module

This module contains the main application window implementation.
"""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf import OcrDependencyState
from bigocrpdf.config import (
    APP_ICON_NAME,
    CONFIG_DIR,
    WINDOW_STATE_KEY,
)
from bigocrpdf.processing_controller import ProcessingController
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.ui.file_selection_manager import FileSelectionManager
from bigocrpdf.ui.navigation_manager import NavigationManager
from bigocrpdf.ui.widgets import present_ocr_unavailable_dialog
from bigocrpdf.ui.window_ui import BigOcrPdfUI
from bigocrpdf.utils.config_manager import DEFAULT_CONFIG, get_config_manager
from bigocrpdf.utils.logger import logger
from bigocrpdf.window_controller import WindowController


class BigOcrPdfWindow(Adw.ApplicationWindow):
    """Main application window for BigOcrPdf with stable progress tracking."""

    # Configuration file to store welcome dialog preference
    WELCOME_DIALOG_CONFIG = os.path.join(CONFIG_DIR, "show_welcome_dialog")

    def __init__(
        self,
        app: Adw.Application,
        *,
        ocr_dependency: OcrDependencyState,
    ) -> None:
        """Initialize application window.

        Args:
            app: The parent Adw.Application instance
            ocr_dependency: Resolved OCR dependency state from startup
        """
        # Load saved window size
        width, height = self._load_window_size()

        super().__init__(
            application=app,
            title="Big OCR PDF",
            default_width=width,
            default_height=height,
        )

        # The sidebar overlays the content below the 700 sp breakpoint, so the
        # minimum must describe the collapsed content instead of both panes.
        self.set_size_request(640, 400)

        # Set up the window icon
        self.set_icon_name(APP_ICON_NAME)

        self.ocr_dependency = ocr_dependency
        self._ocr_unavailable_dialog: Adw.AlertDialog | None = None
        self._close_prepared = False

        # Initialize components
        self.settings = OcrSettings()
        self.ui = BigOcrPdfUI(self)
        self.file_manager = FileSelectionManager(self, self.settings, self.ui)
        self.nav_manager = NavigationManager(self.ui, self.announce_status)
        self.processing = ProcessingController(
            parent=self,
            settings=self.settings,
            ui=self.ui,
            nav_manager=self.nav_manager,
            ocr_dependency=self.ocr_dependency,
            show_ocr_unavailable=self.show_ocr_unavailable_dialog,
            announce_status=self.announce_status,
        )
        self.actions = WindowController(
            parent=self,
            settings=self.settings,
            ui=self.ui,
            file_manager=self.file_manager,
            processing=self.processing,
            ocr_dependency=self.ocr_dependency,
            show_ocr_unavailable=self.show_ocr_unavailable_dialog,
            welcome_config_path=self.WELCOME_DIALOG_CONFIG,
        )

        # Create the main layout
        self.ui.setup()

        # Connect close-request signal to save window state
        self.connect("close-request", self._on_close_request)

    def show_ocr_unavailable_dialog(self) -> bool:
        """Explain why OCR is blocked and the action required to recover."""
        if self.ocr_dependency.is_available or self._ocr_unavailable_dialog is not None:
            return False

        self._ocr_unavailable_dialog = present_ocr_unavailable_dialog(
            self,
            self.ocr_dependency.error,
            self._on_ocr_unavailable_response,
        )
        return False

    def _on_ocr_unavailable_response(
        self,
        _dialog: Adw.AlertDialog,
        _response: str,
    ) -> None:
        self._ocr_unavailable_dialog = None

    def do_realize(self) -> None:
        """Realize the GTK window and select the queue settings page."""
        Adw.ApplicationWindow.do_realize(self)
        self.ui.stack.set_visible_child_name("settings")

    def _return_to_main_view(self) -> None:
        """Clear completed work and return to the queue."""
        self.ui.conclusion_page_manager.reset_page()
        self.settings.reset_processing_state()
        self.actions.clear_file_queue()
        self.ui.main_stack.set_visible_child_name("main_view")

    def announce_status(self, message: str) -> None:
        """Ask assistive technology to announce a status message."""
        self.announce(message, Gtk.AccessibleAnnouncementPriority.MEDIUM)

    def _load_window_size(self) -> tuple[int, int]:
        """Load window size from configuration.

        Returns:
            Tuple of (width, height) for the window
        """
        config = get_config_manager()
        width = config.get(f"{WINDOW_STATE_KEY}.width", DEFAULT_CONFIG["window"]["width"])
        height = config.get(f"{WINDOW_STATE_KEY}.height", DEFAULT_CONFIG["window"]["height"])

        # Ensure minimum reasonable size
        width = max(width, 400)
        height = max(height, 300)

        logger.info(f"Loading window size: {width}x{height}")
        return width, height

    def _save_window_size(self) -> None:
        """Save current window size to configuration."""
        if self.is_maximized() or self.is_fullscreen():
            return

        config = get_config_manager()
        width = self.get_width()
        height = self.get_height()

        # Only save if window has valid dimensions
        if width > 0 and height > 0:
            config.set(f"{WINDOW_STATE_KEY}.width", width, save_immediately=False)
            config.set(f"{WINDOW_STATE_KEY}.height", height, save_immediately=True)
            logger.info(f"Window size saved: {width}x{height}")

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        """Save state and release resources before closing."""
        self.prepare_close()
        return False

    def prepare_close(self) -> None:
        """Idempotently save state and release window-owned resources."""
        if self._close_prepared:
            return
        self._close_prepared = True

        try:
            self._save_window_size()
        except Exception:
            logger.exception("Failed to save window size during cleanup")

        try:
            self.processing.cleanup()
        except Exception:
            logger.exception("Failed to clean up OCR processing")

        try:
            self.ui.cleanup()
        except Exception:
            logger.exception("Failed to clean up window UI")

        try:
            self.settings.reset_processing_state()
        except Exception:
            logger.exception("Failed to reset processing state during cleanup")
