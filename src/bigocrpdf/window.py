"""
BigOcrPdf - Window Module

This module contains the main application window implementation.
"""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf.config import (
    APP_ICON_NAME,
    CONFIG_DIR,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    WINDOW_STATE_KEY,
)
from bigocrpdf.services.processor import OcrProcessor
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.ui.file_selection_manager import FileSelectionManager
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.ui.navigation_manager import NavigationManager
from bigocrpdf.ui.ui_manager import BigOcrPdfUI
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.timer import TimerManager, safe_remove_source
from bigocrpdf.window_actions import WindowActionsSignalsMixin
from bigocrpdf.window_processing import WindowProcessingMixin
from bigocrpdf.window_ui_setup import WindowUISetupMixin


class BigOcrPdfWindow(
    WindowUISetupMixin,
    WindowActionsSignalsMixin,
    WindowProcessingMixin,
    Adw.ApplicationWindow,
):
    """Main application window for BigOcrPdf with stable progress tracking."""

    # Configuration file to store welcome dialog preference
    WELCOME_DIALOG_CONFIG = os.path.join(CONFIG_DIR, "show_welcome_dialog")

    def __init__(self, app: Adw.Application) -> None:
        """Initialize application window.

        Args:
            app: The parent Adw.Application instance
        """
        # Load saved window size
        width, height = self._load_window_size()

        super().__init__(
            application=app,
            title="Big OCR PDF",
            default_width=width,
            default_height=height,
        )

        # Set minimum window size to prevent controls from being cut off
        # Left sidebar (300px) + right content (620px) = 920px minimum width
        self.set_size_request(920, 400)

        # Set up the window icon
        self.set_icon_name(APP_ICON_NAME)

        # Initialize components
        self.settings = OcrSettings()
        self.ocr_processor = OcrProcessor(self.settings)
        self.ui = BigOcrPdfUI(self)

        # Initialize timer manager for centralized timer handling
        self._timer_manager = TimerManager()

        # Initialize state variables
        self.processed_files: list[str] = []
        self.process_start_time: float = 0
        self.conclusion_timer_id: int | None = None

        # Initialize UI components
        self.stack: Adw.ViewStack | None = None
        self.toast_overlay: Adw.ToastOverlay | None = None
        self.custom_header_bar: HeaderBar | None = None
        self.toolbar_view: Adw.ToolbarView | None = None

        # Signal handler tracking
        self.signal_handlers: dict = {}

        # Create the main layout
        self.setup_ui()

        # Initialize managers (after UI setup)
        self.file_manager = FileSelectionManager(self)
        self.nav_manager = NavigationManager(self)

        # Set up navigation callbacks
        self.nav_manager.set_on_apply_callback(self.on_apply_clicked)
        self.nav_manager.set_on_reset_callback(self.reset_and_go_to_settings)

        # Connect close-request signal to save window state
        self.connect("close-request", self._on_close_request)

    def _load_window_size(self) -> tuple[int, int]:
        """Load window size from configuration.

        Returns:
            Tuple of (width, height) for the window
        """
        config = get_config_manager()
        width = config.get(f"{WINDOW_STATE_KEY}.width", DEFAULT_WINDOW_WIDTH)
        height = config.get(f"{WINDOW_STATE_KEY}.height", DEFAULT_WINDOW_HEIGHT)

        # Ensure minimum reasonable size
        width = max(width, 400)
        height = max(height, 300)

        logger.info(f"Loading window size: {width}x{height}")
        return width, height

    def _save_window_size(self) -> None:
        """Save current window size to configuration."""
        config = get_config_manager()
        width = self.get_width()
        height = self.get_height()

        # Only save if window has valid dimensions
        if width > 0 and height > 0:
            config.set(f"{WINDOW_STATE_KEY}.width", width, save_immediately=False)
            config.set(f"{WINDOW_STATE_KEY}.height", height, save_immediately=True)
            logger.info(f"Window size saved: {width}x{height}")

    def _on_close_request(self, window: Gtk.Window) -> bool:
        """Handle window close request - save state before closing.

        Args:
            window: The window being closed

        Returns:
            False to allow window to close
        """
        self._save_window_size()
        return False  # Allow window to close

    def do_destroy(self) -> None:
        """Clean up all resources when the window is destroyed.

        This method ensures proper cleanup of:
        - GLib timers
        - OCR processor and queue
        - Signal handlers
        - UI manager resources
        """
        logger.info("Window destroy initiated - cleaning up resources")

        try:
            # 1. Stop all timers first
            self._cleanup_all_timers()

            # 2. Stop OCR processing if running
            self._cleanup_ocr_processor()

            # 3. Cleanup UI manager resources
            if hasattr(self, "ui") and self.ui:
                self.ui.cleanup()

            # 4. Disconnect all tracked signal handlers
            self._disconnect_all_signals()

            # 5. Clear settings state
            if hasattr(self, "settings") and self.settings:
                self.settings.reset_processing_state()

            logger.info("Window cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during window cleanup: {e}")

    def _cleanup_all_timers(self) -> None:
        """Clean up all GLib timers."""
        # Remove timer manager timers
        if hasattr(self, "_timer_manager") and self._timer_manager:
            count = self._timer_manager.remove_all()
            if count > 0:
                logger.info(f"Removed {count} timers from timer manager")

        # Remove conclusion timer
        if self.conclusion_timer_id is not None:
            safe_remove_source(self.conclusion_timer_id)
            self.conclusion_timer_id = None

    def _disconnect_all_signals(self) -> None:
        """Disconnect all tracked signal handlers."""
        for widget, signals in dict(self.signal_handlers).items():
            for signal_name, handler_id in dict(signals).items():
                try:
                    if widget and hasattr(widget, "handler_is_connected"):
                        if widget.handler_is_connected(handler_id):
                            widget.disconnect(handler_id)
                except Exception as e:
                    logger.debug(f"Error disconnecting signal {signal_name}: {e}")

        self.signal_handlers.clear()
