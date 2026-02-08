"""
BigOcrPdf - Application Module

This module contains the main application class for the BigOcrPdf application.
"""

from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.config import (
    APP_DESCRIPTION,
    APP_DEVELOPERS,
    APP_ICON_NAME,
    APP_ID,
    APP_ISSUES,
    APP_NAME,
    APP_VERSION,
    APP_WEBSITE,
    SHORTCUTS,
)
from bigocrpdf.ui.image_ocr_window import ImageOcrWindow
from bigocrpdf.ui.widgets import load_css
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.window import BigOcrPdfWindow


class BigOcrPdfApp(Adw.Application):
    """Application class for BigOcrPdf."""

    def __init__(self) -> None:
        """Initialize the application."""
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_OPEN)

        # Add command line handling
        self.add_main_option(
            "version",
            ord("v"),
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            _("Print version information and exit"),
            None,
        )
        self.add_main_option(
            "image-mode",
            0,
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            _("Start in image conversion mode"),
            None,
        )

        # Setup signals
        self.connect("activate", self.on_activate)
        self.connect("open", self.on_open)
        self.connect("handle-local-options", self.on_handle_local_options)

        # Set up application actions
        self._setup_actions()

    def _setup_actions(self) -> None:
        """Set up application actions."""
        # About action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self.on_about_action)
        self.add_action(about_action)

        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)

        # Image OCR action
        image_ocr_action = Gio.SimpleAction.new("image-ocr", None)
        image_ocr_action.connect("activate", self.on_image_ocr_action)
        self.add_action(image_ocr_action)

        # Set up keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up application-level keyboard shortcuts."""
        try:
            # Application-level shortcuts
            self.set_accels_for_action("app.quit", [SHORTCUTS.get("quit", "<Control>q")])
            self.set_accels_for_action("app.about", [SHORTCUTS.get("about", "F1")])

            # Window-level shortcuts (win. prefix)
            self.set_accels_for_action("win.add-files", [SHORTCUTS.get("add-files", "<Control>o")])
            self.set_accels_for_action(
                "win.start-processing", [SHORTCUTS.get("start-processing", "<Control>Return")]
            )
            self.set_accels_for_action(
                "win.cancel-processing", [SHORTCUTS.get("cancel-processing", "Escape")]
            )
            self.set_accels_for_action(
                "win.remove-all-files", [SHORTCUTS.get("remove-all-files", "<Control>r")]
            )

            logger.info("Keyboard shortcuts configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup keyboard shortcuts: {e}")

    def on_handle_local_options(self, app: Adw.Application, options: GLib.VariantDict) -> int:
        """Handle command line options.

        Args:
            app: The application
            options: Command line options

        Returns:
            Integer value indicating if processing should continue
        """
        if options.contains("version"):
            print(f"{APP_NAME} {APP_VERSION}")
            return 0  # Exit successfully

        if options.contains("image-mode"):
            self.image_mode = True

        return -1  # Continue processing

    def on_activate(self, app: Adw.Application) -> None:
        """Callback for application activation.

        Args:
            app: The application instance
        """
        try:
            # Load custom CSS
            load_css()

            # Check if we already have a window open
            win = self.get_active_window()
            if not win:
                # Check for image mode
                if getattr(self, "image_mode", False):
                    # Launch ImageOcrWindow
                    win = ImageOcrWindow(app)
                    logger.info("Started in image mode")
                else:
                    # Create the main window
                    win = BigOcrPdfWindow(app)

            # Show the window
            win.present()

            # Check if we should show the welcome dialog
            if hasattr(win, "should_show_welcome_dialog") and win.should_show_welcome_dialog():
                # Use a small delay to ensure the window is fully drawn
                GLib.timeout_add(300, lambda: win.show_welcome_dialog())

            # Check for resumable session (after welcome dialog)
            if hasattr(win, "check_resumable_session"):
                GLib.timeout_add(500, lambda: win.check_resumable_session())

            logger.info(_("Application started successfully"))

        except Exception as e:
            logger.error(f"{_('Error activating application')}: {e}")
            error_dialog = Gtk.AlertDialog()
            error_dialog.set_message(_("Error starting application"))
            error_dialog.set_detail(str(e))
            error_dialog.show()

    def on_open(self, app: Adw.Application, files: list, n_files: int, _hint: str) -> None:
        """Callback for opening files from command line or file manager.

        Args:
            app: The application instance
            files: List of GFile objects to open
            n_files: Number of files
            _hint: Hint string (usually empty, prefixed with _ to indicate unused)
        """
        try:
            # Load custom CSS
            load_css()

            # Extract file paths from GFile objects and categorize
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            pdf_paths = []
            image_paths = []

            for gfile in files:
                path = gfile.get_path()
                if path:
                    import os

                    ext = os.path.splitext(path)[1].lower()
                    if ext in image_extensions:
                        image_paths.append(path)
                    else:
                        pdf_paths.append(path)

            # Determine mode and launch appropriate window
            if image_paths and not pdf_paths:
                # Image-only mode: Launch ImageOcrWindow
                first_image = image_paths[0]
                win = ImageOcrWindow(app, image_path=first_image)
                win.present()
                logger.info(f"Opened image OCR window with: {first_image}")

            elif pdf_paths or not image_paths:
                # Normal PDF mode or mixed (prioritize PDF window)
                # Check if we already have a window open
                win = self.get_active_window()

                # If active window is ImageOcrWindow, create new BigOcrPdfWindow
                if not win or isinstance(win, ImageOcrWindow):
                    win = BigOcrPdfWindow(app)

                # Show the window
                win.present()

                # Add PDF files to the application queue
                if pdf_paths:

                    def add_files_when_ready():
                        try:
                            if hasattr(win, "settings"):
                                added = win.settings.add_files(pdf_paths)
                                if added > 0:
                                    logger.info(f"Added {added} file(s) from command line")
                                    # Refresh the file list UI
                                    if hasattr(win, "update_file_info"):
                                        win.update_file_info()
                        except Exception as e:
                            logger.error(f"Error adding files: {e}")
                        return False  # Don't repeat

                    # Use a small delay to ensure the window is fully initialized
                    GLib.timeout_add(100, add_files_when_ready)

            logger.info(_(f"Opened {n_files} file(s)"))

        except Exception as e:
            logger.error(f"{_('Error opening files')}: {e}")

    def on_about_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Show about dialog.

        Args:
            _action: The action that triggered this callback (unused)
            _param: Action parameters (unused)
        """
        # Get active window as the parent
        win = self.get_active_window()

        # Create an about dialog following GNOME guidelines
        about = Adw.AboutWindow(transient_for=win)
        about.set_application_name(APP_NAME)
        about.set_version(APP_VERSION)
        about.set_developer_name(_("BigLinux Team"))
        about.set_license_type(Gtk.License.GPL_3_0)
        about.set_comments(APP_DESCRIPTION)
        about.set_website(APP_WEBSITE)
        about.set_issue_url(APP_ISSUES)

        # Legal information
        about.add_legal_section(
            _("Interface"),
            None,
            Gtk.License.GPL_3_0,
            None,
        )
        about.add_legal_section(
            _("Third-party Components"),
            _(
                "The OCR engine and other libraries used by this application "
                "are independent projects, each distributed under its own license."
            ),
            Gtk.License.CUSTOM,
            None,
        )

        # Use app icon for the about dialog
        about.set_application_icon(APP_ICON_NAME)

        # Add credits
        about.add_credit_section(_("Developers"), APP_DEVELOPERS)

        # Acknowledge base projects
        about.add_credit_section(
            _("Powered by"),
            [
                "RapidOCR https://github.com/RapidAI/RapidOCR",
                "PaddleOCR (PP-OCRv5) https://github.com/PaddlePaddle/PaddleOCR",
                "OpenCV https://opencv.org",
                "OpenVINO https://github.com/openvinotoolkit/openvino",
                "pikepdf https://github.com/pikepdf/pikepdf",
                "Pillow https://python-pillow.org",
            ],
        )

        # Show the about dialog
        about.present()

    def on_image_ocr_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Open the independent Image OCR window.

        Args:
            _action: The action that triggered this callback
            _param: Action parameters
        """
        win = ImageOcrWindow(self)
        win.present()
