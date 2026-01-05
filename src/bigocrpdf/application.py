"""
BigOcrPdf - Application Module

This module contains the main application class for the BigOcrPdf application.
"""

from typing import Any
import os

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
)
from bigocrpdf.ui.widgets import load_css
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.window import BigOcrPdfWindow


class BigOcrPdfApp(Adw.Application):
    """Application class for BigOcrPdf."""

    def __init__(self) -> None:
        """Initialize the application."""
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_OPEN)

        # Store files to be opened
        self._pending_files: list = []

        # Add command line handling
        self.add_main_option(
            "version",
            ord("v"),
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            _("Print version information and exit"),
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

        # Set keyboard shortcuts
        self.set_accels_for_action("app.quit", ["<Ctrl>q"])

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
                # Create the main window
                win = BigOcrPdfWindow(app)

            # Show the window
            win.present()

            # Check if we should show the welcome dialog
            if hasattr(win, "should_show_welcome_dialog") and win.should_show_welcome_dialog():
                # Use a small delay to ensure the window is fully drawn
                GLib.timeout_add(300, lambda: win.show_welcome_dialog())

            logger.info(_("Application started successfully"))

        except Exception as e:
            logger.error(f"{_('Error activating application')}: {e}")
            error_dialog = Gtk.AlertDialog()
            error_dialog.set_message(_("Error starting application"))
            error_dialog.set_detail(str(e))
            error_dialog.show()

    def on_open(self, app: Adw.Application, files: list, n_files: int, hint: str) -> None:
        """Callback for opening files from command line or file manager.

        Args:
            app: The application instance
            files: List of GFile objects to open
            n_files: Number of files
            hint: Hint string (usually empty)
        """
        try:
            # Load custom CSS
            load_css()

            # Check if we already have a window open
            win = self.get_active_window()
            if not win:
                # Create the main window
                win = BigOcrPdfWindow(app)

            # Extract file paths from GFile objects
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            pdf_paths = []
            image_paths = []

            for gfile in files:
                path = gfile.get_path()
                if path:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in image_extensions:
                        image_paths.append(path)
                    else:
                        pdf_paths.append(path)

            # Determine mode and visibility
            if image_paths and not pdf_paths:
                # Image-only mode: Keep main window hidden
                win.image_only_mode = True
            else:
                # Normal mode: Show main window
                win.image_only_mode = False
                win.present()

            # Handle image files (OCR extraction)
            if image_paths:
                first_image = image_paths[0]

                def open_image_when_ready():
                    try:
                        if hasattr(win, "open_image_for_ocr"):
                            win.open_image_for_ocr(first_image)
                    except Exception as e:
                        logger.error(f"Error opening image: {e}")
                    return False

                # Delay slightly to ensure window initialization
                GLib.timeout_add(200, open_image_when_ready)

            # Add PDF files to the application queue
            if pdf_paths:

                def add_files_when_ready():
                    try:
                        logger.info(f"Adding {len(pdf_paths)} PDF files...")
                        if hasattr(win, "settings"):
                            added = win.settings.add_files(pdf_paths)
                            if added > 0:
                                logger.info(f"Added {added} file(s) from command line")
                                if hasattr(win, "update_file_info"):
                                    win.update_file_info()
                            else:
                                logger.warning("No files added (check mime types)")
                    except Exception as e:
                        logger.error(f"Error adding files: {e}")
                    return False

                # Use a larger delay to ensure the window/settings are fully engaged
                GLib.timeout_add(300, add_files_when_ready)

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

        # Use app icon for the about dialog
        about.set_application_icon(APP_ICON_NAME)

        # Add credits
        about.add_credit_section(_("Developers"), APP_DEVELOPERS)

        # Show the about dialog
        about.present()
