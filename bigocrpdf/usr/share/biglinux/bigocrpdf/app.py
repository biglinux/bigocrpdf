"""
BigOcrPdf - Application Module

This module contains the main application class for the BigOcrPdf application.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio

import os
from typing import Any

from .config import (
    APP_NAME,
    APP_ID,
    APP_VERSION,
    APP_DESCRIPTION,
    APP_WEBSITE,
    APP_ISSUES,
    APP_DEVELOPERS,
    ICON_PATH,
)
from .window import BigOcrPdfWindow
from .utils.logger import logger
from .ui.widgets import load_css
from .utils.i18n import _


class BigOcrPdfApp(Adw.Application):
    """Application class for BigOcrPdf"""

    def __init__(self):
        """Initialize the application"""
        super().__init__(application_id=APP_ID)
        self.connect("activate", self.on_activate)

    def on_activate(self, app: Adw.Application) -> None:
        """Callback for application activation

        Args:
            app: The application instance
        """
        # Load custom CSS
        load_css()

        # Create the main window
        win = BigOcrPdfWindow(app)
        win.present()

    def on_about_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Show about dialog

        Args:
            _action: The action that triggered this callback (unused)
            _param: Action parameters (unused)
        """
        # Create an about dialog following GNOME guidelines
        about = Adw.AboutWindow(transient_for=self)
        about.set_application_name(APP_NAME)
        about.set_version(APP_VERSION)
        about.set_developer_name(_("Big Linux Team"))
        about.set_license_type(Gtk.License.GPL_3_0)
        about.set_comments(APP_DESCRIPTION)
        about.set_website(APP_WEBSITE)
        about.set_issue_url(APP_ISSUES)

        # Use app icon for the about dialog
        about.set_application_icon(ICON_PATH)

        # Add credits
        about.add_credit_section(_("Developers"), APP_DEVELOPERS)

        # Show the about dialog
        about.present()
