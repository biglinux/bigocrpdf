"""
BigOcrPdf - UI Widgets

This module contains shared widgets and UI components used across the application.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Gio, Gdk

import os

from ..utils.logger import logger
from ..config import RESOURCES_DIR


def load_css() -> None:
    """Load custom CSS styles for the application"""
    css_provider = Gtk.CssProvider()
    try:
        css_file = os.path.join(RESOURCES_DIR, "styles.css")
        css_provider.load_from_file(Gio.File.new_for_path(css_file))
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
        logger.info("Custom CSS styles loaded successfully")
    except Exception as e:
        logger.error(f"Error loading CSS styles: {e}")
