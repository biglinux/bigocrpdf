"""
BigOcrPdf - UI Widgets

This module contains shared widgets and UI components used across the application.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
import os

from gi.repository import Gdk, Gio, Gtk

from bigocrpdf.config import RESOURCES_DIR
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


def load_css() -> bool:
    """Load custom CSS styles for the application

    Returns:
        True if CSS loaded successfully, False otherwise
    """
    css_provider = Gtk.CssProvider()
    try:
        css_file = os.path.join(RESOURCES_DIR, "styles.css")

        # Check if file exists
        if not os.path.exists(css_file):
            logger.error(_("CSS file not found: {0}").format(css_file))
            return False

        css_provider.load_from_file(Gio.File.new_for_path(css_file))
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
        logger.info(_("Custom CSS styles loaded successfully"))
        return True
    except Exception as e:
        logger.error(_("Error loading CSS styles: {0}").format(e))
        return False


def format_markup(text: str) -> str:
    """Convert HTML-like markup to properly displayed markup

    Args:
        text: Text with HTML-like markup

    Returns:
        Text with proper markup for GTK widgets
    """
    if not text:
        return ""

    # Ensure bullet points have proper spacing
    text = text.replace("•", "• ")

    # Process the text for Adw.AlertDialog and other GTK widgets
    # that require explicit markup handling

    # Check if we need to wrap in markup tags
    if "<b>" in text or "<i>" in text or "<u>" in text:
        # Text already contains markup tags
        # For MessageDialog, we might need to explicitly tell it to use markup
        return f"{text}"

    # No markup found, return as is
    return text
