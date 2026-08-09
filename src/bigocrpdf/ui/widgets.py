"""
BigOcrPdf - UI Widgets

This module contains shared widgets and UI components used across the application.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
import os
from collections.abc import Callable
from urllib.parse import unquote, urlparse

gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, GdkPixbuf, Gio, GLib, Gtk

from bigocrpdf.config import RESOURCES_DIR
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

_ILLUSTRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "illustrations")
_css_provider: Gtk.CssProvider | None = None


def get_default_clipboard() -> Gdk.Clipboard | None:
    """Return the current display clipboard when a graphical display exists."""
    display = Gdk.Display.get_default()
    return display.get_clipboard() if display is not None else None


def parse_clipboard_file_paths(raw: str) -> list[str]:
    """Return existing local files from a copied-file clipboard payload."""
    file_paths: list[str] = []
    for raw_line in raw.strip().splitlines():
        line = raw_line.strip()
        if not line or line in ("copy", "cut"):
            continue
        parsed = urlparse(line)
        if parsed.scheme == "file":
            if parsed.hostname not in (None, "", "localhost"):
                continue
            path = unquote(parsed.path)
        elif parsed.scheme == "":
            path = line
        else:
            continue
        if os.path.isfile(path):
            file_paths.append(path)
    return file_paths


def load_svg_picture(filename: str, size: int = 92) -> Gtk.Image:
    """Load an SVG illustration rendered to a fixed pixel size."""
    path = os.path.join(_ILLUSTRATIONS_DIR, filename)
    if os.path.exists(path):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, size, size, True)
        image = Gtk.Image.new_from_pixbuf(pixbuf)
    else:
        image = Gtk.Image()
    image.set_pixel_size(size)
    image.set_halign(Gtk.Align.CENTER)
    image.set_valign(Gtk.Align.CENTER)
    image.set_hexpand(False)
    image.set_vexpand(False)
    image.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
    return image


def present_ocr_unavailable_dialog(
    parent: Gtk.Widget,
    reason: str,
    response_callback: Callable[[Adw.AlertDialog, str], None],
) -> Adw.AlertDialog:
    """Present the shared actionable error for an unavailable OCR engine."""
    detail = reason.strip() or _("The OCR engine could not be loaded.")
    body = _(
        "{reason}\n\nInstall or repair the required OCR packages, then restart the application."
    ).format(reason=detail)
    dialog = Adw.AlertDialog(heading=_("OCR is unavailable"), body=body)
    dialog.add_response("close", _("Close"))
    dialog.set_default_response("close")
    dialog.set_close_response("close")
    dialog.connect("response", response_callback)
    dialog.present(parent)
    return dialog


def load_css() -> bool:
    """Load custom CSS styles for the application."""
    global _css_provider

    if _css_provider is not None:
        return True

    css_file = os.path.join(RESOURCES_DIR, "styles.css")
    if not os.path.exists(css_file):
        logger.error(_("CSS file not found: {0}").format(css_file))
        return False

    display = Gdk.Display.get_default()
    if display is None:
        logger.error(_("Error loading CSS styles: {0}").format("display is unavailable"))
        return False

    css_provider = Gtk.CssProvider()
    try:
        css_provider.load_from_file(Gio.File.new_for_path(css_file))
    except GLib.Error as error:
        logger.error(_("Error loading CSS styles: {0}").format(error))
        return False

    Gtk.StyleContext.add_provider_for_display(
        display,
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
    _css_provider = css_provider
    logger.info(_("Custom CSS styles loaded successfully"))
    return True
