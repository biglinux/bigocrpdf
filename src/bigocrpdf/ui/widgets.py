"""Widgets and helpers shared by the windows: clipboard, illustrations,
copyable failure details, the OCR-unavailable dialog, and the stylesheet.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
import os
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, GdkPixbuf, Gio, GLib, Gtk, Pango

from bigocrpdf.config import RESOURCES_DIR
from bigocrpdf.services.ocr.ppocr_native import WORKER
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow

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
    image = Gtk.Image(accessible_role=Gtk.AccessibleRole.PRESENTATION)
    if os.path.exists(path):
        image.set_from_pixbuf(GdkPixbuf.Pixbuf.new_from_file_at_scale(path, size, size, True))
    image.set_pixel_size(size)
    image.set_halign(Gtk.Align.CENTER)
    image.set_valign(Gtk.Align.CENTER)
    image.set_hexpand(False)
    image.set_vexpand(False)
    return image


def failure_details(details: str) -> Gtk.Widget:
    """Show failure causes as selectable text with a button that copies them."""
    label = Gtk.Label(
        label=details,
        selectable=True,
        wrap=True,
        wrap_mode=Pango.WrapMode.WORD_CHAR,
        xalign=0,
    )
    # A focused selectable label selects all its text, and the dialog passes
    # focus through it on the way to its default button; the button copies.
    label.set_focusable(False)
    scrolled = Gtk.ScrolledWindow(
        child=label,
        hscrollbar_policy=Gtk.PolicyType.NEVER,
        propagate_natural_height=True,
        max_content_height=240,
    )
    copy_button = Gtk.Button(label=_("Copy Details"), halign=Gtk.Align.CENTER)

    def copy(button: Gtk.Button) -> None:
        button.get_clipboard().set(details)
        button.set_label(_("Copied to clipboard"))

    copy_button.connect("clicked", copy)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    box.append(scrolled)
    box.append(copy_button)
    return box


def present_ocr_unavailable_dialog(
    parent: Gtk.Widget,
    reason: str,
    response_callback: Callable[[Adw.AlertDialog, str], None],
) -> Adw.AlertDialog:
    """Present the shared actionable error for an unavailable OCR engine.

    The cause names the engine's model folder, so it is shown for copying.
    """
    body = _("Install the {package} package, then restart the application.").format(package=WORKER)
    dialog = Adw.AlertDialog(heading=_("OCR is unavailable"), body=body)
    dialog.set_extra_child(
        failure_details(reason.strip() or _("The OCR engine could not be loaded."))
    )
    dialog.add_response("close", _("Close"))
    dialog.set_default_response("close")
    dialog.set_close_response("close")
    dialog.connect("response", response_callback)
    dialog.present(parent)
    return dialog


def launch_file(window: "BigOcrPdfWindow", path: str, *, in_folder: bool = False) -> None:
    """Open path in its default app, or show it selected in the file manager.

    The launcher passes the compositor an activation token for window, so the
    opened program is raised instead of appearing behind it.
    """
    launcher = Gtk.FileLauncher.new(Gio.File.new_for_path(path))

    def on_done(source: Gtk.FileLauncher, result: Gio.AsyncResult) -> None:
        try:
            if in_folder:
                source.open_containing_folder_finish(result)
            else:
                source.launch_finish(result)
        except GLib.Error as error:
            if error.matches(Gtk.dialog_error_quark(), Gtk.DialogError.DISMISSED):
                return
            logger.warning(f"Could not launch {path}: {error.message}")
            if in_folder:
                # Without a FileManager1 service, open the folder itself.
                launch_file(window, os.path.dirname(path))
                return
            message = _("Could not open {name}: {error}").format(
                name=os.path.basename(path) or path, error=error.message
            )
            window.ui.show_toast(message, timeout=6)

    if in_folder:
        launcher.open_containing_folder(window, None, on_done)
    else:
        launcher.launch(window, None, on_done)


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
