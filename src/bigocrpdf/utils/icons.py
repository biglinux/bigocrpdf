"""Bundled icon theme registration.

The application ships its own copy of every symbolic icon it references, so the
interface looks identical regardless of the icon theme installed on the host.
This matters most for relocatable builds such as AppImage, where the host theme
is frequently missing the names we use and GTK falls back to a generic
"image missing" icon.

Registering a private icon theme (instead of loading each SVG by path) keeps
every existing ``icon-name`` call site working untouched, including widgets that
accept only a name -- ``Adw.ButtonContent``, ``Adw.StatusPage``, ``Gio.MenuItem``
and ``Gio.Notification``.
"""

import os
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk

from bigocrpdf.utils.logger import logger

FALLBACK_THEME_NAME = "hicolor"

_registered = False


def _search_path_candidates() -> list[Path]:
    """Directories that may contain the bundled fallback icons.

    The package-relative path covers both a normal install and an AppImage,
    since the resources travel with the Python package.  The remaining entries
    let a distribution or a developer override the icons without rebuilding.
    """
    candidates = [Path(__file__).resolve().parent.parent / "resources" / "icontheme"]

    override = os.environ.get("BIGOCRPDF_ICON_DIR")
    if override:
        candidates.insert(0, Path(override))

    appdir = os.environ.get("APPDIR")
    if appdir:
        candidates.append(Path(appdir) / "usr/share/bigocrpdf/icons")

    candidates.append(Path("/usr/share/bigocrpdf/icons"))
    return candidates


def get_icon_search_path() -> Path | None:
    """Return the first candidate that actually holds the bundled theme."""
    for candidate in _search_path_candidates():
        if (candidate / FALLBACK_THEME_NAME / "index.theme").is_file():
            return candidate
    return None


def setup_icons(display=None) -> bool:
    """Offer the bundled icons as a last resort, without displacing the host theme.

    The icons ship as a ``hicolor`` theme because every icon theme's lookup
    chain ends there. Adding this directory to the search path therefore fills
    the gaps -- measured on a host running ``bigicons-papient``, four names the
    interface needs resolve to nothing without it and resolve with it -- while
    any name the user's own theme provides still comes from their theme.

    This used to select the bundle through ``gtk-icon-theme-name`` instead,
    which did make the interface identical everywhere, at the price of ignoring
    the user's chosen icons entirely: the chain became bundle -> Adwaita ->
    hicolor, and a Papirus or Breeze user never saw one of their own icons.

    Must be called after a ``Gdk.Display`` exists -- that is, from the
    application ``startup`` handler -- and before any widget is built.

    Returns ``True`` when the bundled icons were registered.  A missing bundle
    is not fatal: the application simply keeps using the host theme.
    """
    global _registered
    if _registered:
        return True

    search_path = get_icon_search_path()
    if search_path is None:
        logger.warning("Bundled fallback icons not found; only the host theme will be used")
        return False

    from gi.repository import Gdk

    display = display or Gdk.Display.get_default()
    if display is None:
        logger.warning("No display available; bundled fallback icons not registered")
        return False

    icon_theme = Gtk.IconTheme.get_for_display(display)
    # Appended, so the host's own directories keep their priority.
    icon_theme.add_search_path(str(search_path))

    _registered = True
    logger.debug("Registered bundled fallback icons from %s", search_path)
    return True
