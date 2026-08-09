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

THEME_NAME = "bigocrpdf"

_registered = False


def _search_path_candidates() -> list[Path]:
    """Directories that may contain the bundled ``bigocrpdf`` icon theme.

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
        if (candidate / THEME_NAME / "index.theme").is_file():
            return candidate
    return None


def setup_icons(display=None) -> bool:
    """Make the bundled icons take precedence over the host icon theme.

    Must be called after a ``Gdk.Display`` exists -- that is, from the
    application ``startup`` handler -- and before any widget is built.

    Returns ``True`` when the bundled theme was registered.  A missing bundle is
    not fatal: the application simply keeps using the host theme.
    """
    global _registered
    if _registered:
        return True

    search_path = get_icon_search_path()
    if search_path is None:
        logger.warning("Bundled icon theme not found; falling back to the system theme")
        return False

    from gi.repository import Gdk

    display = display or Gdk.Display.get_default()
    if display is None:
        logger.warning("No display available; bundled icon theme not registered")
        return False

    icon_theme = Gtk.IconTheme.get_for_display(display)
    icon_theme.add_search_path(str(search_path))

    # Selecting the theme has to go through GtkSettings: calling
    # Gtk.IconTheme.set_theme_name() on the per-display instance is forbidden.
    # Our index.theme inherits Adwaita and hicolor, so names we do not ship
    # still resolve.
    settings = Gtk.Settings.get_for_display(display)
    if settings is not None:
        settings.set_property("gtk-icon-theme-name", THEME_NAME)

    _registered = True
    logger.debug("Registered bundled icon theme from %s", search_path)
    return True
