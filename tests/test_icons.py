"""The bundled icons must cover every name the interface asks for, without
displacing the host icon theme."""

import re
from pathlib import Path

import pytest

from bigocrpdf.utils import icons

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "bigocrpdf"
ICON_DIR = SRC_ROOT / "resources" / "icontheme" / icons.FALLBACK_THEME_NAME / "scalable" / "actions"

# Any string literal that looks like a symbolic icon name.  Icon names are
# always literals in this codebase, never built at runtime.
_ICON_NAME = re.compile(r"""["']([a-z0-9][a-z0-9-]*-symbolic)["']""")


def _referenced_icon_names() -> dict[str, list[str]]:
    """Map every symbolic icon name used in the sources to where it appears."""
    found: dict[str, list[str]] = {}
    for path in SRC_ROOT.rglob("*.py"):
        for name in _ICON_NAME.findall(path.read_text(encoding="utf-8")):
            found.setdefault(name, []).append(str(path.relative_to(SRC_ROOT)))
    return found


def test_every_referenced_icon_is_bundled():
    missing = {
        name: sites
        for name, sites in _referenced_icon_names().items()
        if not (ICON_DIR / f"{name}.svg").is_file()
    }
    assert not missing, (
        "Icon names with no bundled SVG (they would fall back to the host theme "
        f"and may render as image-missing): {missing}"
    )


def test_no_unused_bundled_icons():
    referenced = set(_referenced_icon_names())
    unused = sorted(p.stem for p in ICON_DIR.glob("*.svg") if p.stem not in referenced)
    assert not unused, f"Bundled icons no longer referenced anywhere: {unused}"


def test_the_icons_ship_as_hicolor():
    """Which theme they belong to is the whole mechanism.

    Every icon theme's lookup chain ends at hicolor, so shipping them there
    fills the host theme's gaps without displacing it. Shipping them under a
    private theme name instead only worked by selecting that theme, which threw
    the user's own icons away.
    """
    assert icons.FALLBACK_THEME_NAME == "hicolor"

    index = (ICON_DIR.parent.parent / "index.theme").read_text(encoding="utf-8")

    assert "Inherits=" not in index, "hicolor is the end of the chain; it inherits nothing"


def test_search_path_resolves_to_the_bundled_icons():
    search_path = icons.get_icon_search_path()
    assert search_path is not None
    assert (search_path / icons.FALLBACK_THEME_NAME / "index.theme").is_file()


def test_setup_does_not_choose_the_icon_theme_for_the_user(monkeypatch):
    """The host theme must survive startup.

    Selecting the bundle through gtk-icon-theme-name made the interface look
    the same everywhere by ignoring the user's choice entirely: a Papirus or
    Breeze user saw none of their own icons.
    """
    monkeypatch.setattr(icons, "_registered", False)
    gdk = pytest.importorskip("gi.repository.Gdk")
    gtk = pytest.importorskip("gi.repository.Gtk")
    display = gdk.Display.get_default()
    if display is None:
        pytest.skip("no display")
    settings = gtk.Settings.get_for_display(display)
    before = settings.get_property("gtk-icon-theme-name")

    icons.setup_icons(display)

    assert settings.get_property("gtk-icon-theme-name") == before


def test_setup_icons_without_a_display_is_not_fatal(monkeypatch):
    """Headless callers (the CLI) must not crash on import or setup."""
    monkeypatch.setattr(icons, "_registered", False)
    gdk = pytest.importorskip("gi.repository.Gdk")
    monkeypatch.setattr(gdk.Display, "get_default", staticmethod(lambda: None))
    assert icons.setup_icons() is False
