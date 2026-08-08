"""Widgets newer than the supported floor must degrade, not crash.

An AppImage runs against whatever libadwaita its build container shipped, so
both branches have to work.  Each test exercises the fallback by forcing the
capability flags off, which is what an older stack looks like to the code.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Adw, Gtk

from bigocrpdf.utils import adw_compat

GROUPS = (
    ("File", (("Add files", "<Control>o"), ("Quit", "<Control>q"))),
    ("General", (("About", "F1"),)),
)


@pytest.fixture
def legacy_stack(monkeypatch):
    """Make every post-1.5 widget look absent."""
    monkeypatch.setattr(adw_compat, "HAS_ADW_SPINNER", False)
    monkeypatch.setattr(adw_compat, "HAS_ADW_WRAP_BOX", False)
    monkeypatch.setattr(adw_compat, "HAS_ADW_SHORTCUTS_DIALOG", False)


def test_spinner_falls_back_to_gtk_and_is_already_spinning(legacy_stack):
    spinner = adw_compat.create_spinner(40)
    assert isinstance(spinner, Gtk.Spinner)
    # Adw.Spinner animates on its own; Gtk.Spinner does not, so the fallback
    # must have been started or the user sees a frozen indicator.
    assert spinner.get_spinning()
    assert spinner.get_property("width-request") == 40


def test_wrap_box_fallback_accepts_children_through_append(legacy_stack):
    box = adw_compat.create_wrap_box(child_spacing=32, line_spacing=16, margin_start=16)
    assert isinstance(box, Gtk.FlowBox)

    first, second = Gtk.Label(label="a"), Gtk.Label(label="b")
    box.append(first)
    box.append(second)

    # FlowBox wraps each child in a FlowBoxChild, so walk one level down.
    labels = []
    child = box.get_first_child()
    while child is not None:
        labels.append(child.get_child())
        child = child.get_next_sibling()
    assert labels == [first, second]


def test_wrap_box_fallback_keeps_spacing_and_margins(legacy_stack):
    box = adw_compat.create_wrap_box(child_spacing=32, line_spacing=16, margin_start=16)
    assert box.get_column_spacing() == 32
    assert box.get_row_spacing() == 16
    assert box.get_margin_start() == 16
    assert box.get_selection_mode() == Gtk.SelectionMode.NONE


def test_shortcuts_dialog_fallback_lists_every_shortcut(legacy_stack):
    dialog = adw_compat.build_shortcuts_dialog(GROUPS)
    assert isinstance(dialog, Adw.PreferencesDialog)

    accelerators = []
    titles = []

    def walk(widget):
        if isinstance(widget, Gtk.ShortcutLabel):
            accelerators.append(widget.get_accelerator())
        if isinstance(widget, Adw.ActionRow):
            titles.append(widget.get_title())
        child = widget.get_first_child()
        while child is not None:
            walk(child)
            child = child.get_next_sibling()

    walk(dialog.get_child() or dialog)

    assert sorted(accelerators) == sorted(["<Control>o", "<Control>q", "F1"])
    assert sorted(titles) == sorted(["Add files", "Quit", "About"])


def test_native_path_is_used_when_available():
    """On a current stack the native widgets must still be preferred."""
    if adw_compat.HAS_ADW_SHORTCUTS_DIALOG:
        assert isinstance(adw_compat.build_shortcuts_dialog(GROUPS), Adw.ShortcutsDialog)
    if adw_compat.HAS_ADW_SPINNER:
        assert isinstance(adw_compat.create_spinner(), Adw.Spinner)
    if adw_compat.HAS_ADW_WRAP_BOX:
        assert isinstance(adw_compat.create_wrap_box(), Adw.WrapBox)


def test_no_unguarded_use_of_widgets_above_the_floor():
    """Post-1.5 widgets may only be touched inside the compat module."""
    from pathlib import Path

    import bigocrpdf

    src_root = Path(bigocrpdf.__file__).parent
    guarded = {"Adw.ShortcutsDialog", "Adw.ShortcutsSection", "Adw.ShortcutsItem"}
    guarded |= {"Adw.Spinner", "Adw.WrapBox"}

    offenders = {}
    for path in src_root.rglob("*.py"):
        if path.name == "adw_compat.py":
            continue
        text = path.read_text(encoding="utf-8")
        hits = sorted(symbol for symbol in guarded if symbol in text)
        if hits:
            offenders[str(path.relative_to(src_root))] = hits

    assert not offenders, (
        "These widgets need libadwaita newer than the supported floor and must go "
        f"through utils/adw_compat.py: {offenders}"
    )


def test_declared_floor_matches_what_the_fallbacks_support():
    import bigocrpdf

    assert bigocrpdf._MIN_ADW_VERSION <= (1, 5)
    assert bigocrpdf._MIN_GTK_VERSION <= (4, 14)
