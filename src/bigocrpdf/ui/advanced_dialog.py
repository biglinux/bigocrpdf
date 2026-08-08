"""Educational dialog for configuring advanced OCR settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GObject, Gtk

from bigocrpdf.ui.widgets import load_svg_picture
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _

if TYPE_CHECKING:
    pass

_ADVANCED_SETTINGS = [
    {
        "key": "ocr_precision",
        "type": "combo",
        "svg": "ocr_precision.svg",
        "title": _("OCR Precision"),
        "description": _(
            "How carefully the program reads text from your documents. "
            "Lower precision finds more text (good for blurry pages), "
            "while higher precision makes fewer mistakes but may miss faint text."
        ),
    },
    {
        "key": "replace_ocr",
        "type": "switch",
        "svg": "replace_ocr.svg",
        "title": _("Replace Existing OCR"),
        "description": _(
            "When enabled, redoes the text recognition even if the PDF "
            "already has searchable text. Use this when the existing "
            "text layer is incorrect or of poor quality."
        ),
    },
    {
        "key": "full_resolution",
        "type": "switch",
        "svg": "full_resolution.svg",
        "title": _("Full Resolution Detection"),
        "description": _(
            "Analyzes the image at full resolution for text detection. "
            "Finds more text in high-resolution scans but takes longer. "
            "The faster mode is good enough for most documents."
        ),
    },
]


def show_advanced_settings_dialog(
    parent: Gtk.Widget,
    widgets: dict[str, Gtk.Widget],
) -> None:
    """Show the advanced settings configuration dialog.

    Args:
        parent: Parent widget for the dialog
        widgets: Dict mapping setting key to its widget (SwitchRow or ComboRow)
    """
    dialog = Adw.Dialog()
    dialog.set_title(_("Advanced"))
    dialog.set_content_width(600)
    dialog.set_content_height(600)

    toolbar = Adw.ToolbarView()
    header = Adw.HeaderBar()
    toolbar.add_top_bar(header)

    scroll = Gtk.ScrolledWindow()
    scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

    content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
    content.set_margin_top(16)
    content.set_margin_bottom(24)
    content.set_margin_start(24)
    content.set_margin_end(24)

    intro = Gtk.Label()
    intro.set_text(
        _(
            "Fine-tune how OCR text recognition works. "
            "The default settings work well for most documents."
        )
    )
    intro.set_wrap(True)
    intro.set_xalign(0)
    intro.set_margin_bottom(20)
    intro.add_css_class("dim-label")
    content.append(intro)

    for setting in _ADVANCED_SETTINGS:
        widget = widgets.get(setting["key"])
        if widget:
            content.append(_advanced_setting_card(setting, widget))

    scroll.set_child(content)
    toolbar.set_content(scroll)
    dialog.set_child(toolbar)
    dialog.present(parent)


def _advanced_setting_card(setting: dict, widget: Gtk.Widget) -> Gtk.Box:
    card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
    card.add_css_class("card")
    card.set_margin_bottom(12)

    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
    row.set_margin_top(16)
    row.set_margin_bottom(16)
    row.set_margin_start(16)
    row.set_margin_end(16)

    picture = load_svg_picture(setting["svg"])
    picture.set_valign(Gtk.Align.CENTER)
    row.append(picture)
    row.append(_advanced_setting_text(setting))

    control = _advanced_setting_control(setting, widget)
    if control is not None:
        row.append(control)

    card.append(row)
    return card


def _advanced_setting_text(setting: dict) -> Gtk.Box:
    text_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    text_box.set_hexpand(True)
    text_box.set_valign(Gtk.Align.CENTER)

    title_label = Gtk.Label(label=setting["title"])
    title_label.add_css_class("heading")
    title_label.set_halign(Gtk.Align.START)
    title_label.set_wrap(True)
    text_box.append(title_label)

    desc = Gtk.Label(label=setting["description"])
    desc.set_wrap(True)
    desc.set_xalign(0)
    desc.add_css_class("dim-label")
    text_box.append(desc)
    return text_box


def _advanced_setting_control(setting: dict, widget: Gtk.Widget) -> Gtk.Widget | None:
    if setting["type"] == "switch":
        if not isinstance(widget, Adw.SwitchRow):
            return None
        toggle = Gtk.Switch()
        toggle.set_valign(Gtk.Align.CENTER)
        set_a11y_label(toggle, setting["title"])
        widget.bind_property(
            "active",
            toggle,
            "active",
            GObject.BindingFlags.BIDIRECTIONAL | GObject.BindingFlags.SYNC_CREATE,
        )
        return toggle

    if setting["type"] == "combo":
        if not isinstance(widget, Adw.ComboRow):
            return None
        source_model = widget.get_model()
        if not isinstance(source_model, Gtk.StringList):
            return None
        items = [source_model.get_string(i) or "" for i in range(source_model.get_n_items())]
        dropdown = Gtk.DropDown(model=Gtk.StringList.new(items))
        dropdown.set_valign(Gtk.Align.CENTER)
        set_a11y_label(dropdown, setting["title"])
        widget.bind_property(
            "selected",
            dropdown,
            "selected",
            GObject.BindingFlags.BIDIRECTIONAL | GObject.BindingFlags.SYNC_CREATE,
        )
        return dropdown

    return None
