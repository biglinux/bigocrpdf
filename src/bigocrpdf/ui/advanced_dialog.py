"""Educational dialog for configuring advanced OCR settings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GdkPixbuf, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _

if TYPE_CHECKING:
    pass

_ILLUSTRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "illustrations")


def _load_svg_picture(filename: str, size: int = 92) -> Gtk.Image:
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
    return image


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
    dialog.set_content_width(680)
    dialog.set_content_height(700)
    dialog.set_presentation_mode(Adw.DialogPresentationMode.FLOATING)

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
        key = setting["key"]
        widget = widgets.get(key)
        if not widget:
            continue

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        card.add_css_class("card")
        card.set_margin_bottom(12)

        # Horizontal layout: [SVG] [text] [control]
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        row.set_margin_top(16)
        row.set_margin_bottom(16)
        row.set_margin_start(16)
        row.set_margin_end(16)

        # Left: SVG illustration
        picture = _load_svg_picture(setting["svg"])
        picture.set_valign(Gtk.Align.CENTER)
        row.append(picture)

        # Center: title + description
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

        row.append(text_box)

        # Right: switch or dropdown
        if setting["type"] == "switch":
            toggle = Gtk.Switch()
            toggle.set_active(widget.get_active())
            toggle.set_valign(Gtk.Align.CENTER)
            set_a11y_label(toggle, setting["title"])

            def _make_switch_sync(w, tgl):
                def _on_toggle(switch, _pspec):
                    if w.get_active() != switch.get_active():
                        w.set_active(switch.get_active())

                def _on_row(row, _pspec):
                    if tgl.get_active() != row.get_active():
                        tgl.set_active(row.get_active())

                tgl.connect("notify::active", _on_toggle)
                w.connect("notify::active", _on_row)

            _make_switch_sync(widget, toggle)
            row.append(toggle)

        elif setting["type"] == "combo":
            source_model = widget.get_model()
            n_items = source_model.get_n_items()
            items = [source_model.get_string(i) for i in range(n_items)]
            model = Gtk.StringList.new(items)
            dropdown = Gtk.DropDown(model=model)
            dropdown.set_selected(widget.get_selected())
            dropdown.set_valign(Gtk.Align.CENTER)
            set_a11y_label(dropdown, setting["title"])

            def _make_combo_sync(w, dd):
                def _on_dropdown(drop, _pspec):
                    if w.get_selected() != drop.get_selected():
                        w.set_selected(drop.get_selected())

                def _on_row(row, _pspec):
                    if dd.get_selected() != row.get_selected():
                        dd.set_selected(row.get_selected())

                dd.connect("notify::selected", _on_dropdown)
                w.connect("notify::selected", _on_row)

            _make_combo_sync(widget, dropdown)
            row.append(dropdown)

        card.append(row)
        content.append(card)

    scroll.set_child(content)
    toolbar.set_content(scroll)
    dialog.set_child(toolbar)
    dialog.present(parent)
