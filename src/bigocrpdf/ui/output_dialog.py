"""Educational dialog for configuring output settings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GdkPixbuf, Gtk

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


def _make_switch_sync(widget: Adw.SwitchRow, toggle: Gtk.Switch) -> None:
    """Bidirectional sync between a SwitchRow and a Switch."""

    def _on_toggle(switch, _pspec):
        if widget.get_active() != switch.get_active():
            widget.set_active(switch.get_active())

    def _on_row(row, _pspec):
        if toggle.get_active() != row.get_active():
            toggle.set_active(row.get_active())

    toggle.connect("notify::active", _on_toggle)
    widget.connect("notify::active", _on_row)


def _make_combo_sync(widget: Adw.ComboRow, dropdown: Gtk.DropDown) -> None:
    """Bidirectional sync between a ComboRow and a DropDown."""

    def _on_dropdown(drop, _pspec):
        if widget.get_selected() != drop.get_selected():
            widget.set_selected(drop.get_selected())

    def _on_row(row, _pspec):
        if dropdown.get_selected() != row.get_selected():
            dropdown.set_selected(row.get_selected())

    dropdown.connect("notify::selected", _on_dropdown)
    widget.connect("notify::selected", _on_row)


_OUTPUT_SETTINGS = [
    {
        "key": "image_quality",
        "type": "combo",
        "svg": "quality_bw.svg",
        "title": _("Image Quality"),
        "description": _(
            "Controls the compression applied to images inside the PDF. "
            "Lower quality means smaller files but some detail is lost. "
            "'Keep Original' preserves images exactly as they are.\n\n"
            "The last option, 'Black & White (JBIG2)', converts all pages "
            "to pure black and white using JBIG2 — the most compact format "
            "available. Ideal for text-only documents, but all color is lost."
        ),
    },
    {
        "key": "pdfa",
        "type": "switch",
        "svg": "pdfa.svg",
        "title": _("Export as PDF/A"),
        "description": _(
            "Creates an archival PDF designed for long-term storage. "
            "The file will open correctly on any device, now and "
            "in the future. Recommended for important documents."
        ),
    },
    {
        "key": "max_size",
        "type": "combo",
        "svg": "max_size.svg",
        "title": _("Maximum Output Size"),
        "description": _(
            "Sets a size limit for the final file. If the result is "
            "too large, it is automatically split into smaller numbered "
            "files (e.g. document-01.pdf). Useful for email or uploads."
        ),
    },
]


def show_output_settings_dialog(
    parent: Gtk.Widget,
    widgets: dict[str, Gtk.Widget],
) -> None:
    """Show the output settings configuration dialog."""
    dialog = Adw.Dialog()
    dialog.set_title(_("Output Settings"))
    dialog.set_content_width(680)
    dialog.set_content_height(800)
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
            "Configure how the final PDF is generated. "
            "These settings affect file size, quality, and compatibility."
        )
    )
    intro.set_wrap(True)
    intro.set_xalign(0)
    intro.set_margin_bottom(20)
    intro.add_css_class("dim-label")
    content.append(intro)

    for setting in _OUTPUT_SETTINGS:
        key = setting["key"]
        widget = widgets.get(key)
        if not widget:
            continue

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        card.add_css_class("card")
        card.set_margin_bottom(12)

        # Horizontal layout: [SVG] [text] [widget]
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
            _make_combo_sync(widget, dropdown)
            row.append(dropdown)

        card.append(row)
        content.append(card)

    scroll.set_child(content)
    toolbar.set_content(scroll)
    dialog.set_child(toolbar)
    dialog.present(parent)
