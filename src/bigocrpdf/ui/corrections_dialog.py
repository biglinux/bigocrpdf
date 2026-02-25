"""Educational dialog for configuring image correction settings."""

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


_CORRECTIONS = [
    {
        "key": "deskew",
        "svg": "deskew.svg",
        "title": _("Straighten Pages"),
        "description": _(
            "Fixes pages that were scanned at a slight angle. "
            "The text lines become perfectly horizontal."
        ),
    },
    {
        "key": "dewarp",
        "svg": "dewarp.svg",
        "title": _("Fix Curved Text"),
        "description": _(
            "Corrects wavy or curved text that appears when "
            "photographing a book or a bent document."
        ),
    },
    {
        "key": "perspective",
        "svg": "perspective.svg",
        "title": _("Fix Perspective"),
        "description": _(
            "Corrects pages that look like a trapezoid because "
            "the photo was taken at an angle instead of straight above."
        ),
    },
    {
        "key": "orientation",
        "svg": "autorotate.svg",
        "title": _("Fix Page Direction"),
        "description": _(
            "Detects and fixes pages that are upside-down or sideways, "
            "automatically rotating them to the correct orientation."
        ),
    },
    {
        "key": "scanner",
        "svg": "scanner.svg",
        "title": _("Enhance Readability"),
        "description": _(
            "Makes text darker and the background lighter, "
            "producing a cleaner result similar to a high-quality scan."
        ),
    },
    {
        "key": "enhance_embedded",
        "svg": "enhance_embedded.svg",
        "title": _("Enhance Embedded Images"),
        "description": _(
            "Apply image corrections (deskew, dewarp, perspective) to "
            "images inside pages that already contain text. Normally "
            "skipped because these images are already well-aligned. "
            "Enable only if embedded images appear crooked or distorted."
        ),
    },
]


def show_image_corrections_dialog(
    parent: Gtk.Widget,
    switches: dict[str, Adw.SwitchRow],
) -> None:
    """Show the image corrections configuration dialog.

    Args:
        parent: Parent widget for the dialog
        switches: Dict mapping correction key to its SwitchRow widget
    """
    dialog = Adw.Dialog()
    dialog.set_title(_("Image Corrections"))
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
            "These corrections automatically improve your scanned "
            "or photographed documents. Each one fixes a specific problem."
        )
    )
    intro.set_wrap(True)
    intro.set_xalign(0)
    intro.set_margin_bottom(20)
    intro.add_css_class("dim-label")
    content.append(intro)

    for correction in _CORRECTIONS:
        key = correction["key"]
        switch_row = switches.get(key)
        if not switch_row:
            continue

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        card.add_css_class("card")
        card.set_margin_bottom(12)

        # Horizontal layout: [SVG] [text] [switch]
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        row.set_margin_top(16)
        row.set_margin_bottom(16)
        row.set_margin_start(16)
        row.set_margin_end(16)

        # Left: SVG illustration
        picture = _load_svg_picture(correction["svg"])
        picture.set_valign(Gtk.Align.CENTER)
        row.append(picture)

        # Center: title + description
        text_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        text_box.set_hexpand(True)
        text_box.set_valign(Gtk.Align.CENTER)

        title_label = Gtk.Label(label=correction["title"])
        title_label.add_css_class("heading")
        title_label.set_halign(Gtk.Align.START)
        title_label.set_wrap(True)
        text_box.append(title_label)

        desc = Gtk.Label(label=correction["description"])
        desc.set_wrap(True)
        desc.set_xalign(0)
        desc.add_css_class("dim-label")
        text_box.append(desc)

        row.append(text_box)

        # Right: switch
        toggle = Gtk.Switch()
        toggle.set_active(switch_row.get_active())
        toggle.set_valign(Gtk.Align.CENTER)
        set_a11y_label(toggle, correction["title"])

        # Sync toggle ↔ original switch row (bidirectional)
        def _make_sync(sw_row, tgl):
            def _on_toggle_changed(switch, _pspec):
                if sw_row.get_active() != switch.get_active():
                    sw_row.set_active(switch.get_active())

            def _on_row_changed(src_row, _pspec):
                if tgl.get_active() != src_row.get_active():
                    tgl.set_active(src_row.get_active())

            tgl.connect("notify::active", _on_toggle_changed)
            sw_row.connect("notify::active", _on_row_changed)

        _make_sync(switch_row, toggle)
        row.append(toggle)

        card.append(row)
        content.append(card)

    scroll.set_child(content)
    toolbar.set_content(scroll)
    dialog.set_child(toolbar)
    dialog.present(parent)
