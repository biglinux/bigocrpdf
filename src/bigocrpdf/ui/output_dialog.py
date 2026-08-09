"""Educational dialog for configuring output settings."""

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
        "key": "page_layout",
        "type": "combo",
        "svg": "full_resolution.svg",
        "title": _("Page Layout"),
        "description": _(
            "Sets how PDF viewers arrange the pages when the file is opened: "
            "one page at a time, a continuous vertical scroll, or two pages "
            "side by side. 'Default' lets each viewer choose. Use this if "
            "pages appear at inconsistent sizes on phones."
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
        picture = load_svg_picture(setting["svg"])
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
        if setting["type"] == "switch" and isinstance(widget, Adw.SwitchRow):
            toggle = Gtk.Switch()
            toggle.set_valign(Gtk.Align.CENTER)
            set_a11y_label(toggle, setting["title"])
            widget.bind_property(
                "active",
                toggle,
                "active",
                GObject.BindingFlags.BIDIRECTIONAL | GObject.BindingFlags.SYNC_CREATE,
            )
            row.append(toggle)

        elif setting["type"] == "combo" and isinstance(widget, Adw.ComboRow):
            source_model = widget.get_model()
            if not isinstance(source_model, Gtk.StringList):
                continue
            n_items = source_model.get_n_items()
            items = [source_model.get_string(i) or "" for i in range(n_items)]
            model = Gtk.StringList.new(items)
            dropdown = Gtk.DropDown(model=model)
            dropdown.set_valign(Gtk.Align.CENTER)
            set_a11y_label(dropdown, setting["title"])
            widget.bind_property(
                "selected",
                dropdown,
                "selected",
                GObject.BindingFlags.BIDIRECTIONAL | GObject.BindingFlags.SYNC_CREATE,
            )
            row.append(dropdown)

        card.append(row)
        content.append(card)

    scroll.set_child(content)
    toolbar.set_content(scroll)
    dialog.set_child(toolbar)
    dialog.present(parent)
