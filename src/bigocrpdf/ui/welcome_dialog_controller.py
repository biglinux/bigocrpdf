"""Welcome dialog presentation and startup preference."""

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk

from bigocrpdf.config import APP_ICON_NAME
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.durable_writes import read_regular_file_bytes, write_text_atomically
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class WelcomeDialogController:
    """Own the welcome dialog and its startup preference."""

    def __init__(self, parent: Adw.ApplicationWindow, config_path: str | Path) -> None:
        self._parent = parent
        self._config_path = Path(config_path)

    def should_show(self) -> bool:
        """Return whether the welcome dialog should appear at startup."""
        try:
            content = read_regular_file_bytes(self._config_path).decode("utf-8").strip()
        except FileNotFoundError:
            try:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomically(self._config_path, "true")
                return True
            except OSError as exc:
                logger.error(f"Error creating welcome dialog config: {exc}")
                return True
        except (OSError, UnicodeError) as exc:
            logger.error(f"Error reading welcome dialog config: {exc}")
            return True
        return content.lower() != "false"

    def set_show(self, show: bool) -> None:
        """Persist whether the dialog should appear at startup."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomically(self._config_path, "true" if show else "false")
            logger.info(f"Set show welcome dialog: {show}")
        except OSError as exc:
            logger.error(f"Error setting welcome dialog config: {exc}")

    def show(self) -> None:
        """Present the welcome dialog as a centered modal."""
        dialog = Adw.Dialog()
        dialog.set_title("Big OCR PDF")
        dialog.set_content_width(650)
        dialog.set_content_height(500)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content_box.set_margin_top(24)
        content_box.set_margin_bottom(24)
        content_box.set_margin_start(36)
        content_box.set_margin_end(36)

        icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        icon.set_pixel_size(64)
        icon.set_margin_bottom(16)
        icon.set_halign(Gtk.Align.CENTER)
        icon.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
        content_box.append(icon)

        what_is_title = _("What is Big OCR PDF?")
        what_is = Gtk.Label(label=what_is_title)
        what_is.add_css_class("title-2")
        what_is.set_halign(Gtk.Align.CENTER)
        what_is.set_margin_bottom(14)
        set_a11y_label(what_is, what_is_title)
        content_box.append(what_is)

        what_is_desc = Gtk.Label()
        what_is_desc.set_text(
            _(
                "Big OCR PDF adds optical character recognition to your PDF files, "
                "making them searchable and allowing you to select and copy text "
                "from scanned documents."
            )
        )
        what_is_desc.set_wrap(True)
        what_is_desc.set_justify(Gtk.Justification.LEFT)
        what_is_desc.set_halign(Gtk.Align.START)
        what_is_desc.set_margin_bottom(16)
        what_is_desc.set_max_width_chars(65)
        content_box.append(what_is_desc)

        benefits_title = _("Benefits of using Big OCR PDF:")
        benefits = Gtk.Label(label=benefits_title)
        benefits.add_css_class("heading")
        benefits.set_halign(Gtk.Align.START)
        benefits.set_margin_bottom(8)
        set_a11y_label(benefits, benefits_title)
        content_box.append(benefits)

        benefits_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        benefits_list.set_halign(Gtk.Align.START)
        benefits_list.set_margin_bottom(16)

        benefit_items = [
            (_("Search"), _("Search through your scanned PDF documents")),
            (_("Copy text"), _("Copy text from images and scanned documents")),
            (_("Images"), _("Add image files and generate new PDFs from them")),
            (_("Edit PDF"), _("Rearrange, rotate or remove pages before processing")),
            (_("Batch processing"), _("Process multiple files at once")),
            (
                _("Auto-correction"),
                _("Automatically correct page alignment and rotation"),
            ),
        ]

        for title, description in benefit_items:
            benefit_label = Gtk.Label()
            escaped_title = GLib.markup_escape_text(title)
            escaped_description = GLib.markup_escape_text(description)
            benefit_label.set_markup(f"• <b>{escaped_title}:</b> {escaped_description}")
            benefit_label.set_wrap(True)
            benefit_label.set_halign(Gtk.Align.START)
            benefit_label.set_xalign(0)
            benefit_label.set_margin_bottom(3)
            benefits_list.append(benefit_label)

        content_box.append(benefits_list)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(4)
        separator.set_margin_bottom(16)
        content_box.append(separator)

        bottom_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        show_at_startup_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        show_at_startup_box.set_halign(Gtk.Align.FILL)

        show_at_startup_label = Gtk.Label(label=_("Show this dialog at startup"))
        show_at_startup_label.set_halign(Gtk.Align.START)
        show_at_startup_label.set_hexpand(True)

        show_at_startup_switch = Gtk.Switch()
        show_at_startup_switch.set_active(self.should_show())
        show_at_startup_switch.set_valign(Gtk.Align.CENTER)
        show_at_startup_switch.set_halign(Gtk.Align.END)
        set_a11y_label(show_at_startup_switch, _("Show this dialog at startup"))

        show_at_startup_box.append(show_at_startup_label)
        show_at_startup_box.append(show_at_startup_switch)
        bottom_section.append(show_at_startup_box)

        start_button = Gtk.Button(label=_("Let's Get Started"))
        start_button.add_css_class("suggested-action")
        start_button.add_css_class("pill")
        start_button.set_size_request(160, 36)
        start_button.set_halign(Gtk.Align.CENTER)
        set_a11y_label(start_button, _("Let's Get Started"))
        bottom_section.append(start_button)

        content_box.append(bottom_section)
        dialog.set_child(content_box)
        dialog.set_follows_content_size(True)

        show_at_startup_switch.connect(
            "notify::active",
            lambda switch, _param: self.set_show(switch.get_active()),
        )
        start_button.connect("clicked", lambda _: dialog.close())
        dialog.present(self._parent)
