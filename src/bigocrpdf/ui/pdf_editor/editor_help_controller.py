"""Help dialog controller for the PDF editor window."""

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.durable_writes import read_regular_file_bytes, write_text_atomically
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class EditorHelpController:
    """Own the editor help dialog and its startup preference."""

    _EDITOR_HELP_CONFIG = Path.home() / ".config/bigocrpdf/show_editor_help"

    def __init__(self, owner: Gtk.Widget, config_path: str | Path | None = None) -> None:
        self._owner = owner
        self._config_path = Path(config_path or self._EDITOR_HELP_CONFIG)

    def should_show(self) -> bool:
        try:
            content = read_regular_file_bytes(self._config_path).decode("utf-8").strip()
        except FileNotFoundError:
            return True
        except (OSError, UnicodeError) as exc:
            logger.error(f"Error reading editor help preference: {exc}")
            return True
        return content.lower() != "false"

    def set_show(self, show: bool) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomically(self._config_path, "true" if show else "false")
        except OSError as exc:
            logger.error(f"Error saving editor help preference: {exc}")

    def show(self, *_args) -> None:
        dialog = Adw.Dialog()
        dialog.set_title(_("PDF Editor Help"))
        dialog.set_content_width(520)
        dialog.set_content_height(480)
        content = self._build_editor_help_content(dialog)
        dialog.set_child(content)
        dialog.set_follows_content_size(True)
        dialog.present(self._owner)

    def _build_editor_help_content(self, dialog: Adw.Dialog) -> Gtk.Box:
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(36)
        content.set_margin_end(36)
        self._append_help_header(content)
        self._append_help_shortcuts(content)
        self._append_help_tips(content)
        self._append_help_footer(content, dialog)
        return content

    @staticmethod
    def _append_help_header(content: Gtk.Box) -> None:
        from bigocrpdf.config import APP_ICON_NAME

        icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        icon.set_pixel_size(48)
        icon.set_margin_bottom(12)
        icon.set_halign(Gtk.Align.CENTER)
        icon.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
        content.append(icon)
        title = Gtk.Label(label=_("PDF Editor"))
        title.add_css_class("title-2")
        title.set_halign(Gtk.Align.CENTER)
        title.set_margin_bottom(14)
        content.append(title)
        desc = Gtk.Label()
        desc.set_text(
            _(
                "Use the PDF editor to view and organize your documents. You can rearrange, rotate, or remove pages, compress files, and split documents by pages or file size."
            )
        )
        desc.set_wrap(True)
        desc.set_justify(Gtk.Justification.LEFT)
        desc.set_halign(Gtk.Align.START)
        desc.set_margin_bottom(16)
        desc.set_max_width_chars(55)
        content.append(desc)

    @staticmethod
    def _append_help_shortcuts(content: Gtk.Box) -> None:
        shortcuts_label = Gtk.Label(label=_("Keyboard shortcuts:"))
        shortcuts_label.add_css_class("heading")
        shortcuts_label.set_halign(Gtk.Align.START)
        shortcuts_label.set_margin_bottom(8)
        content.append(shortcuts_label)
        shortcuts = [
            ("Ctrl+Z", _("Undo last action")),
            ("Ctrl+A", _("Select all pages")),
            ("Ctrl+S", _("Save and close")),
            ("Delete", _("Remove selected pages")),
            ("Ctrl+L / Ctrl+R", _("Rotate left / right")),
            ("+  /  -", _("Zoom in / out")),
        ]
        shortcuts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        shortcuts_box.set_margin_bottom(16)
        for key, action in shortcuts:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            key_label = Gtk.Label(label=key)
            key_label.add_css_class("monospace")
            key_label.set_xalign(0)
            key_label.set_size_request(140, -1)
            row.append(key_label)
            action_label = Gtk.Label(label=action)
            action_label.set_xalign(0)
            row.append(action_label)
            shortcuts_box.append(row)
        content.append(shortcuts_box)

    @staticmethod
    def _append_help_tips(content: Gtk.Box) -> None:
        tips_label = Gtk.Label(label=_("Tips:"))
        tips_label.add_css_class("heading")
        tips_label.set_halign(Gtk.Align.START)
        tips_label.set_margin_bottom(8)
        content.append(tips_label)
        tips = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        tip_items = [
            _("Drag and drop pages to reorder them"),
            _("Right-click a page to save it as image or PDF"),
            _("Drag external files onto the editor to add them"),
            _("Use the menu to compress or split documents"),
        ]
        for tip in tip_items:
            label = Gtk.Label(label=f"• {tip}")
            label.set_wrap(True)
            label.set_halign(Gtk.Align.START)
            label.set_xalign(0)
            tips.append(label)
        content.append(tips)

    def _append_help_footer(self, content: Gtk.Box, dialog: Adw.Dialog) -> None:
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(12)
        separator.set_margin_bottom(16)
        content.append(separator)
        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        bottom.append(self._create_startup_help_row())
        close_btn = Gtk.Button(label=_("Got it"))
        close_btn.add_css_class("suggested-action")
        close_btn.add_css_class("pill")
        close_btn.set_size_request(140, 36)
        close_btn.set_halign(Gtk.Align.CENTER)
        set_a11y_label(close_btn, _("Got it"))
        close_btn.connect("clicked", lambda _: dialog.close())
        bottom.append(close_btn)
        content.append(bottom)

    def _create_startup_help_row(self) -> Gtk.Box:
        startup_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        startup_row.set_halign(Gtk.Align.FILL)
        startup_label = Gtk.Label(label=_("Show when opening the editor"))
        startup_label.set_halign(Gtk.Align.START)
        startup_label.set_hexpand(True)
        startup_switch = Gtk.Switch()
        startup_switch.set_active(self.should_show())
        startup_switch.set_valign(Gtk.Align.CENTER)
        startup_switch.set_halign(Gtk.Align.END)
        set_a11y_label(startup_switch, _("Show when opening the editor"))
        startup_switch.connect("notify::active", lambda sw, _p: self.set_show(sw.get_active()))
        startup_row.append(startup_label)
        startup_row.append(startup_switch)
        return startup_row
