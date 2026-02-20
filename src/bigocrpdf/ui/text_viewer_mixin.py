"""Text Viewer Dialog Mixin for DialogsManager."""

import os
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.text_utils import read_text_from_sidecar

# Font size range in points
_MIN_FONT_SIZE = 8
_MAX_FONT_SIZE = 36
_DEFAULT_FONT_SIZE = 12


class TextViewerDialogMixin:
    """Mixin providing text viewer dialog creation and management."""

    def show_extracted_text(self, file_path: str) -> None:
        """Display the extracted text from a PDF file in a resizable window.

        Args:
            file_path: Path to the PDF file
        """
        extracted_text = self._get_extracted_text_for_file(file_path)
        win = self._create_text_viewer_window(file_path, extracted_text)
        win.present()

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file with fallback options.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        if file_path in self.window.settings.extracted_text:
            text = self.window.settings.extracted_text[file_path]
            if text and text.strip():
                return text

        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        sidecar_text = read_text_from_sidecar(sidecar_file)
        if sidecar_text:
            self.window.settings.extracted_text[file_path] = sidecar_text
            return sidecar_text

        temp_text = self._read_from_temp_file(file_path)
        if temp_text:
            self.window.settings.extracted_text[file_path] = temp_text
            return temp_text

        fallback_text = _(
            "OCR processing was completed for this file, but the extracted text could not be found."
        )
        self.window.settings.extracted_text[file_path] = fallback_text
        return fallback_text

    def _read_from_temp_file(self, file_path: str) -> str | None:
        """Try to read text from a temporary file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Text content or None if not found
        """
        temp_dir = os.path.join(os.path.dirname(file_path), ".temp")
        if os.path.exists(temp_dir):
            temp_filename = f"temp_{os.path.basename(os.path.splitext(file_path)[0])}.txt"
            temp_sidecar = os.path.join(temp_dir, temp_filename)
            sidecar_text = read_text_from_sidecar(temp_sidecar)
            if sidecar_text:
                logger.info(f"Found text in temporary file: {temp_sidecar}")
                return sidecar_text
        return None

    def _create_text_viewer_window(self, file_path: str, extracted_text: str) -> Adw.Window:
        """Create a resizable text viewer window.

        Args:
            file_path: Path to the PDF file
            extracted_text: The extracted text content

        Returns:
            Resizable Adw.Window
        """
        win = Adw.Window()
        win.set_title(os.path.basename(file_path))
        win.set_default_size(900, 650)
        win.set_modal(False)

        toolbar_view = Adw.ToolbarView()

        # --- Text View (created first, referenced by search and zoom) ---
        text_view = self._create_styled_text_view(extracted_text)
        state: dict[str, Any] = {"font_size": _DEFAULT_FONT_SIZE}

        # --- Header bar with minimize/maximize/close ---
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        # Detect button side (same logic as main window)
        buttons_left = (
            hasattr(self.window, "_window_buttons_on_left")
            and self.window._window_buttons_on_left()
        )
        if buttons_left:
            header.set_decoration_layout("close,maximize,minimize:")
        else:
            header.set_decoration_layout("menu:minimize,maximize,close")

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.set_valign(Gtk.Align.CENTER)
        title_label = Gtk.Label(label=os.path.basename(file_path))
        title_label.add_css_class("heading")
        title_box.append(title_label)

        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        subtitle_label = Gtk.Label()
        subtitle_label.set_markup(
            f"<small>{word_count:,} "
            + _("words")
            + f" · {char_count:,} "
            + _("characters")
            + "</small>"
        )
        subtitle_label.add_css_class("dim-label")
        title_box.append(subtitle_label)
        header.set_title_widget(title_box)

        toolbar_view.add_top_bar(header)

        # --- Main content ---
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Search bar (always visible)
        search_bar, search_entry = self._create_search_bar(text_view)
        content_box.append(search_bar)

        # Text area — TextView directly in ScrolledWindow for scroll_to_iter
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)
        scrolled.set_margin_start(16)
        scrolled.set_margin_end(16)
        scrolled.set_margin_bottom(8)
        scrolled.add_css_class("card")
        scrolled.set_child(text_view)
        content_box.append(scrolled)

        toolbar_view.set_content(content_box)

        # --- Zoom functions (used by bottom bar and keyboard shortcuts) ---
        zoom_out_btn = Gtk.Button(icon_name="zoom-out-symbolic")
        zoom_in_btn = Gtk.Button(icon_name="zoom-in-symbolic")

        def _apply_font_size(size: int) -> None:
            state["font_size"] = size
            css = f"textview {{ font-family: monospace; font-size: {size}pt; }}"
            provider = Gtk.CssProvider()
            provider.load_from_string(css)
            text_view.get_style_context().add_provider(
                provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            zoom_out_btn.set_sensitive(size > _MIN_FONT_SIZE)
            zoom_in_btn.set_sensitive(size < _MAX_FONT_SIZE)

        def _on_zoom_in(_btn: Gtk.Button) -> None:
            if state["font_size"] < _MAX_FONT_SIZE:
                _apply_font_size(state["font_size"] + 2)

        def _on_zoom_out(_btn: Gtk.Button) -> None:
            if state["font_size"] > _MIN_FONT_SIZE:
                _apply_font_size(state["font_size"] - 2)

        # Apply monospace font immediately at default size
        _apply_font_size(_DEFAULT_FONT_SIZE)

        # --- Bottom action bar ---
        action_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        action_bar.set_margin_start(16)
        action_bar.set_margin_end(16)
        action_bar.set_margin_top(12)
        action_bar.set_margin_bottom(12)

        # Left side — zoom controls
        zoom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        zoom_box.add_css_class("linked")

        zoom_out_btn.set_tooltip_text(_("Decrease font size"))
        set_a11y_label(zoom_out_btn, _("Decrease font size"))
        zoom_out_btn.add_css_class("flat")
        zoom_out_btn.connect("clicked", _on_zoom_out)
        zoom_box.append(zoom_out_btn)

        zoom_in_btn.set_tooltip_text(_("Increase font size"))
        set_a11y_label(zoom_in_btn, _("Increase font size"))
        zoom_in_btn.add_css_class("flat")
        zoom_in_btn.connect("clicked", _on_zoom_in)
        zoom_box.append(zoom_in_btn)

        action_bar.append(zoom_box)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        action_bar.append(spacer)

        # Right side — actions
        copy_btn = Gtk.Button(icon_name="edit-copy-symbolic")
        copy_btn.set_tooltip_text(_("Copy text to clipboard"))
        set_a11y_label(copy_btn, _("Copy text to clipboard"))
        copy_btn.add_css_class("flat")
        copy_btn.connect("clicked", lambda _: self._copy_text_to_clipboard(extracted_text))
        action_bar.append(copy_btn)

        save_txt_btn = Gtk.Button(label=_("Save TXT"))
        save_txt_btn.add_css_class("suggested-action")
        save_txt_btn.connect("clicked", lambda _: self._save_text_to_file(extracted_text))
        set_a11y_label(save_txt_btn, _("Save TXT"))
        action_bar.append(save_txt_btn)

        close_btn = Gtk.Button(label=_("Close"))
        close_btn.connect("clicked", lambda _: win.close())
        set_a11y_label(close_btn, _("Close"))
        action_bar.append(close_btn)

        toolbar_view.add_bottom_bar(action_bar)

        win.set_content(toolbar_view)

        # Keyboard shortcuts
        key_ctrl = Gtk.EventControllerKey()

        def _on_key_pressed(_ctrl, keyval, _keycode, mod):
            ctrl = mod & Gdk.ModifierType.CONTROL_MASK
            if keyval == Gdk.KEY_f and ctrl:
                search_entry.grab_focus()
                return True
            if keyval == Gdk.KEY_plus and ctrl:
                _on_zoom_in(zoom_in_btn)
                return True
            if keyval == Gdk.KEY_minus and ctrl:
                _on_zoom_out(zoom_out_btn)
                return True
            if keyval == Gdk.KEY_Escape:
                win.close()
                return True
            return False

        key_ctrl.connect("key-pressed", _on_key_pressed)
        toolbar_view.add_controller(key_ctrl)

        return win

    def _create_search_bar(self, text_view: Gtk.TextView) -> tuple[Gtk.Box, Gtk.SearchEntry]:
        """Create the search bar (always visible).

        Args:
            text_view: The text view to search in

        Returns:
            Tuple of (search_box, search_entry)
        """
        search_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_box.set_margin_start(16)
        search_box.set_margin_end(16)
        search_box.set_margin_top(12)
        search_box.set_margin_bottom(8)

        search_entry = Gtk.SearchEntry()
        search_entry.set_placeholder_text(_("Search in text..."))
        search_entry.set_hexpand(True)
        set_a11y_label(search_entry, _("Search in text"))
        search_box.append(search_entry)

        nav_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        nav_box.add_css_class("linked")

        prev_btn = Gtk.Button(icon_name="go-up-symbolic")
        prev_btn.set_tooltip_text(_("Previous match"))
        set_a11y_label(prev_btn, _("Previous match"))
        prev_btn.set_sensitive(False)
        prev_btn.add_css_class("flat")
        nav_box.append(prev_btn)

        next_btn = Gtk.Button(icon_name="go-down-symbolic")
        next_btn.set_tooltip_text(_("Next match"))
        set_a11y_label(next_btn, _("Next match"))
        next_btn.set_sensitive(False)
        next_btn.add_css_class("flat")
        nav_box.append(next_btn)

        search_box.append(nav_box)

        match_label = Gtk.Label()
        match_label.add_css_class("dim-label")
        match_label.add_css_class("caption")
        match_label.set_width_chars(12)
        search_box.append(match_label)

        self._setup_search_logic(search_entry, prev_btn, next_btn, match_label, text_view)

        return search_box, search_entry

    def _create_styled_text_view(self, extracted_text: str) -> Gtk.TextView:
        """Create a styled text view.

        Args:
            extracted_text: Text to display

        Returns:
            Styled text view
        """
        text_view = Gtk.TextView()
        text_view.set_editable(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        text_view.set_left_margin(24)
        text_view.set_right_margin(24)
        text_view.set_top_margin(20)
        text_view.set_bottom_margin(20)
        text_view.set_pixels_above_lines(4)
        text_view.set_pixels_below_lines(4)
        text_view.set_pixels_inside_wrap(2)

        buf = text_view.get_buffer()
        buf.set_text(extracted_text)

        tag_table = buf.get_tag_table()

        highlight_tag = Gtk.TextTag.new("search_highlight")
        highlight_tag.set_property("background", "rgba(255, 255, 0, 0.35)")
        tag_table.add(highlight_tag)

        current_tag = Gtk.TextTag.new("current_match")
        current_tag.set_property("background", "rgba(53, 132, 228, 0.5)")
        tag_table.add(current_tag)

        return text_view

    def _setup_search_logic(
        self,
        search_entry: Gtk.SearchEntry,
        prev_btn: Gtk.Button,
        next_btn: Gtk.Button,
        match_label: Gtk.Label,
        text_view: Gtk.TextView,
    ) -> None:
        """Set up search with scroll-to-match and Enter navigation.

        Args:
            search_entry: Search entry widget
            prev_btn: Previous match button
            next_btn: Next match button
            match_label: Match counter label
            text_view: Text view widget
        """
        buf = text_view.get_buffer()
        search_state: dict[str, Any] = {"positions": [], "current": -1}

        def _highlight_matches(search_text: str) -> int:
            start = buf.get_start_iter()
            end = buf.get_end_iter()
            buf.remove_tag_by_name("search_highlight", start, end)
            buf.remove_tag_by_name("current_match", start, end)
            search_state["positions"] = []
            search_state["current"] = -1

            if not search_text:
                return 0

            full_text = buf.get_text(start, end, False).lower()
            needle = search_text.lower()
            pos = 0
            while True:
                pos = full_text.find(needle, pos)
                if pos == -1:
                    break
                search_state["positions"].append(pos)
                m_start = buf.get_iter_at_offset(pos)
                m_end = buf.get_iter_at_offset(pos + len(search_text))
                buf.apply_tag_by_name("search_highlight", m_start, m_end)
                pos += 1
            return len(search_state["positions"])

        def _goto_match(index: int) -> None:
            positions = search_state["positions"]
            if not positions or index < 0 or index >= len(positions):
                return
            text_len = len(search_entry.get_text())

            # Remove previous current highlight
            if search_state["current"] >= 0:
                old = positions[search_state["current"]]
                o_s = buf.get_iter_at_offset(old)
                o_e = buf.get_iter_at_offset(old + text_len)
                buf.remove_tag_by_name("current_match", o_s, o_e)
                buf.apply_tag_by_name("search_highlight", o_s, o_e)

            search_state["current"] = index
            p = positions[index]
            m_s = buf.get_iter_at_offset(p)
            m_e = buf.get_iter_at_offset(p + text_len)
            buf.remove_tag_by_name("search_highlight", m_s, m_e)
            buf.apply_tag_by_name("current_match", m_s, m_e)

            # Place cursor at match to ensure scroll_to_iter works
            buf.place_cursor(m_s)
            text_view.scroll_to_iter(m_s, 0.2, True, 0.0, 0.5)

            match_label.set_text(f"{index + 1}/{len(positions)}")

        def _go_next() -> None:
            if not search_state["positions"]:
                return
            nxt = (search_state["current"] + 1) % len(search_state["positions"])
            _goto_match(nxt)

        def _go_prev() -> None:
            if not search_state["positions"]:
                return
            prv = (search_state["current"] - 1) % len(search_state["positions"])
            _goto_match(prv)

        def _on_search_changed(entry: Gtk.SearchEntry) -> None:
            text = entry.get_text()
            count = _highlight_matches(text)
            has = count > 0
            prev_btn.set_sensitive(has)
            next_btn.set_sensitive(has)
            if not text:
                match_label.set_text("")
            elif count == 0:
                match_label.set_text(_("No matches"))
            else:
                _goto_match(0)

        def _on_activate(_entry: Gtk.SearchEntry) -> None:
            """Enter → go to next match."""
            _go_next()

        search_entry.connect("search-changed", _on_search_changed)
        search_entry.connect("activate", _on_activate)
        prev_btn.connect("clicked", lambda _: _go_prev())
        next_btn.connect("clicked", lambda _: _go_next())

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard.

        Args:
            text: Text to copy
        """
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        logger.info("Text copied to clipboard")
