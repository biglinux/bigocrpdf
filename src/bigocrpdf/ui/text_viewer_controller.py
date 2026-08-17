"""Extracted-text viewer controller for DialogsManager."""

import csv
import os
import subprocess
from dataclasses import dataclass, field

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, GLib, Gtk

from bigocrpdf.ui.widgets import get_default_clipboard
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.text_utils import read_text_from_sidecar

# Font size range in points
_MIN_FONT_SIZE = 8
_MAX_FONT_SIZE = 36
_DEFAULT_FONT_SIZE = 12


@dataclass
class TextViewerState:
    """State owned by one extracted-text window."""

    font_size: int = _DEFAULT_FONT_SIZE
    matches: list[tuple[int, int]] = field(default_factory=list)
    current_match: int = -1
    debounce_id: int = 0
    font_provider: Gtk.CssProvider = field(default_factory=Gtk.CssProvider)


class TextViewerController:
    """Own extracted-text lookup, presentation, search, and zoom behavior."""

    def __init__(self, parent: Gtk.Window, settings, show_toast, file_save) -> None:
        self._parent = parent
        self._settings = settings
        self._show_toast = show_toast
        self._file_save = file_save

    def show_extracted_text(self, file_path: str) -> None:
        """Display the extracted text from a PDF file in a resizable window.

        Args:
            file_path: Path to the PDF file
        """
        extracted_text = self._get_extracted_text_for_file(file_path)
        win = self._create_text_viewer_window(file_path, extracted_text)
        win.present()

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Return cached, sidecar, or structured text for a file."""
        cached_text = self._settings.extracted_text.get(file_path)
        if cached_text and cached_text.strip():
            return cached_text

        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        sidecar_text = read_text_from_sidecar(sidecar_file)
        if sidecar_text and sidecar_text.strip():
            self._settings.extracted_text[file_path] = sidecar_text
            return sidecar_text

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            try:
                from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_text

                structured = convert_pdf_to_text(file_path)
            except (csv.Error, OSError, subprocess.SubprocessError, ValueError) as error:
                logger.debug("Structured text extraction failed: %s", error)
            else:
                if structured.strip():
                    self._settings.extracted_text[file_path] = structured
                    return structured

        return _(
            "OCR processing was completed for this file, but the extracted text could not be found."
        )

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
        win.set_transient_for(self._parent)
        win.set_destroy_with_parent(True)

        toolbar_view = Adw.ToolbarView()
        text_view = self._create_styled_text_view(extracted_text)
        state = TextViewerState()
        text_view.get_style_context().add_provider(
            state.font_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
        search_bar, search_entry = self._create_search_bar(text_view, state)
        zoom_out_btn = Gtk.Button(icon_name="zoom-out-symbolic")
        zoom_in_btn = Gtk.Button(icon_name="zoom-in-symbolic")
        self._apply_text_viewer_font_size(
            zoom_out_btn,
            zoom_in_btn,
            state,
            _DEFAULT_FONT_SIZE,
        )

        toolbar_view.add_top_bar(self._text_viewer_header(file_path, extracted_text))
        toolbar_view.set_content(self._text_viewer_content_box(text_view, search_bar))
        toolbar_view.add_bottom_bar(
            self._text_viewer_action_bar(
                win,
                text_view,
                extracted_text,
                state,
                zoom_out_btn,
                zoom_in_btn,
            )
        )
        win.set_content(toolbar_view)
        self._connect_text_viewer_shortcuts(
            toolbar_view,
            win,
            search_entry,
            state,
            zoom_out_btn,
            zoom_in_btn,
        )
        win.connect("close-request", lambda *_: self._clear_pending_search(state))
        return win

    def _text_viewer_header(self, file_path: str, extracted_text: str) -> Adw.HeaderBar:
        header = Adw.HeaderBar()
        header.add_css_class("flat")

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.set_valign(Gtk.Align.CENTER)
        title_label = Gtk.Label(label=os.path.basename(file_path))
        title_label.add_css_class("heading")
        title_box.append(title_label)

        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        words = ngettext("{count} word", "{count} words", word_count).format(
            count=f"{word_count:,}"
        )
        characters = ngettext("{count} character", "{count} characters", char_count).format(
            count=f"{char_count:,}"
        )
        subtitle_label = Gtk.Label(
            label=_("{words} · {characters}").format(words=words, characters=characters)
        )
        subtitle_label.add_css_class("caption")
        subtitle_label.add_css_class("dim-label")
        title_box.append(subtitle_label)
        header.set_title_widget(title_box)
        return header

    @staticmethod
    def _text_viewer_content_box(text_view: Gtk.TextView, search_bar: Gtk.Box) -> Gtk.Box:
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content_box.append(search_bar)

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
        return content_box

    def _text_viewer_action_bar(
        self,
        win: Adw.Window,
        text_view: Gtk.TextView,
        extracted_text: str,
        state: TextViewerState,
        zoom_out_btn: Gtk.Button,
        zoom_in_btn: Gtk.Button,
    ) -> Gtk.Box:
        action_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        action_bar.set_margin_start(16)
        action_bar.set_margin_end(16)
        action_bar.set_margin_top(12)
        action_bar.set_margin_bottom(12)

        action_bar.append(self._zoom_controls_box(state, zoom_out_btn, zoom_in_btn))

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        action_bar.append(spacer)

        copy_btn = Gtk.Button(icon_name="edit-copy-symbolic")
        copy_btn.set_tooltip_text(_("Copy text to clipboard"))
        set_a11y_label(copy_btn, _("Copy text to clipboard"))
        copy_btn.add_css_class("flat")
        copy_btn.connect("clicked", lambda _: self._copy_text_to_clipboard(extracted_text))
        action_bar.append(copy_btn)

        save_txt_btn = Gtk.Button(label=_("Save TXT"))
        save_txt_btn.add_css_class("suggested-action")
        save_txt_btn.connect("clicked", lambda _: self._file_save.save_text(extracted_text))
        set_a11y_label(save_txt_btn, _("Save TXT"))
        action_bar.append(save_txt_btn)

        close_btn = Gtk.Button(label=_("Close"))
        close_btn.connect("clicked", lambda _: win.close())
        set_a11y_label(close_btn, _("Close"))
        action_bar.append(close_btn)
        return action_bar

    def _zoom_controls_box(
        self,
        state: TextViewerState,
        zoom_out_btn: Gtk.Button,
        zoom_in_btn: Gtk.Button,
    ) -> Gtk.Box:
        zoom_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        zoom_box.add_css_class("linked")

        self._configure_zoom_button(
            zoom_out_btn,
            _("Decrease font size"),
            lambda _: self._zoom_text_viewer(zoom_out_btn, zoom_in_btn, state, -2),
        )
        zoom_box.append(zoom_out_btn)

        self._configure_zoom_button(
            zoom_in_btn,
            _("Increase font size"),
            lambda _: self._zoom_text_viewer(zoom_out_btn, zoom_in_btn, state, 2),
        )
        zoom_box.append(zoom_in_btn)
        return zoom_box

    @staticmethod
    def _configure_zoom_button(button: Gtk.Button, label: str, callback) -> None:
        button.set_tooltip_text(label)
        set_a11y_label(button, label)
        button.add_css_class("flat")
        button.connect("clicked", callback)

    @staticmethod
    def _apply_text_viewer_font_size(
        zoom_out_btn: Gtk.Button,
        zoom_in_btn: Gtk.Button,
        state: TextViewerState,
        size: int,
    ) -> None:
        state.font_size = size
        state.font_provider.load_from_string(
            f"textview {{ font-family: monospace; font-size: {size}pt; }}"
        )
        zoom_out_btn.set_sensitive(size > _MIN_FONT_SIZE)
        zoom_in_btn.set_sensitive(size < _MAX_FONT_SIZE)

    def _zoom_text_viewer(
        self,
        zoom_out_btn: Gtk.Button,
        zoom_in_btn: Gtk.Button,
        state: TextViewerState,
        delta: int,
    ) -> None:
        new_size = state.font_size + delta
        if _MIN_FONT_SIZE <= new_size <= _MAX_FONT_SIZE:
            self._apply_text_viewer_font_size(zoom_out_btn, zoom_in_btn, state, new_size)

    def _connect_text_viewer_shortcuts(
        self,
        toolbar_view: Adw.ToolbarView,
        win: Adw.Window,
        search_entry: Gtk.SearchEntry,
        state: TextViewerState,
        zoom_out_btn: Gtk.Button,
        zoom_in_btn: Gtk.Button,
    ) -> None:
        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect(
            "key-pressed",
            lambda _ctrl, keyval, _keycode, mod: self._on_text_viewer_key_pressed(
                win,
                search_entry,
                state,
                zoom_out_btn,
                zoom_in_btn,
                keyval,
                mod,
            ),
        )
        toolbar_view.add_controller(key_ctrl)

    def _on_text_viewer_key_pressed(
        self,
        win: Adw.Window,
        search_entry: Gtk.SearchEntry,
        state: TextViewerState,
        zoom_out_btn: Gtk.Button,
        zoom_in_btn: Gtk.Button,
        keyval,
        mod,
    ) -> bool:
        ctrl = mod & Gdk.ModifierType.CONTROL_MASK
        if keyval == Gdk.KEY_f and ctrl:
            search_entry.grab_focus()
            return True
        if keyval == Gdk.KEY_plus and ctrl:
            self._zoom_text_viewer(zoom_out_btn, zoom_in_btn, state, 2)
            return True
        if keyval == Gdk.KEY_minus and ctrl:
            self._zoom_text_viewer(zoom_out_btn, zoom_in_btn, state, -2)
            return True
        if keyval == Gdk.KEY_Escape:
            win.close()
            return True
        return False

    def _create_search_bar(
        self,
        text_view: Gtk.TextView,
        state: TextViewerState,
    ) -> tuple[Gtk.Box, Gtk.SearchEntry]:
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

        self._setup_search_logic(
            search_entry,
            prev_btn,
            next_btn,
            match_label,
            text_view,
            state,
        )

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
        state: TextViewerState,
    ) -> None:
        """Set up search with scroll-to-match and Enter navigation.

        Args:
            search_entry: Search entry widget
            prev_btn: Previous match button
            next_btn: Next match button
            match_label: Match counter label
            text_view: Text view widget
        """

        def _on_search_changed(entry: Gtk.SearchEntry) -> None:
            if state.debounce_id:
                GLib.source_remove(state.debounce_id)
                state.debounce_id = 0

            def _do_search() -> bool:
                return self._run_text_search(
                    entry,
                    prev_btn,
                    next_btn,
                    match_label,
                    text_view,
                    state,
                )

            state.debounce_id = GLib.timeout_add(150, _do_search)

        search_entry.connect("search-changed", _on_search_changed)
        search_entry.connect(
            "activate",
            lambda _entry: self._go_to_next_text_match(match_label, text_view, state),
        )
        prev_btn.connect(
            "clicked",
            lambda _: self._go_to_previous_text_match(match_label, text_view, state),
        )
        next_btn.connect(
            "clicked",
            lambda _: self._go_to_next_text_match(match_label, text_view, state),
        )

    def _run_text_search(
        self,
        entry: Gtk.SearchEntry,
        prev_btn: Gtk.Button,
        next_btn: Gtk.Button,
        match_label: Gtk.Label,
        text_view: Gtk.TextView,
        state: TextViewerState,
    ) -> bool:
        state.debounce_id = 0
        text = entry.get_text()
        count = self._highlight_text_matches(text_view, state, text)
        has_matches = count > 0
        prev_btn.set_sensitive(has_matches)
        next_btn.set_sensitive(has_matches)

        if not text:
            match_label.set_text("")
        elif count == 0:
            match_label.set_text(_("No matches"))
        else:
            self._go_to_text_match(0, match_label, text_view, state)
        return GLib.SOURCE_REMOVE

    @staticmethod
    def _highlight_text_matches(
        text_view: Gtk.TextView,
        state: TextViewerState,
        search_text: str,
    ) -> int:
        buf = text_view.get_buffer()
        start = buf.get_start_iter()
        end = buf.get_end_iter()
        buf.remove_tag_by_name("search_highlight", start, end)
        buf.remove_tag_by_name("current_match", start, end)
        state.matches = []
        state.current_match = -1

        if not search_text:
            return 0

        cursor = start
        flags = Gtk.TextSearchFlags.CASE_INSENSITIVE | Gtk.TextSearchFlags.TEXT_ONLY
        while match := cursor.forward_search(search_text, flags, end):
            match_start, match_end = match
            state.matches.append((match_start.get_offset(), match_end.get_offset()))
            buf.apply_tag_by_name("search_highlight", match_start, match_end)
            cursor = match_end
        return len(state.matches)

    def _go_to_next_text_match(
        self,
        match_label: Gtk.Label,
        text_view: Gtk.TextView,
        state: TextViewerState,
    ) -> None:
        if not state.matches:
            return
        next_index = (state.current_match + 1) % len(state.matches)
        if next_index == 0 and state.current_match == len(state.matches) - 1:
            self._show_toast(_("Wrapped to first match"))
        self._go_to_text_match(next_index, match_label, text_view, state)

    def _go_to_previous_text_match(
        self,
        match_label: Gtk.Label,
        text_view: Gtk.TextView,
        state: TextViewerState,
    ) -> None:
        if not state.matches:
            return
        previous_index = (state.current_match - 1) % len(state.matches)
        if previous_index == len(state.matches) - 1 and state.current_match == 0:
            self._show_toast(_("Wrapped to last match"))
        self._go_to_text_match(previous_index, match_label, text_view, state)

    @staticmethod
    def _go_to_text_match(
        index: int,
        match_label: Gtk.Label,
        text_view: Gtk.TextView,
        state: TextViewerState,
    ) -> None:
        if not 0 <= index < len(state.matches):
            return

        buf = text_view.get_buffer()
        if state.current_match >= 0:
            old_start_offset, old_end_offset = state.matches[state.current_match]
            old_start = buf.get_iter_at_offset(old_start_offset)
            old_end = buf.get_iter_at_offset(old_end_offset)
            buf.remove_tag_by_name("current_match", old_start, old_end)
            buf.apply_tag_by_name("search_highlight", old_start, old_end)

        state.current_match = index
        start_offset, end_offset = state.matches[index]
        match_start = buf.get_iter_at_offset(start_offset)
        match_end = buf.get_iter_at_offset(end_offset)
        buf.remove_tag_by_name("search_highlight", match_start, match_end)
        buf.apply_tag_by_name("current_match", match_start, match_end)
        buf.place_cursor(match_start)
        text_view.scroll_to_iter(match_start, 0.2, True, 0.0, 0.5)
        match_label.set_text(
            _("{current}/{total}").format(current=index + 1, total=len(state.matches))
        )

    @staticmethod
    def _clear_pending_search(state: TextViewerState) -> bool:
        if state.debounce_id:
            GLib.source_remove(state.debounce_id)
            state.debounce_id = 0
        return False

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard.

        Args:
            text: Text to copy
        """
        clipboard = get_default_clipboard()
        if clipboard is None:
            logger.warning("Clipboard is unavailable because no display is active")
            return
        # pygobject-stubs 2.17.0 omits this GTK 4 method.
        clipboard.set_text(text)  # pyright: ignore[reportAttributeAccessIssue]
        logger.info("Text copied to clipboard")
