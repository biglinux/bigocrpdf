"""Text Viewer Dialog Mixin for DialogsManager."""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.text_utils import read_text_from_sidecar


class TextViewerDialogMixin:
    """Mixin providing text viewer dialog creation and management."""

    def show_extracted_text(self, file_path: str) -> None:
        """Display the extracted text from a PDF file in a modern, premium dialog.

        Features a clean design with sidebar statistics, beautiful typography,
        and smooth search functionality.

        Args:
            file_path: Path to the PDF file
        """
        # Get extracted text
        extracted_text = self._get_extracted_text_for_file(file_path)

        # Create modern dialog window
        dialog = self._create_text_viewer_window(file_path, extracted_text)
        dialog.present()

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file with fallback options

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        # Initialize extracted text dictionary if needed
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Check if we have text stored in memory
        if file_path in self.window.settings.extracted_text:
            text = self.window.settings.extracted_text[file_path]
            if text and text.strip():
                return text

        # Try to read from sidecar file
        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        sidecar_text = read_text_from_sidecar(sidecar_file)
        if sidecar_text:
            self.window.settings.extracted_text[file_path] = sidecar_text
            return sidecar_text

        # Try to read from temporary file
        temp_text = self._read_from_temp_file(file_path)
        if temp_text:
            self.window.settings.extracted_text[file_path] = temp_text
            return temp_text

        # Provide generic fallback message
        fallback_text = _(
            "OCR processing was completed for this file, but the extracted text could not be found."
        )
        self.window.settings.extracted_text[file_path] = fallback_text
        return fallback_text

    def _read_from_temp_file(self, file_path: str) -> str | None:
        """Try to read text from a temporary file

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
        """Create a modern, premium text viewer window.

        Args:
            file_path: Path to the PDF file
            extracted_text: The extracted text content

        Returns:
            Modern Adw.Window with premium UI
        """
        # Create main window
        window = Adw.Window()
        window.set_default_size(900, 650)
        window.set_modal(True)
        window.set_transient_for(self.window)

        # Main layout with toolbar view for modern look
        toolbar_view = Adw.ToolbarView()

        # Create header bar
        header = self._create_text_viewer_header(file_path, extracted_text)
        toolbar_view.add_top_bar(header)

        # Create main content
        main_content = self._create_text_viewer_main_content(extracted_text, file_path)
        toolbar_view.set_content(main_content)

        # Create bottom action bar
        bottom_bar = self._create_text_viewer_action_bar(extracted_text, file_path, window)
        toolbar_view.add_bottom_bar(bottom_bar)

        window.set_content(toolbar_view)
        return window

    def _create_text_viewer_header(self, file_path: str, extracted_text: str) -> Adw.HeaderBar:
        """Create the header bar for text viewer.

        Args:
            file_path: Path to the source file
            extracted_text: The extracted text

        Returns:
            Configured header bar
        """
        header = Adw.HeaderBar()
        header.add_css_class("flat")

        # Title widget with file info
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.set_valign(Gtk.Align.CENTER)

        title_label = Gtk.Label(label=os.path.basename(file_path))
        title_label.add_css_class("heading")
        title_box.append(title_label)

        # Statistics subtitle
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

        return header

    def _create_text_viewer_main_content(self, extracted_text: str, file_path: str) -> Gtk.Box:
        """Create the main content area with search and text view.

        Args:
            extracted_text: The text to display
            file_path: Path to the source file

        Returns:
            Main content box
        """
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Search revealer at top
        search_revealer, search_entry, text_view = self._create_modern_search_area(extracted_text)
        content_box.append(search_revealer)

        # Store references for search functionality
        self._text_viewer_search_entry = search_entry
        self._text_viewer_search_revealer = search_revealer

        # Main text area with modern styling
        text_container = self._create_premium_text_area(text_view)
        content_box.append(text_container)

        return content_box

    def _create_modern_search_area(
        self, extracted_text: str
    ) -> tuple[Gtk.Revealer, Gtk.SearchEntry, Gtk.TextView]:
        """Create modern search area with revealer animation.

        Args:
            extracted_text: Text to display and search

        Returns:
            Tuple of (revealer, search_entry, text_view)
        """
        # Create text view first (needed for search setup)
        text_view = self._create_styled_text_view(extracted_text)

        # Search revealer for smooth show/hide
        search_revealer = Gtk.Revealer()
        search_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        search_revealer.set_transition_duration(200)
        search_revealer.set_reveal_child(True)

        # Search container with card styling
        search_card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_card.set_margin_start(16)
        search_card.set_margin_end(16)
        search_card.set_margin_top(12)
        search_card.set_margin_bottom(8)

        # Search entry with modern styling
        search_entry = Gtk.SearchEntry()
        search_entry.set_placeholder_text(_("Search in text..."))
        search_entry.set_hexpand(True)
        search_entry.add_css_class("search-entry")
        search_card.append(search_entry)

        # Navigation buttons
        nav_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        nav_box.add_css_class("linked")

        prev_btn = Gtk.Button()
        prev_btn.set_icon_name("go-up-symbolic")
        prev_btn.set_tooltip_text(_("Go to previous match (Ctrl+Shift+G)"))
        prev_btn.set_sensitive(False)
        prev_btn.add_css_class("flat")
        nav_box.append(prev_btn)

        next_btn = Gtk.Button()
        next_btn.set_icon_name("go-down-symbolic")
        next_btn.set_tooltip_text(_("Go to next match (Ctrl+G)"))
        next_btn.set_sensitive(False)
        next_btn.add_css_class("flat")
        nav_box.append(next_btn)

        search_card.append(nav_box)

        # Match counter label
        match_label = Gtk.Label()
        match_label.add_css_class("dim-label")
        match_label.add_css_class("caption")
        match_label.set_width_chars(12)
        search_card.append(match_label)

        search_revealer.set_child(search_card)

        # Set up search functionality
        self._setup_modern_search(search_entry, prev_btn, next_btn, match_label, text_view)

        return search_revealer, search_entry, text_view

    def _create_styled_text_view(self, extracted_text: str) -> Gtk.TextView:
        """Create a beautifully styled text view.

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

        # Better typography - use system font with good line height
        text_view.set_pixels_above_lines(4)
        text_view.set_pixels_below_lines(4)
        text_view.set_pixels_inside_wrap(2)

        # Set text content
        buffer = text_view.get_buffer()
        buffer.set_text(extracted_text)

        # Add search highlight tag
        tag_table = buffer.get_tag_table()
        highlight_tag = Gtk.TextTag.new("search_highlight")
        highlight_tag.set_property("background", "#fce94f")
        highlight_tag.set_property("foreground", "#2e3436")
        tag_table.add(highlight_tag)

        # Add current match tag (brighter highlight)
        current_tag = Gtk.TextTag.new("current_match")
        current_tag.set_property("background", "#f57900")
        current_tag.set_property("foreground", "#ffffff")
        tag_table.add(current_tag)

        return text_view

    def _create_premium_text_area(self, text_view: Gtk.TextView) -> Gtk.ScrolledWindow:
        """Create the premium text area container.

        Args:
            text_view: The text view widget

        Returns:
            Scrolled window with text view
        """
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)

        # Add a subtle background card effect
        text_frame = Gtk.Frame()
        text_frame.set_margin_start(16)
        text_frame.set_margin_end(16)
        text_frame.set_margin_bottom(8)
        text_frame.add_css_class("view")
        text_frame.set_child(text_view)

        scrolled.set_child(text_frame)
        return scrolled

    def _create_text_viewer_action_bar(
        self, extracted_text: str, file_path: str, window: Adw.Window
    ) -> Gtk.Box:
        """Create the bottom action bar with buttons.

        Args:
            extracted_text: The text content
            file_path: Source file path
            window: Parent window

        Returns:
            Action bar box
        """
        action_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        action_bar.set_margin_start(16)
        action_bar.set_margin_end(16)
        action_bar.set_margin_top(12)
        action_bar.set_margin_bottom(12)

        # Left side - search toggle
        search_btn = Gtk.ToggleButton()
        search_btn.set_icon_name("system-search-symbolic")
        search_btn.set_tooltip_text(_("Search in text (Ctrl+F)"))
        search_btn.set_active(True)
        search_btn.add_css_class("flat")
        search_btn.connect("toggled", self._on_search_toggle)
        action_bar.append(search_btn)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        action_bar.append(spacer)

        # Right side - action buttons
        # Copy button
        copy_btn = Gtk.Button()
        copy_btn.set_icon_name("edit-copy-symbolic")
        copy_btn.set_tooltip_text(_("Copy text to clipboard"))
        copy_btn.add_css_class("flat")
        copy_btn.connect("clicked", lambda _: self._copy_text_to_clipboard(extracted_text))
        action_bar.append(copy_btn)

        # Save TXT button (prominent)
        save_txt_btn = Gtk.Button()
        save_txt_btn.set_label(_("Save TXT"))
        save_txt_btn.add_css_class("suggested-action")
        save_txt_btn.connect("clicked", lambda _: self._save_text_to_file(extracted_text))
        action_bar.append(save_txt_btn)

        # Close button
        close_btn = Gtk.Button()
        close_btn.set_label(_("Close"))
        close_btn.connect("clicked", lambda _: window.close())
        action_bar.append(close_btn)

        # Setup keyboard shortcut for search toggle
        key_controller = Gtk.EventControllerKey()

        def on_key_pressed(_controller, keyval, _keycode, state):
            if keyval == Gdk.KEY_f and state & Gdk.ModifierType.CONTROL_MASK:
                search_btn.set_active(not search_btn.get_active())
                if search_btn.get_active() and hasattr(self, "_text_viewer_search_entry"):
                    self._text_viewer_search_entry.grab_focus()
                return True
            if keyval == Gdk.KEY_Escape:
                window.close()
                return True
            return False

        key_controller.connect("key-pressed", on_key_pressed)
        window.add_controller(key_controller)

        return action_bar

    def _on_search_toggle(self, button: Gtk.ToggleButton) -> None:
        """Handle search toggle button.

        Args:
            button: The toggle button
        """
        if hasattr(self, "_text_viewer_search_revealer"):
            self._text_viewer_search_revealer.set_reveal_child(button.get_active())
            if button.get_active() and hasattr(self, "_text_viewer_search_entry"):
                self._text_viewer_search_entry.grab_focus()

    def _setup_modern_search(
        self,
        search_entry: Gtk.SearchEntry,
        prev_btn: Gtk.Button,
        next_btn: Gtk.Button,
        match_label: Gtk.Label,
        text_view: Gtk.TextView,
    ) -> None:
        """Set up modern search functionality with smooth navigation.

        Args:
            search_entry: Search entry widget
            prev_btn: Previous match button
            next_btn: Next match button
            match_label: Match counter label
            text_view: Text view widget
        """
        buffer = text_view.get_buffer()
        search_state = {"positions": [], "current": -1}

        def highlight_matches(search_text: str) -> int:
            """Highlight all matches and return count."""
            # Clear previous highlights
            start = buffer.get_start_iter()
            end = buffer.get_end_iter()
            buffer.remove_tag_by_name("search_highlight", start, end)
            buffer.remove_tag_by_name("current_match", start, end)

            search_state["positions"] = []
            search_state["current"] = -1

            if not search_text:
                return 0

            # Find all matches (case-insensitive)
            full_text = buffer.get_text(start, end, False).lower()
            search_lower = search_text.lower()
            pos = 0

            while True:
                pos = full_text.find(search_lower, pos)
                if pos == -1:
                    break
                search_state["positions"].append(pos)

                # Apply highlight
                match_start = buffer.get_iter_at_offset(pos)
                match_end = buffer.get_iter_at_offset(pos + len(search_text))
                buffer.apply_tag_by_name("search_highlight", match_start, match_end)
                pos += 1

            return len(search_state["positions"])

        def goto_match(index: int) -> None:
            """Navigate to a specific match."""
            positions = search_state["positions"]
            if not positions or index < 0 or index >= len(positions):
                return

            # Remove current match highlight from previous
            if search_state["current"] >= 0:
                old_pos = positions[search_state["current"]]
                old_start = buffer.get_iter_at_offset(old_pos)
                old_end = buffer.get_iter_at_offset(old_pos + len(search_entry.get_text()))
                buffer.remove_tag_by_name("current_match", old_start, old_end)
                buffer.apply_tag_by_name("search_highlight", old_start, old_end)

            search_state["current"] = index
            pos = positions[index]

            # Apply current match highlight
            match_start = buffer.get_iter_at_offset(pos)
            match_end = buffer.get_iter_at_offset(pos + len(search_entry.get_text()))
            buffer.remove_tag_by_name("search_highlight", match_start, match_end)
            buffer.apply_tag_by_name("current_match", match_start, match_end)

            # Scroll to match
            text_view.scroll_to_iter(match_start, 0.2, True, 0.0, 0.5)

            # Update label
            match_label.set_text(f"{index + 1}/{len(positions)}")

        def on_search_changed(entry: Gtk.SearchEntry) -> None:
            """Handle search text changes."""
            search_text = entry.get_text()
            count = highlight_matches(search_text)

            # Update UI
            has_matches = count > 0
            prev_btn.set_sensitive(has_matches)
            next_btn.set_sensitive(has_matches)

            if not search_text:
                match_label.set_text("")
            elif count == 0:
                match_label.set_text(_("No matches"))
            else:
                goto_match(0)

        def on_prev(_btn: Gtk.Button) -> None:
            """Go to previous match."""
            if search_state["current"] > 0:
                goto_match(search_state["current"] - 1)
            elif search_state["positions"]:
                goto_match(len(search_state["positions"]) - 1)  # Wrap around

        def on_next(_btn: Gtk.Button) -> None:
            """Go to next match."""
            if search_state["current"] < len(search_state["positions"]) - 1:
                goto_match(search_state["current"] + 1)
            elif search_state["positions"]:
                goto_match(0)  # Wrap around

        # Connect signals
        search_entry.connect("search-changed", on_search_changed)
        prev_btn.connect("clicked", on_prev)
        next_btn.connect("clicked", on_next)

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard

        Args:
            text: Text to copy
        """
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        logger.info("Text copied to clipboard")
