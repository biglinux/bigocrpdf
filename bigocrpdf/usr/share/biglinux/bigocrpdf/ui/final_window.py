"""
BigOcrPdf - Final Window (Conclusion Page)

This module contains the conclusion page implementation.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, Gdk

import os
import time
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.logger import logger
from utils.i18n import _


class FinalWindow:
    """Final/conclusion page for BigOcrPdf application"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize final window

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # UI components references
        self.result_file_count = None
        self.result_page_count = None
        self.result_time = None
        self.result_file_size = None
        self.output_list_box = None

    def create_conclusion_page(self) -> Gtk.Box:
        """Create the conclusion page showing OCR processing results"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        # Set individual margins
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        # Add summary card
        summary_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        summary_card.add_css_class("card")
        # Card header
        card_header = Gtk.Label(label=_("Processing Summary"))
        card_header.add_css_class("heading")
        card_header.set_halign(Gtk.Align.START)
        card_header.set_margin_top(16)
        card_header.set_margin_start(16)
        summary_card.append(card_header)

        # Create a horizontal box to contain two columns
        columns_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=32)
        columns_box.set_margin_start(16)
        columns_box.set_margin_end(16)
        columns_box.set_margin_bottom(16)
        columns_box.set_margin_top(8)

        # Left column - Files and Pages
        left_grid = Gtk.Grid()
        left_grid.set_column_spacing(16)
        left_grid.set_row_spacing(16)
        left_grid.set_hexpand(True)
        left_grid.set_halign(Gtk.Align.START)

        # Files processed
        files_icon = Gtk.Image.new_from_icon_name("document-multiple-symbolic")
        files_icon.set_pixel_size(16)
        left_grid.attach(files_icon, 0, 0, 1, 1)

        files_label = Gtk.Label(label=_("Files processed:"))
        files_label.set_halign(Gtk.Align.START)
        files_label.set_margin_start(8)
        left_grid.attach(files_label, 1, 0, 1, 1)

        self.result_file_count = Gtk.Label(label="0")
        self.result_file_count.set_halign(Gtk.Align.END)
        self.result_file_count.add_css_class("heading")
        self.result_file_count.set_hexpand(True)
        left_grid.attach(self.result_file_count, 2, 0, 1, 1)

        # Total pages
        pages_icon = Gtk.Image.new_from_icon_name("view-paged-symbolic")
        pages_icon.set_pixel_size(16)
        left_grid.attach(pages_icon, 0, 1, 1, 1)

        pages_label = Gtk.Label(label=_("Total pages:"))
        pages_label.set_halign(Gtk.Align.START)
        pages_label.set_margin_start(8)
        left_grid.attach(pages_label, 1, 1, 1, 1)

        self.result_page_count = Gtk.Label(label="0")
        self.result_page_count.set_halign(Gtk.Align.END)
        self.result_page_count.add_css_class("heading")
        left_grid.attach(self.result_page_count, 2, 1, 1, 1)

        # Right column - Time and Size
        right_grid = Gtk.Grid()
        right_grid.set_column_spacing(16)
        right_grid.set_row_spacing(16)
        right_grid.set_hexpand(True)
        right_grid.set_halign(Gtk.Align.START)

        # Processing time
        time_icon = Gtk.Image.new_from_icon_name("clock-symbolic")
        time_icon.set_pixel_size(16)
        right_grid.attach(time_icon, 0, 0, 1, 1)

        time_label = Gtk.Label(label=_("Processing time:"))
        time_label.set_halign(Gtk.Align.START)
        time_label.set_margin_start(8)
        right_grid.attach(time_label, 1, 0, 1, 1)

        self.result_time = Gtk.Label(label="00:00")
        self.result_time.set_halign(Gtk.Align.END)
        self.result_time.add_css_class("heading")
        self.result_time.set_hexpand(True)
        right_grid.attach(self.result_time, 2, 0, 1, 1)

        # Output file size
        size_icon = Gtk.Image.new_from_icon_name("drive-harddisk-symbolic")
        size_icon.set_pixel_size(16)
        right_grid.attach(size_icon, 0, 1, 1, 1)

        size_label = Gtk.Label(label=_("File size:"))
        size_label.set_halign(Gtk.Align.START)
        size_label.set_margin_start(8)
        right_grid.attach(size_label, 1, 1, 1, 1)

        self.result_file_size = Gtk.Label(label="0 KB")
        self.result_file_size.set_halign(Gtk.Align.END)
        self.result_file_size.add_css_class("heading")
        right_grid.attach(self.result_file_size, 2, 1, 1, 1)

        # Add the two grids to the columns box
        columns_box.append(left_grid)
        columns_box.append(right_grid)

        # Add the columns box to the summary card
        summary_card.append(columns_box)
        main_box.append(summary_card)

        # Output files section
        files_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        files_card.add_css_class("card")
        files_card.set_margin_top(16)
        files_card.set_margin_bottom(16)

        # Card header
        files_header = Gtk.Label(label=_("Generated Files"))
        files_header.add_css_class("heading")
        files_header.set_halign(Gtk.Align.START)
        files_header.set_margin_top(16)
        files_header.set_margin_start(16)
        files_card.append(files_header)

        # Create scrolled window for output files
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(100)
        scrolled.set_max_content_height(200)
        scrolled.set_margin_start(16)
        scrolled.set_margin_end(16)
        scrolled.set_margin_bottom(16)

        # Create list box for output files
        self.output_list_box = Gtk.ListBox()
        self.output_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.output_list_box.add_css_class("boxed-list")
        self.output_list_box.add_css_class(
            "output-files-list"
        )  # Add custom class for styling hover effect
        self.output_list_box.set_vexpand(True)

        scrolled.set_child(self.output_list_box)
        files_card.append(scrolled)

        main_box.append(files_card)

        return main_box

    def update_conclusion_page(self) -> None:
        """Update the conclusion page with results from OCR processing"""
        if not hasattr(self, "result_file_count") or not self.result_file_count:
            logger.warning("Conclusion page components not initialized")
            return

        # Make sure extracted_text dictionary is initialized
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Get processed file count from OCR processor
        file_count = self.window.ocr_processor.get_processed_count()
        self.result_file_count.set_text(str(file_count))

        # Calculate total page count
        page_count = 0
        total_file_size = 0

        # Clear output file list
        while True:
            child = self.output_list_box.get_first_child()
            if child:
                self.output_list_box.remove(child)
            else:
                break

        # Process output files and gather stats
        for output_file in self.window.settings.processed_files:
            # Skip if file doesn't exist or wasn't created
            if not os.path.exists(output_file):
                continue

            # Get filename for display
            filename = os.path.basename(output_file)

            # Verify the file was created during this OCR run (within the last 5 minutes)
            try:
                file_creation_time = os.path.getctime(output_file)
                if time.time() - file_creation_time > 300:  # 300 seconds = 5 minutes
                    logger.warning(
                        f"Skipping {filename} as it was not created in this OCR run"
                    )
                    continue
            except:
                # If we can't get creation time, we'll trust the filename
                pass

            # Get file info
            try:
                # Get file size
                file_size = os.path.getsize(output_file)
                total_file_size += file_size

                # Get page count
                result = subprocess.run(
                    ["pdfinfo", output_file],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                pages = 0  # Default value
                for line in result.stdout.split("\n"):
                    if line.startswith("Pages:"):
                        pages = int(line.split(":")[1].strip())
                        page_count += pages
                        break

                # Add file to list
                row = Adw.ActionRow()
                row.set_title(os.path.basename(output_file))
                row.set_subtitle(os.path.dirname(output_file))

                # Add page count
                page_label = Gtk.Label()
                page_label.set_markup(f"<small>{pages} pg.</small>")
                row.add_suffix(page_label)

                # Add size
                size_label = Gtk.Label()
                size_label.set_markup(f"<small>{self._format_size(file_size)}</small>")
                row.add_suffix(size_label)

                # Add file icon
                file_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
                row.add_prefix(file_icon)

                # Create a button container for better spacing and organization
                button_container = Gtk.Box(
                    orientation=Gtk.Orientation.HORIZONTAL, spacing=4
                )
                button_container.set_halign(Gtk.Align.END)

                # Add open button
                open_button = Gtk.Button()
                open_button.set_icon_name("document-open-symbolic")
                open_button.set_tooltip_text(_("Open file"))
                open_button.add_css_class("circular")
                open_button.set_valign(Gtk.Align.CENTER)
                open_button.set_margin_end(12)
                open_button.set_margin_start(12)

                open_button.connect(
                    "clicked", lambda _b, f=output_file: self._open_file(f)
                )
                button_container.append(open_button)

                # Add view text button
                self._load_extracted_text_if_needed(output_file)

                # Always add the text button
                text_button = Gtk.Button()
                text_button.set_icon_name("format-text-uppercase-symbolic")
                text_button.set_tooltip_text(_("View extracted text"))
                text_button.add_css_class("circular")
                text_button.set_valign(Gtk.Align.CENTER)
                text_button.connect(
                    "clicked",
                    lambda _b, f=output_file: self._show_extracted_text(f),
                )
                button_container.append(text_button)

                # Add the button container to the row
                row.add_suffix(button_container)

                # Add the row to the list box
                self.output_list_box.append(row)
            except Exception as e:
                logger.error(f"Error processing output file {output_file}: {e}")

        # Update page count
        self.result_page_count.set_text(str(page_count))

        # Update processing time
        if hasattr(self.window, "process_start_time"):
            elapsed_time = time.time() - self.window.process_start_time
            minutes = int(elapsed_time / 60)
            seconds = int(elapsed_time % 60)
            self.result_time.set_text(f"{minutes:02d}:{seconds:02d}")
        else:
            self.result_time.set_text("--:--")

        # Update file size
        self.result_file_size.set_text(self._format_size(total_file_size))

    def _load_extracted_text_if_needed(self, output_file: str) -> None:
        """Load extracted text for a file if not already loaded

        Args:
            output_file: Path to the output file
        """
        # Ensure the extracted_text dictionary exists
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Try to load any extracted text if not already in memory
        if not (
            output_file in self.window.settings.extracted_text
            and self.window.settings.extracted_text[output_file]
        ):
            # Check for sidecar text file (regular location)
            sidecar_file = os.path.splitext(output_file)[0] + ".txt"
            if os.path.exists(sidecar_file):
                try:
                    with open(sidecar_file, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                        self.window.settings.extracted_text[output_file] = (
                            extracted_text
                        )
                        logger.info(
                            f"Loaded {len(extracted_text)} chars from sidecar file for {os.path.basename(output_file)}"
                        )
                except Exception as e:
                    logger.error(f"Error reading sidecar file: {e}")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        # Handle edge cases
        if size_bytes < 0:
            return "0 B"

        # Define size units
        units = ["B", "KB", "MB", "GB", "TB"]

        # Determine unit to use
        i = 0
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024
            i += 1

        # Format with appropriate precision
        if i == 0:  # Bytes
            return f"{int(size_bytes)} {units[i]}"
        else:
            return f"{size_bytes:.2f} {units[i]}"

    def _open_file(self, file_path: str) -> None:
        """Open a file using the default application

        Args:
            file_path: Path to the file to open
        """
        try:
            subprocess.Popen(["xdg-open", file_path])
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")

    def _show_extracted_text(self, file_path: str) -> None:
        """Display the extracted text from a PDF file in a dialog

        Args:
            file_path: Path to the PDF file
        """
        # Make sure extracted_text dictionary exists
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Make sure we have text for this file
        if file_path not in self.window.settings.extracted_text:
            logger.warning(
                _("No extracted text available for {0}").format(
                    os.path.basename(file_path)
                )
            )
            # Try to read from the sidecar text file if it exists
            sidecar_file = os.path.splitext(file_path)[0] + ".txt"
            if os.path.exists(sidecar_file):
                try:
                    with open(sidecar_file, "r", encoding="utf-8") as f:
                        self.window.settings.extracted_text[file_path] = f.read()
                    logger.info(
                        _("Read {0} characters from sidecar file").format(
                            len(self.window.settings.extracted_text[file_path])
                        )
                    )
                except Exception as e:
                    logger.error(f"Error reading sidecar file: {e}")
                    # Show error dialog
                    dialog = Adw.MessageDialog(transient_for=self.window)
                    dialog.set_heading(_("Text Not Available"))
                    dialog.set_body(
                        _(
                            "Could not read extracted text for this file. Error: {0}"
                        ).format(str(e))
                    )
                    dialog.add_response("ok", _("OK"))
                    dialog.present()
                    return
            else:
                # Try looking for a temporary file in the .temp directory first
                temp_dir = os.path.join(os.path.dirname(file_path), ".temp")
                if os.path.exists(temp_dir):
                    temp_filename = (
                        f"temp_{os.path.basename(os.path.splitext(file_path)[0])}.txt"
                    )
                    temp_sidecar = os.path.join(temp_dir, temp_filename)
                    if os.path.exists(temp_sidecar):
                        try:
                            with open(temp_sidecar, "r", encoding="utf-8") as f:
                                self.window.settings.extracted_text[file_path] = (
                                    f.read()
                                )
                            logger.info(f"Found text in temporary file: {temp_sidecar}")
                        except Exception as e:
                            logger.error(f"Error reading temp sidecar file: {e}")

                # If no file exists, provide a generic message
                if file_path not in self.window.settings.extracted_text:
                    self.window.settings.extracted_text[file_path] = (
                        "OCR processing was completed for this file, but the extracted text could not be found."
                    )

        extracted_text = self.window.settings.extracted_text[file_path]
        if not extracted_text or not extracted_text.strip():
            # Even if no text was extracted, provide a generic message
            extracted_text = "No text content was detected in this file during OCR processing.\n\nThis could happen if the PDF contains only images without recognizable text or if OCR processing encountered issues."

        # Show the text dialog
        self._show_text_dialog(extracted_text, file_path)

    def _show_text_dialog(self, extracted_text: str, file_path: str) -> None:
        """Show the extracted text in a dialog

        Args:
            extracted_text: The text to display
            file_path: Path to the source file
        """
        # Create a dialog to display the text
        dialog = Gtk.Window()
        dialog.set_title(f"Extracted Text - {os.path.basename(file_path)}")
        dialog.set_default_size(700, 500)
        dialog.set_modal(True)
        dialog.set_transient_for(self.window)

        # Create a box for the content
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content_box.set_margin_top(16)
        content_box.set_margin_bottom(16)
        content_box.set_margin_start(16)
        content_box.set_margin_end(16)

        # Add a search box for large texts
        search_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_box.set_margin_bottom(8)

        search_entry = Gtk.SearchEntry()
        search_entry.set_placeholder_text(_("Search in text..."))
        search_entry.set_hexpand(True)
        search_box.append(search_entry)

        # Add previous/next search buttons
        prev_button = Gtk.Button()
        prev_button.set_icon_name("go-up-symbolic")
        prev_button.set_tooltip_text(_("Find previous match"))
        prev_button.add_css_class("flat")
        prev_button.set_sensitive(False)  # Initially disabled
        search_box.append(prev_button)

        next_button = Gtk.Button()
        next_button.set_icon_name("go-down-symbolic")
        next_button.set_tooltip_text(_("Find next match"))
        next_button.add_css_class("flat")
        next_button.set_sensitive(False)  # Initially disabled
        search_box.append(next_button)

        # Add the search box to the content
        content_box.append(search_box)

        # Create a scrolled window to contain the text
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)

        # Create a text view to display the text
        text_view = Gtk.TextView()
        text_view.set_editable(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        text_view.set_monospace(True)
        text_view.set_left_margin(12)
        text_view.set_right_margin(12)
        text_view.set_top_margin(12)
        text_view.set_bottom_margin(12)

        # Add the text to the text view
        buffer = text_view.get_buffer()
        buffer.set_text(extracted_text)

        # Add text tags for search highlighting
        tag_table = buffer.get_tag_table()
        highlight_tag = Gtk.TextTag.new("search_highlight")
        highlight_tag.set_property("background", "#ffff00")  # Yellow highlight
        highlight_tag.set_property("foreground", "#000000")  # Black text
        tag_table.add(highlight_tag)

        # Add a status label for search results
        status_label = Gtk.Label()
        status_label.set_halign(Gtk.Align.START)
        status_label.add_css_class("caption")
        status_label.add_css_class("dim-label")

        # Set up search functionality
        search_positions = []
        current_match = -1

        def on_search_changed(entry):
            nonlocal search_positions, current_match
            search_text = entry.get_text().lower()

            # Clear previous highlights
            start_iter = buffer.get_start_iter()
            end_iter = buffer.get_end_iter()
            buffer.remove_tag_by_name("search_highlight", start_iter, end_iter)

            # Reset search positions
            search_positions = []
            current_match = -1

            if not search_text:
                prev_button.set_sensitive(False)
                next_button.set_sensitive(False)
                status_label.set_text("")
                return

            # Find all occurrences of the search text
            text = extracted_text.lower()
            pos = 0
            while True:
                pos = text.find(search_text, pos)
                if pos == -1:
                    break
                search_positions.append(pos)
                pos += 1

            # Highlight all matches
            for pos in search_positions:
                start_iter = buffer.get_iter_at_offset(pos)
                end_iter = buffer.get_iter_at_offset(pos + len(search_text))
                buffer.apply_tag_by_name("search_highlight", start_iter, end_iter)

            # Update status and buttons
            match_count = len(search_positions)
            status_label.set_text(_("{0} matches found").format(match_count))

            prev_button.set_sensitive(match_count > 0)
            next_button.set_sensitive(match_count > 0)

            # Go to first match
            if match_count > 0:
                goto_match(0)

        def goto_match(index):
            nonlocal current_match
            if not search_positions or index < 0 or index >= len(search_positions):
                return

            current_match = index
            pos = search_positions[index]

            # Scroll to the match
            match_iter = buffer.get_iter_at_offset(pos)
            text_view.scroll_to_iter(match_iter, 0.2, False, 0.0, 0.0)

            # Update status
            status_label.set_text(
                _("Match {0} of {1}").format(index + 1, len(search_positions))
            )

        def on_prev_clicked(_button):
            nonlocal current_match
            if current_match > 0:
                goto_match(current_match - 1)

        def on_next_clicked(_button):
            nonlocal current_match
            if current_match < len(search_positions) - 1:
                goto_match(current_match + 1)

        # Connect search signals
        search_entry.connect("search-changed", on_search_changed)
        prev_button.connect("clicked", on_prev_clicked)
        next_button.connect("clicked", on_next_clicked)

        # Add the text view to the scrolled window
        scrolled.set_child(text_view)

        # Add a status box to contain both the status label and buttons
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        status_box.set_margin_top(16)

        # Add the status label to the left side
        status_box.append(status_label)
        status_label.set_hexpand(True)

        # Add buttons to the right side
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        button_box.set_halign(Gtk.Align.END)

        # Add copy to clipboard button
        copy_button = Gtk.Button(label=_("Copy to Clipboard"))
        copy_button.set_tooltip_text(_("Copy the entire text to clipboard"))
        copy_button.connect(
            "clicked", lambda b: self._copy_text_to_clipboard(extracted_text)
        )
        button_box.append(copy_button)

        # Add save to file button
        save_button = Gtk.Button(label=_("Save as TXT"))
        save_button.set_tooltip_text(_("Save the extracted text to a .txt file"))
        save_button.connect(
            "clicked", lambda b: self._save_text_to_file(extracted_text, file_path)
        )
        button_box.append(save_button)

        # Add close button
        close_button = Gtk.Button(label=_("Close"))
        close_button.add_css_class("suggested-action")
        close_button.connect("clicked", lambda b: dialog.destroy())
        button_box.append(close_button)

        # Add components to the content box
        content_box.append(scrolled)
        content_box.append(status_box)
        status_box.append(button_box)

        # Add the content box to the dialog
        dialog.set_child(content_box)

        # Add keyboard shortcut controller for Ctrl+F to focus search
        key_controller = Gtk.EventControllerKey()
        dialog.add_controller(key_controller)

        def on_key_pressed(controller, keyval, keycode, state):
            # Check for Ctrl+F
            if keyval == Gdk.KEY_f and state & Gdk.ModifierType.CONTROL_MASK:
                search_entry.grab_focus()
                return True
            return False

        key_controller.connect("key-pressed", on_key_pressed)

        # Show the dialog
        dialog.present()

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy the text to the clipboard

        Args:
            text: The text to copy to clipboard
        """
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        logger.info("Text copied to clipboard")

    def _save_text_to_file(self, text: str, source_pdf_path: str) -> None:
        """Save the text to a file

        Args:
            text: The text to save
            source_pdf_path: Path to the source PDF file
        """
        # Generate a simple suggested filename
        base_name = os.path.splitext(os.path.basename(source_pdf_path))[0]
        suggested_filename = f"{base_name}-text.txt"

        # Create a file dialog for saving
        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Save Extracted Text"))
        save_dialog.set_modal(True)
        save_dialog.set_initial_name(suggested_filename)

        # Use the same folder as the output PDF by default
        initial_folder = os.path.dirname(source_pdf_path)
        if os.path.exists(initial_folder):
            save_dialog.set_initial_folder(Gio.File.new_for_path(initial_folder))

        # Show the save dialog
        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda dialog, result: self._on_save_text_callback(
                dialog, result, text
            ),
        )

    def _on_save_text_callback(self, dialog: Gtk.FileDialog, result, text: str) -> None:
        """Handle save text dialog callback

        Args:
            dialog: The file dialog
            result: The async result
            text: The text to save
        """
        try:
            file = dialog.save_finish(result)
            if file:
                file_path = file.get_path()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"Text saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving text file: {e}")
