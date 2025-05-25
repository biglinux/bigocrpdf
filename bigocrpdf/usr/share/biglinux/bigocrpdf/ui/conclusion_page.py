"""
BigOcrPdf - Conclusion Page Module

This module handles the creation and management of the conclusion/results page UI.
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


class ConclusionPageManager:
    """Manages the conclusion/results page UI and interactions"""
    
    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the conclusion page manager
        
        Args:
            window: Reference to the main application window
        """
        self.window = window
        
        # UI component references for results display
        self.result_file_count = None
        self.result_page_count = None
        self.result_time = None
        self.result_file_size = None
        self.output_list_box = None

    def create_conclusion_page(self) -> Gtk.Box:
        """Create the conclusion page showing OCR processing results
        
        Returns:
            A Gtk.Box containing the conclusion page UI
        """
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        # Set individual margins
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        # Add summary card
        summary_card = self._create_summary_card()
        main_box.append(summary_card)

        # Add output files section
        files_card = self._create_files_card()
        main_box.append(files_card)

        return main_box

    def _create_summary_card(self) -> Gtk.Box:
        """Create the processing summary card
        
        Returns:
            A Gtk.Box containing the summary card
        """
        summary_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        summary_card.add_css_class("card")
        
        # Card header
        card_header = Gtk.Label(label=_("Processing Summary"))
        card_header.add_css_class("heading")
        card_header.set_halign(Gtk.Align.START)
        card_header.set_margin_top(16)
        card_header.set_margin_start(16)
        summary_card.append(card_header)

        # Create statistics grid
        stats_grid = self._create_statistics_grid()
        summary_card.append(stats_grid)

        return summary_card

    def _create_statistics_grid(self) -> Gtk.Box:
        """Create the statistics grid with two columns
        
        Returns:
            A Gtk.Box containing the statistics
        """
        # Create a horizontal box to contain two columns
        columns_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=32)
        columns_box.set_margin_start(16)
        columns_box.set_margin_end(16)
        columns_box.set_margin_bottom(16)
        columns_box.set_margin_top(8)

        # Create left and right columns
        left_grid = self._create_left_statistics_column()
        right_grid = self._create_right_statistics_column()

        # Add the two grids to the columns box
        columns_box.append(left_grid)
        columns_box.append(right_grid)

        return columns_box

    def _create_left_statistics_column(self) -> Gtk.Grid:
        """Create the left statistics column (Files and Pages)
        
        Returns:
            A Gtk.Grid containing the left column statistics
        """
        left_grid = Gtk.Grid()
        left_grid.set_column_spacing(16)
        left_grid.set_row_spacing(16)
        left_grid.set_hexpand(True)
        left_grid.set_halign(Gtk.Align.START)

        # Files processed
        self._add_statistic_row(
            left_grid, 0, "document-multiple-symbolic", 
            _("Files processed:"), self._create_file_count_label()
        )

        # Total pages
        self._add_statistic_row(
            left_grid, 1, "view-paged-symbolic", 
            _("Total pages:"), self._create_page_count_label()
        )

        return left_grid

    def _create_right_statistics_column(self) -> Gtk.Grid:
        """Create the right statistics column (Time and Size)
        
        Returns:
            A Gtk.Grid containing the right column statistics
        """
        right_grid = Gtk.Grid()
        right_grid.set_column_spacing(16)
        right_grid.set_row_spacing(16)
        right_grid.set_hexpand(True)
        right_grid.set_halign(Gtk.Align.START)

        # Processing time
        self._add_statistic_row(
            right_grid, 0, "clock-symbolic", 
            _("Processing time:"), self._create_time_label()
        )

        # Output file size
        self._add_statistic_row(
            right_grid, 1, "drive-harddisk-symbolic", 
            _("File size:"), self._create_file_size_label()
        )

        return right_grid

    def _add_statistic_row(self, grid: Gtk.Grid, row: int, icon_name: str, 
                          label_text: str, value_label: Gtk.Label) -> None:
        """Add a statistic row to a grid
        
        Args:
            grid: Grid to add the row to
            row: Row number
            icon_name: Name of the icon to display
            label_text: Text for the label
            value_label: Label widget for the value
        """
        # Icon
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        grid.attach(icon, 0, row, 1, 1)

        # Label
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_margin_start(8)
        grid.attach(label, 1, row, 1, 1)

        # Value
        value_label.set_halign(Gtk.Align.END)
        value_label.add_css_class("heading")
        value_label.set_hexpand(True)
        grid.attach(value_label, 2, row, 1, 1)

    def _create_file_count_label(self) -> Gtk.Label:
        """Create the file count label
        
        Returns:
            A Gtk.Label for displaying file count
        """
        self.result_file_count = Gtk.Label(label="0")
        return self.result_file_count

    def _create_page_count_label(self) -> Gtk.Label:
        """Create the page count label
        
        Returns:
            A Gtk.Label for displaying page count
        """
        self.result_page_count = Gtk.Label(label="0")
        return self.result_page_count

    def _create_time_label(self) -> Gtk.Label:
        """Create the processing time label
        
        Returns:
            A Gtk.Label for displaying processing time
        """
        self.result_time = Gtk.Label(label="00:00")
        return self.result_time

    def _create_file_size_label(self) -> Gtk.Label:
        """Create the file size label
        
        Returns:
            A Gtk.Label for displaying file size
        """
        self.result_file_size = Gtk.Label(label="0 KB")
        return self.result_file_size

    def _create_files_card(self) -> Gtk.Box:
        """Create the output files card
        
        Returns:
            A Gtk.Box containing the files card
        """
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

        # Create scrollable file list
        scrolled_list = self._create_scrollable_file_list()
        files_card.append(scrolled_list)

        return files_card

    def _create_scrollable_file_list(self) -> Gtk.ScrolledWindow:
        """Create the scrollable file list
        
        Returns:
            A Gtk.ScrolledWindow containing the file list
        """
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
        self.output_list_box.add_css_class("output-files-list")
        self.output_list_box.set_vexpand(True)

        scrolled.set_child(self.output_list_box)
        return scrolled

    def update_conclusion_page(self) -> None:
        """Update the conclusion page with results from OCR processing"""
        if not self._validate_components():
            return

        # Initialize extracted text dictionary
        self._ensure_extracted_text_dict()

        # Update statistics
        self._update_statistics()

        # Update file list
        self._update_file_list()

    def _validate_components(self) -> bool:
        """Validate that conclusion page components are initialized
        
        Returns:
            True if components are valid, False otherwise
        """
        if not self.result_file_count:
            logger.warning("Conclusion page components not initialized")
            return False
        return True

    def _ensure_extracted_text_dict(self) -> None:
        """Ensure the extracted_text dictionary exists"""
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

    def _update_statistics(self) -> None:
        """Update the statistics display"""
        # Get processed file count from OCR processor
        file_count = self.window.ocr_processor.get_processed_count()
        self.result_file_count.set_text(str(file_count))

        # Calculate and update other statistics
        page_count, total_file_size = self._calculate_file_statistics()
        
        self.result_page_count.set_text(str(page_count))
        self.result_file_size.set_text(self._format_size(total_file_size))

        # Update processing time
        self._update_processing_time()

    def _calculate_file_statistics(self) -> tuple[int, int]:
        """Calculate page count and total file size
        
        Returns:
            Tuple of (page_count, total_file_size)
        """
        page_count = 0
        total_file_size = 0

        for output_file in self.window.settings.processed_files:
            if not os.path.exists(output_file):
                continue

            # Verify the file was created during this OCR run
            if not self._is_recent_file(output_file):
                continue

            try:
                # Get file size
                file_size = os.path.getsize(output_file)
                total_file_size += file_size

                # Get page count
                pages = self._get_pdf_page_count(output_file)
                if pages:
                    page_count += pages

            except Exception as e:
                logger.error(f"Error processing output file {output_file}: {e}")

        return page_count, total_file_size

    def _is_recent_file(self, file_path: str) -> bool:
        """Check if file was created recently (within 5 minutes)
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is recent, False otherwise
        """
        try:
            file_creation_time = os.path.getctime(file_path)
            return time.time() - file_creation_time <= 300  # 5 minutes
        except Exception:
            # If we can't get creation time, assume it's valid
            return True

    def _get_pdf_page_count(self, file_path: str) -> int:
        """Get the page count of a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of pages, or 0 if unable to determine
        """
        try:
            result = subprocess.run(
                ["pdfinfo", file_path],
                capture_output=True,
                text=True,
                check=False,
            )

            for line in result.stdout.split("\n"):
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
        except Exception:
            pass
        return 0

    def _update_processing_time(self) -> None:
        """Update the processing time display"""
        if hasattr(self.window, "process_start_time"):
            elapsed_time = time.time() - self.window.process_start_time
            minutes = int(elapsed_time / 60)
            seconds = int(elapsed_time % 60)
            self.result_time.set_text(f"{minutes:02d}:{seconds:02d}")
        else:
            self.result_time.set_text("--:--")

    def _update_file_list(self) -> None:
        """Update the file list with processed files"""
        # Clear existing list
        self._clear_output_list()

        # Add each processed file
        for output_file in self.window.settings.processed_files:
            if os.path.exists(output_file) and self._is_recent_file(output_file):
                self._add_file_to_list(output_file)

    def _clear_output_list(self) -> None:
        """Clear the output file list"""
        while True:
            child = self.output_list_box.get_first_child()
            if child:
                self.output_list_box.remove(child)
            else:
                break

    def _add_file_to_list(self, output_file: str) -> None:
        """Add a file to the output file list
        
        Args:
            output_file: Path to the output file
        """
        try:
            # Get file information
            file_size = os.path.getsize(output_file)
            pages = self._get_pdf_page_count(output_file)

            # Create row for the file
            row = self._create_file_row(output_file, pages, file_size)
            self.output_list_box.append(row)

        except Exception as e:
            logger.error(f"Error adding file to list {output_file}: {e}")

    def _create_file_row(self, output_file: str, pages: int, file_size: int) -> Adw.ActionRow:
        """Create a row for a processed file
        
        Args:
            output_file: Path to the output file
            pages: Number of pages
            file_size: File size in bytes
            
        Returns:
            An Adw.ActionRow for the file
        """
        row = Adw.ActionRow()
        row.set_title(os.path.basename(output_file))
        row.set_subtitle(os.path.dirname(output_file))

        # Add file statistics
        self._add_file_statistics_to_row(row, pages, file_size)

        # Add file icon
        file_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        row.add_prefix(file_icon)

        # Add action buttons
        button_container = self._create_file_action_buttons(output_file)
        row.add_suffix(button_container)

        return row

    def _add_file_statistics_to_row(self, row: Adw.ActionRow, pages: int, file_size: int) -> None:
        """Add file statistics to a file row
        
        Args:
            row: The row to add statistics to
            pages: Number of pages
            file_size: File size in bytes
        """
        # Add page count
        page_label = Gtk.Label()
        page_label.set_markup(f"<small>{pages} pg.</small>")
        row.add_suffix(page_label)

        # Add size
        size_label = Gtk.Label()
        size_label.set_markup(f"<small>{self._format_size(file_size)}</small>")
        row.add_suffix(size_label)

    def _create_file_action_buttons(self, output_file: str) -> Gtk.Box:
        """Create action buttons for a file row
        
        Args:
            output_file: Path to the output file
            
        Returns:
            A Gtk.Box containing the action buttons
        """
        # Create a button container for better spacing and organization
        button_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        button_container.set_halign(Gtk.Align.END)

        # Add open button
        open_button = self._create_open_button(output_file)
        button_container.append(open_button)

        # Add view text button
        text_button = self._create_text_button(output_file)
        button_container.append(text_button)

        return button_container

    def _create_open_button(self, output_file: str) -> Gtk.Button:
        """Create an open file button
        
        Args:
            output_file: Path to the file to open
            
        Returns:
            A Gtk.Button for opening the file
        """
        open_button = Gtk.Button()
        open_button.set_icon_name("document-open-symbolic")
        open_button.set_tooltip_text(_("Open file"))
        open_button.add_css_class("circular")
        open_button.set_valign(Gtk.Align.CENTER)
        open_button.set_margin_end(12)
        open_button.set_margin_start(12)
        open_button.connect("clicked", lambda _b: self._open_file(output_file))
        return open_button

    def _create_text_button(self, output_file: str) -> Gtk.Button:
        """Create a view text button
        
        Args:
            output_file: Path to the file
            
        Returns:
            A Gtk.Button for viewing extracted text
        """
        text_button = Gtk.Button()
        text_button.set_icon_name("format-text-uppercase-symbolic")
        text_button.set_tooltip_text(_("View extracted text"))
        text_button.add_css_class("circular")
        text_button.set_valign(Gtk.Align.CENTER)
        text_button.connect("clicked", lambda _b: self._show_extracted_text(output_file))
        return text_button

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
        """Show extracted text dialog - placeholder for now
        
        Args:
            file_path: Path to the PDF file
        """
        # This will be implemented when we create the dialogs module
        # For now, import and call the method from ui_manager
        if hasattr(self.window, 'ui') and hasattr(self.window.ui, '_show_extracted_text'):
            self.window.ui._show_extracted_text(file_path)
        else:
            logger.warning("Text viewer dialog not yet implemented")
            # Simple fallback - show a basic dialog
            self._show_simple_text_dialog(file_path)

    def _show_simple_text_dialog(self, file_path: str) -> None:
        """Show a simple text dialog as fallback
        
        Args:
            file_path: Path to the PDF file
        """
        # Get extracted text
        extracted_text = self._get_extracted_text_for_file(file_path)
        
        # Create simple dialog
        dialog = Adw.MessageDialog(transient_for=self.window)
        dialog.set_heading(_("Extracted Text"))
        dialog.set_body(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
        dialog.add_response("ok", _("OK"))
        dialog.present()

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text or placeholder message
        """
        # Check if we have text for this file
        if (hasattr(self.window.settings, "extracted_text") and 
            file_path in self.window.settings.extracted_text):
            return self.window.settings.extracted_text[file_path]

        # Try to read from sidecar file
        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading sidecar file: {e}")

        # Return placeholder text
        return _("No text content was detected in this file during OCR processing.")

    def reset_page(self) -> None:
        """Reset the conclusion page to initial state"""
        if self.result_file_count:
            self.result_file_count.set_text("0")
        if self.result_page_count:
            self.result_page_count.set_text("0")
        if self.result_time:
            self.result_time.set_text("00:00")
        if self.result_file_size:
            self.result_file_size.set_text("0 KB")
        
        # Clear file list
        if self.output_list_box:
            self._clear_output_list()