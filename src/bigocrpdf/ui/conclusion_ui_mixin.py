"""Conclusion Page Statistics and File List Mixin."""

import os
import time
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow

from bigocrpdf.config import FILE_RECENCY_THRESHOLD_SECONDS
from bigocrpdf.ui.components import create_icon_button
from bigocrpdf.utils.comparison import PDFComparisonResult, get_batch_statistics
from bigocrpdf.utils.format_utils import format_file_size
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import get_pdf_page_count


class ConclusionStatsFileListMixin:
    """Mixin providing conclusion page statistics update and file list management."""

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
        self.result_size_change = None
        self.output_list_box = None

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
        self.result_file_size.set_text(format_file_size(total_file_size))

        # Update processing time
        self._update_processing_time()

        # Update size change from comparisons
        self._update_size_change()

    def _update_size_change(self) -> None:
        """Update the size change label from comparison results."""
        if not self.result_size_change:
            return

        # Get comparison results if available
        if not hasattr(self.window.settings, "comparison_results"):
            self.result_size_change.set_text("--")
            return

        results = self.window.settings.comparison_results
        if not results:
            self.result_size_change.set_text("--")
            return

        # Calculate aggregate statistics
        stats = get_batch_statistics(results)
        total_input = stats["total_input_size_bytes"]

        # Use actual file sizes from processed_files instead of comparison output sizes.
        # When PDFs are split into parts, comparisons only track one part's size,
        # but processed_files contains all split parts with correct sizes.
        _, total_output = self._calculate_file_statistics()

        if total_input <= 0:
            self.result_size_change.set_text("--")
            return

        change_percent = round(((total_output - total_input) / total_input) * 100, 1)

        # Format size change with color indication
        change_text = f"{format_file_size(total_input)} → {format_file_size(total_output)}"
        sign = "+" if change_percent >= 0 else ""
        change_text += f" ({sign}{change_percent:.1f}%)"

        self.result_size_change.set_text(change_text)

        # Add visual indication via CSS class
        self.result_size_change.remove_css_class("success")
        self.result_size_change.remove_css_class("warning")
        if change_percent < 0:
            self.result_size_change.add_css_class("success")  # Got smaller
        elif change_percent > 50:
            self.result_size_change.add_css_class("warning")  # Got much bigger

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
                pages = get_pdf_page_count(output_file)
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
            return time.time() - file_creation_time <= FILE_RECENCY_THRESHOLD_SECONDS
        except Exception:
            # If we can't get creation time, assume it's valid
            return True

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
            pages = get_pdf_page_count(output_file)

            # Find comparison result for this file
            comparison = self._get_comparison_for_file(output_file)

            # Create row for the file
            row = self._create_file_row(output_file, pages, file_size, comparison)
            self.output_list_box.append(row)

        except Exception as e:
            logger.error(f"Error adding file to list {output_file}: {e}")

    def _get_comparison_for_file(self, output_file: str) -> PDFComparisonResult | None:
        """Get comparison result for a specific output file.

        Args:
            output_file: Path to the output file

        Returns:
            PDFComparisonResult or None if not found
        """
        if not hasattr(self.window.settings, "comparison_results"):
            return None

        for comparison in self.window.settings.comparison_results:
            if comparison.output_path == output_file:
                return comparison
        return None

    def _create_file_row(
        self,
        output_file: str,
        pages: int,
        file_size: int,
        comparison: PDFComparisonResult | None = None,
    ) -> Adw.ActionRow:
        """Create a row for a processed file

        Args:
            output_file: Path to the output file
            pages: Number of pages
            file_size: File size in bytes
            comparison: Optional comparison result for size change display

        Returns:
            An Adw.ActionRow for the file
        """
        row = Adw.ActionRow()
        row.set_title(os.path.basename(output_file))
        row.set_subtitle(os.path.dirname(output_file))

        # Add file statistics
        self._add_file_statistics_to_row(row, pages, file_size, comparison)

        # Add file icon
        file_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        row.add_prefix(file_icon)

        # Add action buttons
        button_container = self._create_file_action_buttons(output_file)
        row.add_suffix(button_container)

        return row

    def _add_file_statistics_to_row(
        self,
        row: Adw.ActionRow,
        pages: int,
        file_size: int,
        comparison: PDFComparisonResult | None = None,
    ) -> None:
        """Add file statistics to a file row

        Args:
            row: The row to add statistics to
            pages: Number of pages
            file_size: File size in bytes
            comparison: Optional comparison result for size change
        """
        # Add page count
        page_label = Gtk.Label()
        page_label.set_markup(f"<small>{pages} pg.</small>")
        row.add_suffix(page_label)

        # Add size label
        size_label = Gtk.Label()
        size_label.set_markup(f"<small>{format_file_size(file_size)}</small>")
        row.add_suffix(size_label)

        # Add size change indicator with theme-aware CSS classes
        if comparison and comparison.input_size_bytes > 0:
            change_pct = comparison.size_change_percent
            sign = "+" if change_pct >= 0 else ""
            change_label = Gtk.Label()
            change_label.set_markup(f"<small>({sign}{change_pct:.0f}%)</small>")
            change_label.add_css_class("caption")
            if change_pct < 0:
                change_label.add_css_class("success")
            elif change_pct > 50:
                change_label.add_css_class("warning")
            row.add_suffix(change_label)

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

        # Add open in browser button
        browser_button = self._create_browser_button(output_file)
        button_container.append(browser_button)

        # Add view text button
        text_button = self._create_text_button(output_file)
        button_container.append(text_button)

        # Add export to ODF button
        odf_button = self._create_odf_button(output_file)
        button_container.append(odf_button)

        return button_container

    def _create_open_button(self, output_file: str) -> Gtk.Button:
        """Create an open file button

        Args:
            output_file: Path to the file to open

        Returns:
            A Gtk.Button for opening the file
        """
        button = create_icon_button(
            icon_name="document-open-symbolic",
            tooltip=_("Open the processed file"),
            on_click=lambda: self._open_file(output_file),
        )
        button.set_margin_end(12)
        button.set_margin_start(12)
        return button

    def _create_browser_button(self, output_file: str) -> Gtk.Button:
        """Create an open in browser button

        Args:
            output_file: Path to the file to open

        Returns:
            A Gtk.Button for opening the file in the browser
        """
        return create_icon_button(
            icon_name="web-browser-symbolic",
            tooltip=_(
                "Open in web browser — browsers usually display "
                "the recognized text directly over the page images"
            ),
            on_click=lambda: self._open_in_browser(output_file),
        )

    def _create_text_button(self, output_file: str) -> Gtk.Button:
        """Create a view text button

        Args:
            output_file: Path to the file

        Returns:
            A Gtk.Button for viewing extracted text
        """
        return create_icon_button(
            icon_name="format-text-uppercase-symbolic",
            tooltip=_("View the text found in this document"),
            on_click=lambda: self._show_extracted_text(output_file),
        )

    def _create_odf_button(self, output_file: str) -> Gtk.Button:
        """Create an export to ODF button

        Args:
            output_file: Path to the file

        Returns:
            A Gtk.Button for exporting to ODF
        """
        return create_icon_button(
            icon_name="x-office-document-symbolic",
            tooltip=_("Save as a document for LibreOffice"),
            on_click=lambda: self._export_to_odf(output_file),
        )

    def _export_to_odf(self, file_path: str) -> None:
        """Export extracted text to ODF file

        Args:
            file_path: Path to the PDF file
        """
        # Show export options dialog first
        self._show_odf_export_options_dialog(file_path)
