"""
BigOcrPdf - Terminal Page Module

This module handles the creation and management of the processing/terminal page UI.
"""

import os
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GLib

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.logger import logger
from utils.i18n import _
from utils.timer import safe_remove_source

# Constants for smooth incremental progress display
PROGRESS_UPDATE_INTERVAL = 800   # Update every 800ms for smooth responsiveness
PROGRESS_MINIMAL_CHANGE = 0.01   # Update if progress changed by 1% (for incremental updates)


class TerminalPageManager:
    """Manages the terminal/processing page UI and interactions with smooth incremental progress"""
    
    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the terminal page manager"""
        self.window = window
        
        # UI component references
        self.terminal_progress_bar = None
        self.terminal_status_bar = None
        self.terminal_spinner = None
        
        # Smooth progress tracking
        self.progress_timer_id = None
        self._last_displayed_progress = 0.0
        self._last_displayed_text = ""
        self._last_status_text = ""

    def create_terminal_page(self) -> Gtk.Box:
        """Create the processing page with progress display"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        progress_card = self._create_progress_card()
        main_box.append(progress_card)

        return main_box

    def _create_progress_card(self) -> Gtk.Box:
        """Create the progress card container"""
        progress_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        progress_card.set_margin_bottom(8)
        progress_card.add_css_class("card")
        progress_card.set_vexpand(True)

        progress_area = self._create_progress_area()
        progress_card.append(progress_area)

        return progress_card

    def _create_progress_area(self) -> Gtk.Box:
        """Create the centered progress area"""
        progress_area = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        progress_area.set_valign(Gtk.Align.CENTER)
        progress_area.set_vexpand(True)
        progress_area.set_margin_start(24)
        progress_area.set_margin_end(24)
        progress_area.set_margin_bottom(24)

        self._add_progress_icon(progress_area)
        self._add_progress_label(progress_area)
        self._add_progress_bar(progress_area)
        self._add_status_label(progress_area)
        self._add_cancel_button(progress_area)

        return progress_area

    def _add_progress_icon(self, container: Gtk.Box) -> None:
        """Add the PDF processing icon"""
        pdf_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        pdf_icon.set_pixel_size(48)
        pdf_icon.set_margin_bottom(16)
        pdf_icon.set_halign(Gtk.Align.CENTER)
        container.append(pdf_icon)

    def _add_progress_label(self, container: Gtk.Box) -> None:
        """Add the main progress label"""
        current_file_label = Gtk.Label()
        current_file_label.set_markup("<big>" + _("Processing PDF files...") + "</big>")
        current_file_label.set_halign(Gtk.Align.CENTER)
        current_file_label.set_margin_bottom(24)
        container.append(current_file_label)

    def _add_progress_bar(self, container: Gtk.Box) -> None:
        """Add the progress bar"""
        self.terminal_progress_bar = Gtk.ProgressBar()
        self.terminal_progress_bar.set_show_text(True)
        self.terminal_progress_bar.set_text("0%")
        self.terminal_progress_bar.set_fraction(0)
        self.terminal_progress_bar.set_margin_bottom(8)
        container.append(self.terminal_progress_bar)

    def _add_status_label(self, container: Gtk.Box) -> None:
        """Add the status label"""
        self.terminal_status_bar = Gtk.Label(label=_("Preparing processing..."))
        self.terminal_status_bar.add_css_class("body")
        self.terminal_status_bar.set_halign(Gtk.Align.CENTER)
        self.terminal_status_bar.set_margin_bottom(8)
        container.append(self.terminal_status_bar)

    def _add_cancel_button(self, container: Gtk.Box) -> None:
        """Add the cancel button"""
        cancel_button = Gtk.Button()
        cancel_button.set_label(_("Cancel"))
        cancel_button.add_css_class("destructive-action")
        cancel_button.set_halign(Gtk.Align.CENTER)
        cancel_button.set_margin_top(16)
        cancel_button.connect("clicked", lambda b: self.window.on_cancel_clicked())
        container.append(cancel_button)

    def start_progress_monitor(self) -> None:
        """Start monitoring the OCR progress with smooth incremental updates"""
        # Stop any existing timer first
        self.stop_progress_monitor()
        
        # Reset tracking variables
        self._last_displayed_progress = 0.0
        self._last_displayed_text = ""
        self._last_status_text = ""
        
        # Set up timer with optimized interval for smooth incremental updates
        self.progress_timer_id = GLib.timeout_add(
            PROGRESS_UPDATE_INTERVAL, self._update_ocr_progress
        )

    def stop_progress_monitor(self) -> None:
        """Stop the progress monitor"""
        if self.progress_timer_id is not None:
            safe_remove_source(self.progress_timer_id)
            self.progress_timer_id = None

    def update_processing_status(self, input_file: str = None) -> None:
        """Update the status bar with current processing information"""
        if not self.window.ocr_processor:
            return
            
        file_count = self.window.ocr_processor.get_processed_count()
        total_files = self.window.ocr_processor.get_total_count()

        if self.terminal_status_bar:
            remaining = total_files - file_count
            self.terminal_status_bar.set_markup(
                _("<b>Processing: {current}/{total}:</b> {completed} file(s) completed • <b>{remaining}</b> remaining").format(
                    current=file_count,
                    total=total_files,
                    completed=file_count,
                    remaining=remaining
                )
            )

        if input_file:
            logger.info(
                _("Processed file {current}/{total}: {filename}").format(
                    current=file_count,
                    total=total_files,
                    filename=os.path.basename(input_file)
                )
            )

    def update_terminal_progress(self, fraction: float, text: str = None) -> None:
        """Update the terminal progress bar with incremental precision"""
        if not self.terminal_progress_bar:
            return
            
        # Update more frequently for incremental progress (1% changes)
        if abs(fraction - self._last_displayed_progress) >= PROGRESS_MINIMAL_CHANGE:
            self.terminal_progress_bar.set_fraction(fraction)
            self._last_displayed_progress = fraction
            
        if text and text != self._last_displayed_text:
            self.terminal_progress_bar.set_text(text)
            self._last_displayed_text = text

    def update_terminal_status_complete(self) -> None:
        """Update terminal status to show completion"""
        if self.terminal_status_bar and self.window.ocr_processor:
            total_files = self.window.ocr_processor.get_processed_count()
            self.terminal_status_bar.set_markup(
                _("<b>OCR processing complete!</b> {total} file(s) processed").format(
                    total=total_files
                )
            )

    def stop_terminal_spinner(self) -> None:
        """Stop the terminal spinner"""
        if self.terminal_spinner:
            self.terminal_spinner.set_spinning(False)

    def _update_ocr_progress(self) -> bool:
        """Update the OCR progress in the UI with smooth incremental updates"""
        # Check if we're still on the terminal page
        if not self._is_on_terminal_page():
            return False

        # Get current progress data
        progress_data = self._get_progress_data()
        if not progress_data:
            return True  # Continue checking
            
        # Update UI with smooth incremental changes
        self._update_progress_ui_incremental(progress_data)
        
        return True

    def _is_on_terminal_page(self) -> bool:
        """Check if we're on the terminal page"""
        if self.window.stack.get_visible_child_name() != "terminal":
            self.stop_progress_monitor()
            return False
        return True

    def _get_progress_data(self) -> dict:
        """Get current progress data from the processor"""
        if not self.window.ocr_processor:
            return None
            
        try:
            return {
                'progress': self.window.ocr_processor.get_progress(),
                'processed_files': self.window.ocr_processor.get_processed_count(),
                'total_files': self.window.ocr_processor.get_total_count(),
                'processed_pages': self.window.ocr_processor.get_processed_pages(),
                'total_pages': self.window.ocr_processor.get_total_pages(),
                'current_file_info': self.window.ocr_processor.get_current_file_info(),
                'is_processing': self.window.ocr_processor.is_processing()
            }
        except Exception as e:
            logger.warning(f"Error getting progress data: {e}")
            return None

    def _update_progress_ui_incremental(self, progress_data: dict) -> None:
        """Update progress UI with smooth incremental changes"""
        progress = progress_data.get('progress', 0.0)
        processed_pages = progress_data.get('processed_pages', 0)
        total_pages = progress_data.get('total_pages', 0)
        processed_files = progress_data.get('processed_files', 0)
        total_files = progress_data.get('total_files', 0)
        is_processing = progress_data.get('is_processing', True)
        
        # Update progress bar with incremental precision
        self._update_progress_bar_incremental(progress, processed_pages, total_pages)

        # Update status text for significant changes
        self._update_status_text_incremental(progress_data)

    def _update_progress_bar_incremental(self, progress: float, processed_pages: int, total_pages: int) -> None:
        """Update progress bar with smooth incremental display"""
        if not self.terminal_progress_bar:
            return
            
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Update with 1% precision for smooth experience
        if abs(progress - self._last_displayed_progress) >= PROGRESS_MINIMAL_CHANGE:
            self.terminal_progress_bar.set_fraction(progress)
            progress_percent = int(progress * 100)
            
            # Show page-based progress for more granular feedback
            if total_pages > 0:
                progress_text = f"{progress_percent}% ({processed_pages}/{total_pages} pages)"
            else:
                progress_text = f"{progress_percent}%"
            
            if progress_text != self._last_displayed_text:
                self.terminal_progress_bar.set_text(progress_text)
                self._last_displayed_text = progress_text
                
            self._last_displayed_progress = progress

    def _update_status_text_incremental(self, progress_data: dict) -> None:
        """Update status text with current processing information"""
        if not self.terminal_status_bar:
            return

        processed_files = progress_data.get('processed_files', 0)
        total_files = progress_data.get('total_files', 0)
        current_file_info = progress_data.get('current_file_info', {})
        is_processing = progress_data.get('is_processing', True)
        progress = progress_data.get('progress', 0.0)

        elapsed_time = 0
        if hasattr(self.window, "process_start_time"):
            elapsed_time = int(time.time() - self.window.process_start_time)

        time_str = self._format_elapsed_time(elapsed_time)

        # Determine status based on processing state
        if not is_processing or progress >= 1.0:
            self._show_completion_status_incremental(total_files, time_str)
        elif current_file_info and current_file_info.get("filename"):
            self._show_processing_status_incremental(current_file_info, time_str)
        elif processed_files > 0:
            self._show_simple_progress_status_incremental(processed_files, total_files, time_str)
        else:
            self._show_initial_status_incremental(total_files, time_str)

    def _format_elapsed_time(self, elapsed_time: int) -> str:
        """Format elapsed time for display"""
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        return f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    def _show_completion_status_incremental(self, total_files: int, time_str: str) -> None:
        """Show completion status"""
        status_text = _("<b>OCR processing complete!</b> {total} file(s) processed • Total time: {time}").format(
            total=total_files, time=time_str
        )
        
        if status_text != self._last_status_text:
            self.terminal_status_bar.set_markup(status_text)
            self._last_status_text = status_text
            
        self.stop_terminal_spinner()
        self.stop_progress_monitor()

    def _show_processing_status_incremental(self, current_file_info: dict, time_str: str) -> None:
        """Show processing status with current file and page information"""
        filename = current_file_info.get("filename", "")
        file_number = current_file_info.get("file_number", 1)
        total_files = current_file_info.get("total_files", 1)
        current_page = current_file_info.get("current_page", 0)
        total_pages = current_file_info.get("total_pages", 0)
        
        # Show incremental page progress within current file
        if current_page > 0 and total_pages > 0:
            status_text = _("Processing file {current}/{total}: {filename} (page {page}/{pages}) • Time: {time}").format(
                current=file_number,
                total=total_files,
                filename=filename,
                page=current_page,
                pages=total_pages,
                time=time_str
            )
        else:
            status_text = _("Processing file {current}/{total}: {filename} ({pages} pages) • Time: {time}").format(
                current=file_number,
                total=total_files,
                filename=filename,
                pages=total_pages,
                time=time_str
            )
        
        if status_text != self._last_status_text:
            self.terminal_status_bar.set_markup(status_text)
            self._last_status_text = status_text

    def _show_simple_progress_status_incremental(self, processed_files: int, total_files: int, time_str: str) -> None:
        """Show simple progress status without current file details"""
        status_text = _("Processing files: {processed}/{total} completed • Time: {time}").format(
            processed=processed_files,
            total=total_files,
            time=time_str
        )
        
        if status_text != self._last_status_text:
            self.terminal_status_bar.set_markup(status_text)
            self._last_status_text = status_text

    def _show_initial_status_incremental(self, total_files: int, time_str: str) -> None:
        """Show initial processing status"""
        status_text = _("Starting processing of {total} files... • Time: {time}").format(
            total=total_files, time=time_str
        )
        
        if status_text != self._last_status_text:
            self.terminal_status_bar.set_markup(status_text)
            self._last_status_text = status_text

    def show_completion_ui(self) -> None:
        """Update UI to show processing completion"""
        self.update_terminal_progress(1.0, "100%")
        self.update_terminal_status_complete()
        self.stop_terminal_spinner()
        self.stop_progress_monitor()

    def reset_progress(self) -> None:
        """Reset progress indicators to initial state"""
        if self.terminal_progress_bar:
            self.terminal_progress_bar.set_fraction(0.0)
            self.terminal_progress_bar.set_text("0%")

        if self.terminal_status_bar:
            self.terminal_status_bar.set_text(_("Preparing processing..."))

        if self.terminal_spinner:
            self.terminal_spinner.set_spinning(False)

        # Reset tracking variables
        self._last_displayed_progress = 0.0
        self._last_displayed_text = ""
        self._last_status_text = ""

        self.stop_progress_monitor()

    def cleanup(self) -> None:
        """Clean up resources and stop timers"""
        self.stop_progress_monitor()
        self.stop_terminal_spinner()