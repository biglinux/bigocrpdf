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


class TerminalPageManager:
    """Manages the terminal/processing page UI and interactions"""
    
    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the terminal page manager
        
        Args:
            window: Reference to the main application window
        """
        self.window = window
        
        # UI component references
        self.terminal_progress_bar = None
        self.terminal_status_bar = None
        self.terminal_spinner = None
        
        # Progress tracking
        self.progress_timer_id = None
        self._animation_progress = 0.05
        self._animation_direction = 0.01

    def create_terminal_page(self) -> Gtk.Box:
        """Create the processing page with progress display

        Returns:
            A Gtk.Box containing the processing UI
        """
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)

        # Create a card for progress visualization
        progress_card = self._create_progress_card()
        main_box.append(progress_card)

        return main_box

    def _create_progress_card(self) -> Gtk.Box:
        """Create the progress card container
        
        Returns:
            A Gtk.Box containing the progress card
        """
        progress_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        progress_card.set_margin_bottom(8)
        progress_card.add_css_class("card")
        progress_card.set_vexpand(True)

        # Create a centered progress area
        progress_area = self._create_progress_area()
        progress_card.append(progress_area)

        return progress_card

    def _create_progress_area(self) -> Gtk.Box:
        """Create the centered progress area
        
        Returns:
            A Gtk.Box containing the progress elements
        """
        progress_area = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        progress_area.set_valign(Gtk.Align.CENTER)
        progress_area.set_vexpand(True)
        progress_area.set_margin_start(24)
        progress_area.set_margin_end(24)
        progress_area.set_margin_bottom(24)

        # Add visual elements
        self._add_progress_icon(progress_area)
        self._add_progress_label(progress_area)
        self._add_progress_bar(progress_area)
        self._add_status_label(progress_area)
        self._add_cancel_button(progress_area)

        return progress_area

    def _add_progress_icon(self, container: Gtk.Box) -> None:
        """Add the PDF processing icon
        
        Args:
            container: Container to add the icon to
        """
        # Add file icon and progress visualization
        pdf_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        pdf_icon.set_pixel_size(48)
        pdf_icon.set_margin_bottom(16)
        pdf_icon.set_halign(Gtk.Align.CENTER)
        container.append(pdf_icon)

    def _add_progress_label(self, container: Gtk.Box) -> None:
        """Add the main progress label
        
        Args:
            container: Container to add the label to
        """
        # Add current file label
        current_file_label = Gtk.Label()
        current_file_label.set_markup("<big>" + _("Processing PDF files...") + "</big>")
        current_file_label.set_halign(Gtk.Align.CENTER)
        current_file_label.set_margin_bottom(24)
        container.append(current_file_label)

    def _add_progress_bar(self, container: Gtk.Box) -> None:
        """Add the progress bar
        
        Args:
            container: Container to add the progress bar to
        """
        # Add a large progress indicator
        self.terminal_progress_bar = Gtk.ProgressBar()
        self.terminal_progress_bar.set_show_text(True)
        self.terminal_progress_bar.set_text("0%")
        self.terminal_progress_bar.set_fraction(0)
        self.terminal_progress_bar.set_margin_bottom(8)
        container.append(self.terminal_progress_bar)

    def _add_status_label(self, container: Gtk.Box) -> None:
        """Add the status label
        
        Args:
            container: Container to add the status label to
        """
        # Create status label with file counter
        self.terminal_status_bar = Gtk.Label(label=_("Preparing processing..."))
        self.terminal_status_bar.add_css_class("body")
        self.terminal_status_bar.set_halign(Gtk.Align.CENTER)
        self.terminal_status_bar.set_margin_bottom(8)
        container.append(self.terminal_status_bar)

    def _add_cancel_button(self, container: Gtk.Box) -> None:
        """Add the cancel button
        
        Args:
            container: Container to add the cancel button to
        """
        # Add a cancel button
        cancel_button = Gtk.Button()
        cancel_button.set_label(_("Cancel"))
        cancel_button.add_css_class("destructive-action")
        cancel_button.set_halign(Gtk.Align.CENTER)
        cancel_button.set_margin_top(16)
        cancel_button.connect("clicked", lambda b: self.window.on_cancel_clicked())
        container.append(cancel_button)

    def start_progress_monitor(self) -> None:
        """Start monitoring the OCR progress"""
        # Set up a timer to update the progress
        self.progress_timer_id = GLib.timeout_add(
            250, lambda: self._update_ocr_progress()
        )

    def stop_progress_monitor(self) -> None:
        """Stop the progress monitor"""
        if self.progress_timer_id is not None:
            safe_remove_source(self.progress_timer_id)
            self.progress_timer_id = None

    def update_processing_status(self, input_file: str = None) -> None:
        """Update the status bar with current processing information
        
        Args:
            input_file: Currently processing file (optional)
        """
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

        # Log processing complete for this file
        if input_file:
            logger.info(
                _("Processed file {current}/{total}: {filename}").format(
                    current=file_count,
                    total=total_files,
                    filename=os.path.basename(input_file)
                )
            )

    def update_terminal_progress(self, fraction: float, text: str = None) -> None:
        """Update the terminal progress bar
        
        Args:
            fraction: Progress fraction (0.0-1.0)
            text: Optional text to display
        """
        if self.terminal_progress_bar:
            self.terminal_progress_bar.set_fraction(fraction)
            if text:
                self.terminal_progress_bar.set_text(text)

    def update_terminal_status_complete(self) -> None:
        """Update terminal status to show completion"""
        if self.terminal_status_bar:
            # Get the total files processed from the OCR queue
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
        """Update the OCR progress in the UI

        Returns:
            True to continue updating, False to stop
        """
        # Check if we're still on the terminal page
        if not self._is_on_terminal_page():
            return False

        # Calculate and update the progress
        progress = self._calculate_progress()
        
        # Update the UI with progress information
        self._update_progress_ui(progress)
        
        return True

    def _is_on_terminal_page(self) -> bool:
        """Check if we're on the terminal page and handle timer if not

        Returns:
            True if on terminal page, False otherwise
        """
        if self.window.stack.get_visible_child_name() != "terminal":
            # If we've left the terminal page, stop the timer
            self.stop_progress_monitor()
            return False
        return True

    def _calculate_progress(self) -> float:
        """Calculate the current progress value

        Returns:
            Progress value between 0.0 and 1.0
        """
        progress = self.window.ocr_processor.get_progress()

        # Use a timebased animation if the progress is still at 0
        # This gives feedback to the user that processing is happening
        if progress == 0:
            # Oscillate between 0.05 and 0.15 to show activity
            if self._animation_progress >= 0.15:
                self._animation_direction = -0.01
            elif self._animation_progress <= 0.05:
                self._animation_direction = 0.01

            self._animation_progress += self._animation_direction
            
            # Use animation progress during active processing
            return self._animation_progress
        else:
            # Use real progress
            return progress

    def _update_progress_ui(self, progress: float) -> None:
        """Update all progress UI elements
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        # Update progress bar
        if self.terminal_progress_bar:
            self.terminal_progress_bar.set_fraction(progress)
            progress_percent = int(progress * 100)
            self.terminal_progress_bar.set_text(f"{progress_percent}%")

        # Update spinner to indicate activity
        if self.terminal_spinner:
            if not self.terminal_spinner.get_spinning():
                self.terminal_spinner.set_spinning(True)
                
        # Update status text
        self._update_progress_status()

    def _update_progress_status(self) -> None:
        """Update the status text with current progress information"""
        if not self.terminal_status_bar:
            return

        # Get file counts from OCR processor
        files_processed = self.window.ocr_processor.get_processed_count()
        total_files = self.window.ocr_processor.get_total_count()
        elapsed_time = 0

        if hasattr(self.window, "process_start_time"):
            elapsed_time = int(time.time() - self.window.process_start_time)

        # Format the elapsed time
        time_str = self._format_elapsed_time(elapsed_time)

        remaining = total_files - files_processed
        progress = self.window.ocr_processor.get_progress()

        # Show the current status
        if progress >= 1.0 or files_processed >= total_files:
            self._show_completion_status(total_files, time_str)
        elif files_processed > 0:
            self._show_processing_status(files_processed, total_files, remaining, time_str)
        else:
            self._show_initial_status(total_files, elapsed_time, time_str)

    def _format_elapsed_time(self, elapsed_time: int) -> str:
        """Format elapsed time for display
        
        Args:
            elapsed_time: Elapsed time in seconds
            
        Returns:
            Formatted time string
        """
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        return f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    def _show_completion_status(self, total_files: int, time_str: str) -> None:
        """Show completion status
        
        Args:
            total_files: Total number of files
            time_str: Formatted time string
        """
        self.terminal_status_bar.set_markup(
            _("<b>OCR processing complete!</b> {total} file(s) processed • Total time: {time}").format(
                total=total_files, time=time_str
            )
        )

        # Stop the spinner
        self.stop_terminal_spinner()

        # Stop the timer
        self.stop_progress_monitor()

    def _show_processing_status(self, files_processed: int, total_files: int, 
                              remaining: int, time_str: str) -> None:
        """Show processing status
        
        Args:
            files_processed: Number of files processed
            total_files: Total number of files
            remaining: Number of remaining files
            time_str: Formatted time string
        """
        current_file = (
            files_processed + 1
            if files_processed < total_files
            else total_files
        )
        self.terminal_status_bar.set_markup(
            _("<b>Processing file {cur}/{total}:</b> {done} file(s) completed • <b>{rem}</b> remaining • Time: {time}").format(
                cur=current_file,
                total=total_files,
                done=files_processed,
                rem=remaining,
                time=time_str,
            )
        )

    def _show_initial_status(self, total_files: int, elapsed_time: int, time_str: str) -> None:
        """Show initial processing status
        
        Args:
            total_files: Total number of files
            elapsed_time: Elapsed time in seconds
            time_str: Formatted time string
        """
        # Show different stages of processing based on elapsed time
        stage = self._get_processing_stage(elapsed_time)
        
        self.terminal_status_bar.set_markup(
            _("<b>Processing file 1/{total}:</b> {stage} • Time: {time}").format(
                total=total_files, stage=stage, time=time_str
            )
        )

    def _get_processing_stage(self, elapsed_time: int) -> str:
        """Get the current processing stage based on elapsed time
        
        Args:
            elapsed_time: Elapsed time in seconds
            
        Returns:
            String describing the current stage
        """
        if elapsed_time < 5:
            return _("starting conversion")
        elif elapsed_time < 10:
            return _("analyzing document")
        elif elapsed_time < 15:
            return _("applying OCR")
        elif elapsed_time < 20:
            return _("processing texts")
        else:
            return _("finalizing processing")

    def show_completion_ui(self) -> None:
        """Update UI to show processing completion"""
        # Update the progress display to show 100%
        self.update_terminal_progress(1.0, "100%")

        # Update the status text
        self.update_terminal_status_complete()

        # Stop the spinner if it's still spinning
        self.stop_terminal_spinner()

        # Stop progress monitoring
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

        # Reset animation state
        self._animation_progress = 0.05
        self._animation_direction = 0.01

        # Stop any running timer
        self.stop_progress_monitor()

    def cleanup(self) -> None:
        """Clean up resources and stop timers"""
        self.stop_progress_monitor()
        self.stop_terminal_spinner()