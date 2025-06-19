"""
BigOcrPdf - Progress Window

This module contains the progress/processing window implementation.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.i18n import _


class ProgressWindow:
    """Progress/processing window for BigOcrPdf application"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize progress window

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # UI components references
        self.page_status_label = None
        self.elapsed_time_label = None
        self.file_status_label = None  # This will be the status_page itself
        self.terminal_progress_bar = None

    def create_terminal_page(self) -> Gtk.Widget:
        """Create the processing page with progress display using Adw.StatusPage.

        Returns:
            A widget containing the modern processing UI.
        """
        # Use Adw.StatusPage for a modern, clean look
        self.file_status_label = Adw.StatusPage.new()
        self.file_status_label.set_icon_name("document-send-symbolic")
        self.file_status_label.set_title(_("Processing..."))
        self.file_status_label.set_description(_("Preparing to process..."))
        self.file_status_label.set_vexpand(True)

        # Main content area for the StatusPage, containing our custom widgets
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_valign(Gtk.Align.CENTER)
        content_box.set_size_request(300, -1)  # Give it a reasonable width
        self.file_status_label.set_child(content_box)

        # Main page counter (large and prominent)
        self.page_status_label = Gtk.Label()
        self.page_status_label.set_markup("<span size='xx-large'>-</span>")
        self.page_status_label.set_margin_bottom(8)
        content_box.append(self.page_status_label)

        # Elapsed time label
        self.elapsed_time_label = Gtk.Label()
        self.elapsed_time_label.add_css_class("caption")
        self.elapsed_time_label.add_css_class("dim-label")
        self.elapsed_time_label.set_text(_("Elapsed time: 00:00"))
        self.elapsed_time_label.set_margin_bottom(12)
        content_box.append(self.elapsed_time_label)

        # Progress bar - no text, as info is in the labels above
        self.terminal_progress_bar = Gtk.ProgressBar()
        self.terminal_progress_bar.set_show_text(False)
        self.terminal_progress_bar.set_fraction(0)
        content_box.append(self.terminal_progress_bar)

        # Cancel button
        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.add_css_class("destructive-action")
        cancel_button.set_halign(Gtk.Align.CENTER)
        cancel_button.connect("clicked", lambda b: self.window.on_cancel_clicked())
        content_box.append(cancel_button)

        return self.file_status_label

    def update_progress(self, fraction: float) -> None:
        """Update the progress bar

        Args:
            fraction: Progress fraction (0.0 to 1.0)
        """
        if self.terminal_progress_bar:
            self.terminal_progress_bar.set_fraction(fraction)

    def update_page_status(self, current_page: int, total_pages: int) -> None:
        """Update the page status display

        Args:
            current_page: Current page being processed
            total_pages: Total number of pages
        """
        if self.page_status_label:
            self.page_status_label.set_markup(
                f"<span size='xx-large'>{_('Page {p} of {t}').format(p=current_page, t=total_pages)}</span>"
            )

    def update_file_status(self, description: str) -> None:
        """Update the file status description

        Args:
            description: Status description to display
        """
        if self.file_status_label:
            self.file_status_label.set_description(description)

    def update_elapsed_time(self, elapsed_time: str) -> None:
        """Update the elapsed time display

        Args:
            elapsed_time: Formatted elapsed time string
        """
        if self.elapsed_time_label:
            self.elapsed_time_label.set_text(f"{_('Elapsed time:')} {elapsed_time}")

    def set_processing_complete(self) -> None:
        """Set the UI to show processing is complete"""
        if self.file_status_label:
            self.file_status_label.set_icon_name("emblem-ok-symbolic")
            self.file_status_label.set_title(_("Processing Complete"))
            self.file_status_label.set_description(
                _("All files have been processed successfully")
            )

    def set_processing_cancelled(self) -> None:
        """Set the UI to show processing was cancelled"""
        if self.file_status_label:
            self.file_status_label.set_icon_name("process-stop-symbolic")
            self.file_status_label.set_title(_("Processing Cancelled"))
            self.file_status_label.set_description(
                _("Processing was cancelled by user")
            )

    def set_processing_error(self, error_message: str) -> None:
        """Set the UI to show processing error

        Args:
            error_message: Error message to display
        """
        if self.file_status_label:
            self.file_status_label.set_icon_name("dialog-error-symbolic")
            self.file_status_label.set_title(_("Processing Error"))
            self.file_status_label.set_description(error_message)


class ProcessingStatus:
    """Helper class to manage processing status updates"""

    def __init__(self, progress_window: ProgressWindow):
        """Initialize processing status

        Args:
            progress_window: Reference to the progress window
        """
        self.progress_window = progress_window
        self.current_file = ""
        self.files_processed = 0
        self.total_files = 0
        self.pages_processed = 0
        self.total_pages = 0
        self.start_time = 0

    def set_totals(self, total_files: int, total_pages: int) -> None:
        """Set the total counts

        Args:
            total_files: Total number of files to process
            total_pages: Total number of pages to process
        """
        self.total_files = total_files
        self.total_pages = total_pages

    def update_current_file(self, filename: str) -> None:
        """Update the current file being processed

        Args:
            filename: Name of the current file
        """
        self.current_file = filename
        self._update_display()

    def increment_pages(self) -> None:
        """Increment the pages processed count"""
        self.pages_processed += 1
        self._update_display()

    def increment_files(self) -> None:
        """Increment the files processed count"""
        self.files_processed += 1
        self._update_display()

    def _update_display(self) -> None:
        """Update the progress display"""
        # Update progress bar
        if self.total_pages > 0:
            progress = self.pages_processed / self.total_pages
            self.progress_window.update_progress(progress)

        # Update page status
        self.progress_window.update_page_status(self.pages_processed, self.total_pages)

        # Update file status
        if self.current_file:
            description = _("File {c} of {t}: {name}").format(
                c=self.files_processed + 1,
                t=self.total_files,
                name=self.current_file,
            )
            self.progress_window.update_file_status(description)
        else:
            self.progress_window.update_file_status(_("Finalizing..."))

    def reset(self) -> None:
        """Reset all counters"""
        self.current_file = ""
        self.files_processed = 0
        self.total_files = 0
        self.pages_processed = 0
        self.total_pages = 0
        self.start_time = 0
