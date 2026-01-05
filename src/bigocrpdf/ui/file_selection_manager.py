"""
BigOcrPdf - File Selection Manager Module

Handles all file and folder selection dialogs for the application.
"""

import gi

gi.require_version("Gtk", "4.0")
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from gi.repository import Gio, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from window import BigOcrPdfWindow


class FileSelectionManager:
    """
    Manages file and folder selection dialogs.

    This class encapsulates all file selection functionality including:
    - Opening multiple PDF files
    - Selecting destination folders
    - Setting initial folder locations
    """

    def __init__(self, window: "BigOcrPdfWindow"):
        """
        Initialize the file selection manager.

        Args:
            window: Reference to the main application window
        """
        self.window = window

    @property
    def settings(self):
        """Get the settings from the window."""
        return self.window.settings

    @property
    def ui(self):
        """Get the UI manager from the window."""
        return self.window.ui

    def show_open_files_dialog(self, callback: Callable[[list[str]], None] | None = None) -> None:
        """
        Show dialog for selecting multiple PDF files.

        Args:
            callback: Optional callback to call with list of selected file paths.
                     If None, files are added to settings directly.
        """
        file_chooser = Gtk.FileDialog.new()
        file_chooser.set_title(_("Select PDF Files"))
        file_chooser.set_modal(True)

        # Create PDF filter
        pdf_filter = Gtk.FileFilter()
        pdf_filter.set_name(_("PDF Files"))
        pdf_filter.add_mime_type("application/pdf")
        pdf_filter.add_pattern("*.pdf")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(pdf_filter)
        file_chooser.set_filters(filters)

        # Store callback for later use
        self._open_callback = callback

        file_chooser.open_multiple(
            parent=self.window,
            cancellable=None,
            callback=self._on_open_multiple_finished,
        )

    def _on_open_multiple_finished(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """
        Handle completion of the open multiple files dialog.

        Args:
            dialog: The file dialog that was completed
            result: The async result containing selected files
        """
        try:
            files = dialog.open_multiple_finish(result)

            if not files or files.get_n_items() == 0:
                return

            # Extract file paths
            file_paths = self._extract_file_paths(files)

            if not file_paths:
                return

            # Call custom callback or default behavior
            if self._open_callback:
                self._open_callback(file_paths)
            else:
                self._add_files_to_settings(file_paths)

        except Exception as e:
            logger.error(f"Error adding files: {e}")
            self.window.show_toast(_("Error adding files"))
        finally:
            self._open_callback = None

    def _extract_file_paths(self, files: Gio.ListModel) -> list[str]:
        """
        Extract file paths from a Gio ListModel.

        Args:
            files: ListModel containing Gio.File objects

        Returns:
            List of file path strings
        """
        file_paths = []
        for i in range(files.get_n_items()):
            file = files.get_item(i)
            if isinstance(file, Gio.File):
                path = file.get_path()
                if path:
                    file_paths.append(path)
        return file_paths

    def _add_files_to_settings(self, file_paths: list[str]) -> None:
        """
        Add files to settings and update UI.

        Args:
            file_paths: List of file paths to add
        """
        added = self.settings.add_files(file_paths)

        if added > 0:
            self.window.update_file_info()
        else:
            logger.warning(_("No valid files were selected"))
            self.window.show_toast(_("No valid PDF files were selected"))

    def show_folder_selection_dialog(self, callback: Callable[[str], None] | None = None) -> None:
        """
        Show dialog for selecting a destination folder.

        Args:
            callback: Optional callback to call with selected folder path.
                     If None, folder is set in settings directly.
        """
        folder_chooser = Gtk.FileDialog.new()
        folder_chooser.set_title(_("Select destination folder"))
        folder_chooser.set_modal(True)

        self._set_initial_folder(folder_chooser)

        # Store callback for later use
        self._folder_callback = callback

        folder_chooser.select_folder(
            parent=self.window, cancellable=None, callback=self._on_folder_selected
        )

    def _set_initial_folder(self, folder_chooser: Gtk.FileDialog) -> None:
        """
        Set the initial folder for a folder chooser dialog.

        Args:
            folder_chooser: The file dialog to configure
        """
        if not self.settings.destination_folder:
            return

        initial_folder = self._get_valid_initial_folder()

        if initial_folder and os.path.exists(initial_folder):
            folder = Gio.File.new_for_path(initial_folder)
            folder_chooser.set_initial_folder(folder)

    def _get_valid_initial_folder(self) -> str | None:
        """
        Get a valid initial folder path from settings.

        Returns:
            Valid folder path or None
        """
        dest = self.settings.destination_folder

        if not dest:
            return None

        if os.path.isdir(dest):
            return dest

        # Try parent directory
        parent = os.path.dirname(dest)
        if os.path.isdir(parent):
            return parent

        return None

    def _on_folder_selected(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """
        Handle the folder selection dialog response.

        Args:
            dialog: The file dialog that was completed
            result: The async result containing the selected folder
        """
        try:
            folder = dialog.select_folder_finish(result)

            if not folder:
                return

            path = folder.get_path()

            if not path:
                return

            # Call custom callback or default behavior
            if self._folder_callback:
                self._folder_callback(path)
            else:
                self._set_destination_folder(path)

        except Exception as e:
            logger.error(f"Error selecting save location: {e}")
            self.window.show_toast(_("Error selecting destination folder"))
        finally:
            self._folder_callback = None

    def _set_destination_folder(self, path: str) -> None:
        """
        Set the destination folder in settings and update UI.

        Args:
            path: The selected folder path
        """
        self.settings.destination_folder = path

        if hasattr(self.ui, "dest_entry") and self.ui.dest_entry:
            self.ui.dest_entry.set_text(path)

        logger.info(f"Destination folder selected: {path}")
        self.window.show_toast(_("Destination folder selected"))
