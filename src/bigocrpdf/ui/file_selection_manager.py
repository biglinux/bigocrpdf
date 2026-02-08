"""
BigOcrPdf - File Selection Manager Module

Handles all file and folder selection dialogs for the application.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import images_to_pdf, is_image_file

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
        Show dialog for selecting PDF and image files.

        Args:
            callback: Optional callback to call with list of selected file paths.
                     If None, files are added to settings directly.
        """
        file_chooser = Gtk.FileDialog.new()
        file_chooser.set_title(_("Select Files"))
        file_chooser.set_modal(True)

        # Create filter for PDFs and Images
        file_filter = Gtk.FileFilter()
        file_filter.set_name(_("PDFs and Images"))
        file_filter.add_mime_type("application/pdf")
        file_filter.add_pattern("*.pdf")

        # Add image patterns
        for pattern in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff", "*.bmp", "*.avif"]:
            file_filter.add_pattern(pattern)
            file_filter.add_pattern(pattern.upper())

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(file_filter)
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

        If multiple image files are provided, shows a dialog asking whether
        to treat them as separate PDFs or merge them into one.

        Args:
            file_paths: List of file paths to add
        """
        # Separate images from PDFs
        image_files = [p for p in file_paths if is_image_file(p)]
        pdf_files = [p for p in file_paths if not is_image_file(p)]

        # If multiple images selected, ask merge or separate
        if len(image_files) > 1:
            # Add PDFs immediately
            if pdf_files:
                self.settings.add_files(pdf_files)

            # Show merge dialog for images
            self._show_image_merge_dialog(image_files)
        else:
            # Single image or only PDFs or mix with ≤1 image — convert images to PDF and add
            converted_paths = list(pdf_files)
            for img_path in image_files:
                try:
                    pdf_path = images_to_pdf([img_path])
                    converted_paths.append(pdf_path)
                    # Track original path for output naming
                    self.settings.original_file_paths[pdf_path] = img_path
                except Exception as e:
                    logger.error(f"Failed to convert image to PDF: {e}")

            added = self.settings.add_files(converted_paths)
            if added > 0:
                self.window.update_file_info()
            else:
                logger.warning(_("No valid files were selected"))
                self.window.show_toast(_("No valid files were selected"))

    def _show_image_merge_dialog(self, image_files: list[str]) -> None:
        """Show dialog asking whether to merge images into one PDF or treat separately.

        Args:
            image_files: List of image file paths
        """
        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Multiple Images Selected"))
        dialog.set_body(
            _("You selected {} images. How would you like to add them?").format(len(image_files))
        )

        dialog.add_response("separate", _("Separate PDFs"))
        dialog.add_response("merge", _("Merge into One PDF"))
        dialog.set_response_appearance("merge", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("merge")

        dialog.connect("response", self._on_image_merge_response, image_files)
        dialog.present(self.window)

    def _on_image_merge_response(
        self, dialog: Adw.AlertDialog, response: str, image_files: list[str]
    ) -> None:
        """Handle the image merge dialog response.

        Args:
            dialog: The dialog
            response: Response ID ("merge" or "separate")
            image_files: List of image file paths
        """
        if response == "merge":
            # Merge all images into a single PDF
            try:
                pdf_path = images_to_pdf(image_files)
                self.settings.original_file_paths[pdf_path] = image_files[0]
                added = self.settings.add_files([pdf_path])
                if added > 0:
                    self.window.update_file_info()
                    self.window.show_toast(
                        _("Merged {} images into one PDF").format(len(image_files))
                    )
            except Exception as e:
                logger.error(f"Failed to merge images: {e}")
                self.window.show_toast(_("Error merging images"))
        elif response == "separate":
            # Convert each image to a separate PDF
            converted_paths = []
            for img_path in image_files:
                try:
                    pdf_path = images_to_pdf([img_path])
                    converted_paths.append(pdf_path)
                    self.settings.original_file_paths[pdf_path] = img_path
                except Exception as e:
                    logger.error(f"Failed to convert image to PDF: {e}")

            if converted_paths:
                added = self.settings.add_files(converted_paths)
                if added > 0:
                    self.window.update_file_info()
        # "close" (X button) does nothing

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
