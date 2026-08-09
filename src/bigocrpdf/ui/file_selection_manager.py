"""File and destination selection for the main OCR window."""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import images_to_pdf, is_image_file

if TYPE_CHECKING:
    from bigocrpdf.services.settings import OcrSettings
    from bigocrpdf.ui.window_ui import BigOcrPdfUI


def _dialog_was_cancelled(error: GLib.Error) -> bool:
    return (
        error.matches(Gtk.DialogError.quark(), Gtk.DialogError.CANCELLED)
        or error.matches(Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED)
        or error.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED)
    )


class FileSelectionManager:
    """Own file and destination-folder dialogs for the main window."""

    def __init__(
        self,
        parent: Adw.ApplicationWindow,
        settings: "OcrSettings",
        ui: "BigOcrPdfUI",
    ) -> None:
        self._parent = parent
        self.settings = settings
        self.ui = ui

    def show_open_files_dialog(self, callback: Callable[[list[str]], None] | None = None) -> None:
        """Select local PDF and image files."""
        dialog = Gtk.FileDialog(title=_("Select Files"), modal=True)
        file_filter = Gtk.FileFilter(name=_("PDFs and Images"))
        file_filter.add_mime_type("application/pdf")
        for mime_type in (
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/tiff",
            "image/bmp",
            "image/avif",
        ):
            file_filter.add_mime_type(mime_type)

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(file_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(file_filter)
        dialog.open_multiple(
            parent=self._parent,
            cancellable=None,
            callback=lambda chooser, result: self._on_open_multiple_finished(
                chooser, result, callback
            ),
        )

    def _on_open_multiple_finished(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        callback: Callable[[list[str]], None] | None,
    ) -> None:
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error as error:
            if not _dialog_was_cancelled(error):
                logger.error("Error adding files: %s", error)
                self.ui.show_toast(_("Error adding files"))
            return

        file_paths: list[str] = []
        remote_selected = False
        for index in range(files.get_n_items()):
            file = files.get_item(index)
            if not isinstance(file, Gio.File):
                continue
            path = file.get_path()
            if path is None:
                remote_selected = True
            else:
                file_paths.append(path)

        if remote_selected:
            self.ui.show_toast(_("Remote locations are not supported"))
        if not file_paths:
            return
        if callback is not None:
            callback(file_paths)
        else:
            self._add_files_to_settings(file_paths)

    def _add_files_to_settings(self, file_paths: list[str]) -> None:
        """Add selected PDFs and materialize selected images as PDFs."""
        image_files = [path for path in file_paths if is_image_file(path)]
        pdf_files = [path for path in file_paths if not is_image_file(path)]

        if len(image_files) > 1:
            if pdf_files:
                self.settings.add_files(pdf_files)
            self._show_image_merge_dialog(image_files)
            return

        added = self.settings.add_files(pdf_files)
        conversion_failed = False
        if image_files:
            image_path = image_files[0]
            try:
                pdf_path = images_to_pdf([image_path])
            except (OSError, RuntimeError, ValueError) as error:
                conversion_failed = True
                logger.error("Failed to convert image to PDF: %s", error)
                self.ui.show_toast(_("Error adding files"))
            else:
                added += int(self.settings._add_generated_file(pdf_path, image_path))

        if added:
            self.ui.update_file_info()
        elif not conversion_failed:
            logger.warning("No valid files were selected")
            self.ui.show_toast(_("No valid files were selected"))

    def _show_image_merge_dialog(self, image_files: list[str]) -> None:
        """Ask whether multiple images should be merged or kept separate."""
        self.ui.dialogs_manager.show_image_merge_dialog(
            image_files,
            heading=_("Multiple Images Selected"),
            body=ngettext(
                "You selected {count} image. How would you like to add it?",
                "You selected {count} images. How would you like to add them?",
                len(image_files),
            ).format(count=len(image_files)),
            on_complete=self.ui.update_file_info,
        )

    def show_folder_selection_dialog(self, callback: Callable[[str], None] | None = None) -> None:
        """Select a local destination folder."""
        dialog = Gtk.FileDialog(title=_("Select destination folder"), modal=True)
        destination = self.settings.destination_folder
        if destination:
            initial_folder = (
                destination if os.path.isdir(destination) else os.path.dirname(destination)
            )
            if os.path.isdir(initial_folder):
                dialog.set_initial_folder(Gio.File.new_for_path(initial_folder))

        dialog.select_folder(
            parent=self._parent,
            cancellable=None,
            callback=lambda chooser, result: self._on_folder_selected(chooser, result, callback),
        )

    def _on_folder_selected(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        callback: Callable[[str], None] | None,
    ) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as error:
            if not _dialog_was_cancelled(error):
                logger.error("Error selecting destination folder: %s", error)
                self.ui.show_toast(_("Error selecting destination folder"))
            return

        path = folder.get_path()
        if path is None:
            self.ui.show_toast(_("Remote locations are not supported"))
            return
        if callback is not None:
            callback(path)
        else:
            self._set_destination_folder(path)

    def _set_destination_folder(self, path: str) -> None:
        self.settings.destination_folder = path
        destination_entry = self.ui.settings_page_manager.dest_entry
        if destination_entry is not None:
            destination_entry.set_text(path)
        logger.info("Destination folder selected: %s", path)
        self.ui.show_toast(_("Destination folder selected"))
