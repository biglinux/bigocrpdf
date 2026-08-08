"""Save extracted text through native GTK dialogs."""

import os
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.utils.durable_writes import write_text_atomically
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class FileSaveController:
    """Own extracted-text publication dialogs."""

    def __init__(
        self,
        parent: Adw.ApplicationWindow,
        show_toast: Callable[[str], None],
    ) -> None:
        self._parent = parent
        self._show_toast = show_toast

    def save_text(self, text: str) -> None:
        """Select a destination for extracted plain text."""
        dialog = Gtk.FileDialog(
            title=_("Save Extracted Text"),
            modal=True,
            initial_name="extracted_text.txt",
        )
        file_filter = Gtk.FileFilter(name=_("Text Extraction"))
        file_filter.add_mime_type("text/plain")
        file_filter.add_pattern("*.txt")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(file_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(file_filter)
        dialog.save(
            parent=self._parent,
            cancellable=None,
            callback=lambda chooser, result: self._on_save_dialog_response(chooser, result, text),
        )

    def _on_save_dialog_response(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        text: str,
    ) -> None:
        try:
            file = dialog.save_finish(result)
        except GLib.Error as error:
            dismissed = (
                error.matches(Gtk.DialogError.quark(), Gtk.DialogError.CANCELLED)
                or error.matches(Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED)
                or error.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED)
            )
            if not dismissed:
                logger.error("Error choosing text destination: %s", error)
                self._show_error_dialog(_("Save Failed"), str(error))
            return

        file_path = file.get_path()
        if file_path is None:
            self._show_error_dialog(_("Save Failed"), _("Remote locations are not supported"))
        elif os.path.lexists(file_path):
            self._show_file_exists_dialog(file_path, text)
        else:
            self._write_text_to_file(file_path, text, overwrite=False)

    def _show_file_exists_dialog(self, file_path: str, text: str) -> None:
        dialog = Adw.AlertDialog(
            heading=_("File Already Exists"),
            body=_("The file '{0}' already exists. What would you like to do?").format(
                os.path.basename(file_path)
            ),
        )
        dialog.add_response("overwrite", _("Overwrite"))
        dialog.add_response("rename", _("Auto-Rename"))
        dialog.add_response("cancel", _("Cancel"))
        dialog.set_response_appearance("overwrite", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_response_appearance("rename", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("rename")
        dialog.set_close_response("cancel")
        dialog.connect("response", self._on_file_exists_response, file_path, text)
        dialog.present(self._parent)

    def _on_file_exists_response(
        self,
        _dialog: Adw.AlertDialog,
        response: str,
        file_path: str,
        text: str,
    ) -> None:
        if response == "overwrite":
            self._write_text_to_file(file_path, text, overwrite=True)
        elif response == "rename":
            self._write_text_to_file(file_path, text, overwrite=False)

    def _write_text_to_file(self, file_path: str, text: str, *, overwrite: bool) -> None:
        try:
            saved_path = write_text_atomically(file_path, text, overwrite=overwrite)
        except (OSError, ValueError) as error:
            logger.error("Error writing text to file: %s", error)
            self._show_error_dialog(_("Save Failed"), str(error))
            return

        logger.info("Text saved to %s", saved_path)
        self._show_toast(_("Text saved to {filename}").format(filename=saved_path.name))

    def _show_error_dialog(self, title: str, message: str) -> None:
        dialog = Adw.AlertDialog(heading=title, body=message)
        dialog.add_response("ok", _("OK"))
        dialog.set_close_response("ok")
        dialog.present(self._parent)
