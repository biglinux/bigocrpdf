"""Contracts for file and destination selection dialogs."""

from pathlib import Path
from typing import cast

import gi

gi.require_version("Gtk", "4.0")

from unittest.mock import MagicMock, patch

from gi.repository import Gio, GLib, Gtk

from bigocrpdf.ui.file_selection_manager import FileSelectionManager


def _manager() -> FileSelectionManager:
    return FileSelectionManager(MagicMock(), MagicMock(), MagicMock())


def _ui(manager: FileSelectionManager) -> MagicMock:
    return cast(MagicMock, manager.ui)


def _file_list(*files: Gio.File) -> MagicMock:
    model = MagicMock()
    model.get_n_items.return_value = len(files)
    model.get_item.side_effect = files
    return model


def _dialog_error(code: Gtk.DialogError, message: str = "dialog error") -> GLib.Error:
    return GLib.Error.new_literal(Gtk.DialogError.quark(), message, code)


def test_overlapping_file_dialogs_keep_their_original_callbacks() -> None:
    manager = _manager()
    first_files = _file_list(Gio.File.new_for_path("/first.pdf"))
    second_files = _file_list(Gio.File.new_for_path("/second.pdf"))
    first_dialog = MagicMock()
    first_dialog.open_multiple_finish.return_value = first_files
    second_dialog = MagicMock()
    second_dialog.open_multiple_finish.return_value = second_files
    first_callback = MagicMock()
    second_callback = MagicMock()

    manager._on_open_multiple_finished(second_dialog, MagicMock(), second_callback)
    manager._on_open_multiple_finished(first_dialog, MagicMock(), first_callback)

    first_callback.assert_called_once_with(["/first.pdf"])
    second_callback.assert_called_once_with(["/second.pdf"])


def test_dismissed_file_dialog_is_silent() -> None:
    manager = _manager()
    dialog = MagicMock()
    dialog.open_multiple_finish.side_effect = _dialog_error(Gtk.DialogError.DISMISSED)

    manager._on_open_multiple_finished(dialog, MagicMock(), MagicMock())

    _ui(manager).show_toast.assert_not_called()


def test_remote_files_are_skipped_without_losing_local_selection() -> None:
    manager = _manager()
    dialog = MagicMock()
    dialog.open_multiple_finish.return_value = _file_list(
        Gio.File.new_for_uri("sftp://example.invalid/document.pdf"),
        Gio.File.new_for_path("/local.pdf"),
    )
    callback = MagicMock()

    manager._on_open_multiple_finished(dialog, MagicMock(), callback)

    callback.assert_called_once_with(["/local.pdf"])
    _ui(manager).show_toast.assert_called_once_with("Remote locations are not supported")


def test_overlapping_folder_dialogs_keep_their_original_callbacks() -> None:
    manager = _manager()
    first_dialog = MagicMock()
    first_dialog.select_folder_finish.return_value = Gio.File.new_for_path("/first")
    second_dialog = MagicMock()
    second_dialog.select_folder_finish.return_value = Gio.File.new_for_path("/second")
    first_callback = MagicMock()
    second_callback = MagicMock()

    manager._on_folder_selected(second_dialog, MagicMock(), second_callback)
    manager._on_folder_selected(first_dialog, MagicMock(), first_callback)

    first_callback.assert_called_once_with("/first")
    second_callback.assert_called_once_with("/second")


def test_remote_destination_is_rejected() -> None:
    manager = _manager()
    dialog = MagicMock()
    dialog.select_folder_finish.return_value = Gio.File.new_for_uri("sftp://example.invalid/output")

    manager._set_destination_folder = MagicMock()

    manager._on_folder_selected(dialog, MagicMock(), None)

    manager._set_destination_folder.assert_not_called()
    _ui(manager).show_toast.assert_called_once_with("Remote locations are not supported")


def test_single_converted_image_publishes_its_original_path() -> None:
    settings = MagicMock()
    settings.add_files.return_value = 0
    settings._add_generated_file.return_value = True
    manager = FileSelectionManager(MagicMock(), settings, MagicMock())

    with patch(
        "bigocrpdf.ui.file_selection_manager.images_to_pdf",
        return_value="/tmp/generated.pdf",
    ):
        manager._add_files_to_settings(["source.png"])

    settings._add_generated_file.assert_called_once_with("/tmp/generated.pdf", "source.png")
    _ui(manager).update_file_info.assert_called_once_with()


def test_failed_image_conversion_keeps_added_pdfs(tmp_path: Path) -> None:
    settings = MagicMock()
    settings.add_files.return_value = 1
    manager = FileSelectionManager(MagicMock(), settings, MagicMock())
    image = tmp_path / "source.png"
    pdf = tmp_path / "document.pdf"

    with patch(
        "bigocrpdf.ui.file_selection_manager.images_to_pdf",
        side_effect=RuntimeError("conversion failed"),
    ):
        manager._add_files_to_settings([str(pdf), str(image)])

    settings.add_files.assert_called_once_with([str(pdf)])
    _ui(manager).update_file_info.assert_called_once_with()
    _ui(manager).show_toast.assert_called_once_with("Error adding files")
