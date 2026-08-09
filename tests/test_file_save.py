"""Tests for safe extracted-text publication."""

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")

from unittest.mock import MagicMock

from gi.repository import Gio, GLib, Gtk

from bigocrpdf.ui.file_save_controller import FileSaveController


def _controller() -> FileSaveController:
    return FileSaveController(MagicMock(), MagicMock())


def _dialog_error(code: Gtk.DialogError, message: str = "dialog error") -> GLib.Error:
    return GLib.Error.new_literal(Gtk.DialogError.quark(), message, code)


def test_dismissed_save_dialog_is_silent() -> None:
    controller = _controller()
    dialog = MagicMock()
    dialog.save_finish.side_effect = _dialog_error(Gtk.DialogError.DISMISSED)
    controller._show_error_dialog = MagicMock()

    controller._on_save_dialog_response(dialog, MagicMock(), "text")

    controller._show_error_dialog.assert_not_called()


def test_failed_save_dialog_reports_the_error() -> None:
    controller = _controller()
    dialog = MagicMock()
    dialog.save_finish.side_effect = _dialog_error(Gtk.DialogError.FAILED, "portal failed")
    controller._show_error_dialog = MagicMock()

    controller._on_save_dialog_response(dialog, MagicMock(), "text")

    controller._show_error_dialog.assert_called_once()
    assert "portal failed" in controller._show_error_dialog.call_args.args[1]


def test_remote_save_destination_is_rejected() -> None:
    controller = _controller()
    dialog = MagicMock()
    dialog.save_finish.return_value = Gio.File.new_for_uri("sftp://example.invalid/extracted.txt")
    controller._show_error_dialog = MagicMock()

    controller._on_save_dialog_response(dialog, MagicMock(), "text")

    controller._show_error_dialog.assert_called_once_with(
        "Save Failed", "Remote locations are not supported"
    )


def test_broken_symlink_requires_an_explicit_choice(tmp_path: Path) -> None:
    target = tmp_path / "text.txt"
    target.symlink_to(tmp_path / "missing.txt")
    controller = _controller()
    controller._show_file_exists_dialog = MagicMock()
    dialog = MagicMock()
    dialog.save_finish.return_value = Gio.File.new_for_path(str(target))

    controller._on_save_dialog_response(dialog, MagicMock(), "text")

    controller._show_file_exists_dialog.assert_called_once_with(str(target), "text")


def test_auto_rename_preserves_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "text.txt"
    target.write_text("existing", encoding="utf-8")
    controller = _controller()

    controller._write_text_to_file(str(target), "new", overwrite=False)

    assert target.read_text(encoding="utf-8") == "existing"
    assert (tmp_path / "text-1.txt").read_text(encoding="utf-8") == "new"


def test_overwrite_replaces_symlink_without_touching_target(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP", encoding="utf-8")
    target = tmp_path / "text.txt"
    target.symlink_to(victim)
    controller = _controller()

    controller._write_text_to_file(str(target), "replacement", overwrite=True)

    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "replacement"


def test_missing_parent_is_reported_without_creating_directories(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "text.txt"
    controller = _controller()
    controller._show_error_dialog = MagicMock()

    controller._write_text_to_file(str(target), "text", overwrite=False)

    assert not target.parent.exists()
    controller._show_error_dialog.assert_called_once()
