"""Contracts for PDF output option dialogs."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import gi
import pytest

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk

from bigocrpdf.ui.pdf_options_controller import PDFOptionsController, _sanitize_suffix


def _active(value: bool) -> MagicMock:
    row = MagicMock()
    row.get_active.return_value = value
    return row


def _date_group(
    *,
    selected: int = 0,
    year: bool = True,
    month: bool = True,
    day: bool = True,
    clock: bool = False,
) -> Any:
    return SimpleNamespace(
        include_date_row=_active(True),
        year_row=_active(year),
        month_row=_active(month),
        day_row=_active(day),
        time_row=_active(clock),
        format_row=SimpleNamespace(
            get_selected=lambda: selected,
            set_sensitive=MagicMock(),
        ),
    )


def _dialog_error(code: Gtk.DialogError) -> GLib.Error:
    return GLib.Error.new_literal(Gtk.DialogError.quark(), "dialog error", code)


def test_suffix_is_trimmed_sanitized_and_never_empty() -> None:
    assert _sanitize_suffix("  report:/?  ") == "report---"
    assert _sanitize_suffix("   ") == "ocr"


def test_date_preview_respects_selected_order_and_components() -> None:
    now = __import__("time").struct_time((2026, 8, 4, 12, 7, 0, 0, 0, -1))

    assert (
        PDFOptionsController._format_date_for_preview(_date_group(selected=0), now) == "2026-08-04"
    )
    assert (
        PDFOptionsController._format_date_for_preview(_date_group(selected=1), now) == "04-08-2026"
    )
    assert (
        PDFOptionsController._format_date_for_preview(
            _date_group(selected=2, year=False, clock=True), now
        )
        == "08-04-1207"
    )


def test_remote_text_folder_is_rejected() -> None:
    dialog = MagicMock()
    dialog.select_folder_finish.return_value = Gio.File.new_for_uri("sftp://example.invalid/output")
    label = MagicMock()
    selected = MagicMock()

    with patch("bigocrpdf.ui.pdf_options_controller.logger") as logger:
        PDFOptionsController._on_folder_selected(dialog, MagicMock(), label, selected)

    label.set_label.assert_not_called()
    selected.assert_not_called()
    logger.warning.assert_called_once_with("Remote locations are not supported")


@pytest.mark.parametrize(
    "code",
    (Gtk.DialogError.CANCELLED, Gtk.DialogError.DISMISSED),
)
def test_cancelled_folder_dialog_is_silent(code: Gtk.DialogError) -> None:
    dialog = MagicMock()
    dialog.select_folder_finish.side_effect = _dialog_error(code)

    with patch("bigocrpdf.ui.pdf_options_controller.logger") as logger:
        PDFOptionsController._on_folder_selected(
            dialog,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    logger.error.assert_not_called()


def test_local_text_folder_updates_label_and_sensitivity(tmp_path: Path) -> None:
    dialog = MagicMock()
    dialog.select_folder_finish.return_value = Gio.File.new_for_path(str(tmp_path))
    label = MagicMock()
    selected = MagicMock()

    PDFOptionsController._on_folder_selected(dialog, MagicMock(), label, selected)

    label.set_label.assert_called_once_with(str(tmp_path))
    selected.assert_called_once_with()


def test_setup_disables_save_until_required_folder_exists() -> None:
    settings = MagicMock()
    controller = PDFOptionsController(MagicMock(), settings)
    save_button = MagicMock()
    file_group: Any = SimpleNamespace(
        use_original_name_row=_active(False),
        suffix_row=MagicMock(),
        warning_row=MagicMock(),
    )
    file_group.suffix_row.get_text.return_value = "ocr"
    text_group: Any = SimpleNamespace(
        save_txt_row=_active(True),
        separate_folder_row=_active(True),
        text_folder_row=SimpleNamespace(
            folder_label=SimpleNamespace(get_label=lambda: "Not set"),
            folder_button=MagicMock(),
            set_sensitive=MagicMock(),
        ),
    )
    odf_group: Any = SimpleNamespace(
        save_odf_row=_active(False),
        include_images_row=MagicMock(),
    )
    date_group: Any = _date_group()
    for row in (
        file_group.use_original_name_row,
        file_group.suffix_row,
        text_group.save_txt_row,
        text_group.separate_folder_row,
        text_group.text_folder_row.folder_button,
        odf_group.save_odf_row,
        date_group.include_date_row,
        date_group.year_row,
        date_group.month_row,
        date_group.day_row,
        date_group.time_row,
        date_group.format_row,
    ):
        row.connect = MagicMock()
    date_group.format_row.get_selected = lambda: 0
    prefs_page: Any = SimpleNamespace(
        file_group=file_group,
        text_group=text_group,
        odf_group=odf_group,
        date_group=date_group,
        preview_group=SimpleNamespace(preview_value=MagicMock()),
    )

    controller._setup_callbacks(MagicMock(), prefs_page, save_button, MagicMock())

    save_button.set_sensitive.assert_called_with(False)


def test_save_rejects_missing_required_text_folder() -> None:
    settings = MagicMock()
    callbacks = PDFOptionsController(MagicMock(), settings)
    dialog = MagicMock()
    callback = MagicMock()
    file_group: Any = SimpleNamespace(
        use_original_name_row=_active(False),
        suffix_row=SimpleNamespace(get_text=lambda: "ocr"),
        overwrite_row=_active(False),
    )
    text_group: Any = SimpleNamespace(
        save_txt_row=_active(True),
        separate_folder_row=_active(True),
        text_folder_row=SimpleNamespace(folder_label=SimpleNamespace(get_label=lambda: "Not set")),
    )
    odf_group: Any = SimpleNamespace(save_odf_row=_active(False), include_images_row=_active(True))
    date_group = _date_group()

    callbacks._save_pdf_options(
        dialog,
        file_group,
        text_group,
        odf_group,
        date_group,
        callback,
    )

    settings._save_all_settings.assert_not_called()
    dialog.close.assert_not_called()
    callback.assert_not_called()


def test_save_does_not_confirm_when_persistence_fails(tmp_path: Path) -> None:
    settings = MagicMock()
    settings._save_all_settings.return_value = False
    controller = PDFOptionsController(MagicMock(), settings)
    dialog = MagicMock()
    callback = MagicMock()
    file_group: Any = SimpleNamespace(
        use_original_name_row=_active(False),
        suffix_row=SimpleNamespace(get_text=lambda: "ocr"),
        overwrite_row=_active(False),
    )
    text_group: Any = SimpleNamespace(
        save_txt_row=_active(False),
        separate_folder_row=_active(False),
        text_folder_row=SimpleNamespace(folder_label=SimpleNamespace(get_label=lambda: "Not set")),
    )
    odf_group: Any = SimpleNamespace(save_odf_row=_active(False), include_images_row=_active(True))

    controller._save_pdf_options(
        dialog,
        file_group,
        text_group,
        odf_group,
        _date_group(),
        callback,
    )

    dialog.close.assert_not_called()
    callback.assert_not_called()


def test_save_persists_sanitized_values_and_confirms(tmp_path: Path) -> None:
    settings = MagicMock()
    settings._save_all_settings.return_value = True
    callbacks = PDFOptionsController(MagicMock(), settings)
    dialog = MagicMock()
    callback = MagicMock()
    file_group: Any = SimpleNamespace(
        use_original_name_row=_active(False),
        suffix_row=SimpleNamespace(get_text=lambda: "  searchable:?  "),
        overwrite_row=_active(True),
    )
    text_group: Any = SimpleNamespace(
        save_txt_row=_active(True),
        separate_folder_row=_active(True),
        text_folder_row=SimpleNamespace(
            folder_label=SimpleNamespace(get_label=lambda: str(tmp_path))
        ),
    )
    odf_group: Any = SimpleNamespace(save_odf_row=_active(True), include_images_row=_active(False))
    date_group = _date_group(selected=1, clock=True)

    callbacks._save_pdf_options(
        dialog,
        file_group,
        text_group,
        odf_group,
        date_group,
        callback,
    )

    assert settings.pdf_suffix == "searchable--"
    assert settings.txt_folder == str(tmp_path)
    assert settings.date_format_order == {"day": 1, "month": 2, "year": 3}
    settings._save_all_settings.assert_called_once_with()
    dialog.close.assert_called_once_with()
    callback.assert_called_once_with(True)
