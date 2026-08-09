"""Truthful queue outcomes for PDF editor save callbacks."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bigocrpdf.ui.settings_queue_mixin import SettingsQueueMixin


def _manager(replace_result: bool) -> tuple[SettingsQueueMixin, SimpleNamespace]:
    settings = SimpleNamespace(
        _replace_file=MagicMock(return_value=replace_result),
        file_modifications={},
    )
    ui = SimpleNamespace(
        update_file_info=MagicMock(),
        show_toast=MagicMock(),
    )
    manager = SettingsQueueMixin.__new__(SettingsQueueMixin)
    manager.window = SimpleNamespace(settings=settings, ui=ui)
    manager.refresh_queue_status = MagicMock()
    return manager, settings


def test_failed_queue_replacement_does_not_announce_editor_success() -> None:
    manager, settings = _manager(False)
    document = SimpleNamespace(path="/tmp/merged.pdf")

    with patch(
        "bigocrpdf.ui.settings_queue_mixin._",
        side_effect=lambda text: text,
    ):
        saved = manager._handle_pdf_editor_save("/input.pdf", document)

    assert saved is False
    settings._replace_file.assert_called_once_with(
        "/input.pdf",
        "/tmp/merged.pdf",
    )
    manager.window.ui.update_file_info.assert_not_called()
    manager.refresh_queue_status.assert_not_called()
    manager.window.ui.show_toast.assert_not_called()


def test_successful_queue_replacement_announces_editor_success() -> None:
    manager, _settings = _manager(True)
    document = SimpleNamespace(path="/tmp/merged.pdf")

    with patch(
        "bigocrpdf.ui.settings_queue_mixin._",
        side_effect=lambda text: text,
    ):
        saved = manager._handle_pdf_editor_save("/input.pdf", document)

    assert saved is True
    manager.window.ui.update_file_info.assert_called_once_with()
    manager.refresh_queue_status.assert_called_once_with()
    manager.window.ui.show_toast.assert_called_once_with("Changes saved")
