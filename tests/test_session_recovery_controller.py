"""Contracts for interrupted-session UI recovery."""

from unittest.mock import ANY, MagicMock, patch

from bigocrpdf.ui.session_recovery_controller import SessionRecoveryController


def _controller(*, resumable: bool = True, resumed: bool = True, discarded: bool = True):
    processor = MagicMock()
    processor.has_resumable_session.return_value = resumable
    processor.get_resumable_session_info.return_value = {"pending_files": 2}
    processor.resume_previous_session.return_value = resumed
    processor.discard_previous_session.return_value = discarded
    settings = MagicMock()
    settings.selected_files = ["one.pdf", "two.pdf"]
    ui = MagicMock()
    processing = MagicMock()
    processing.ocr_processor = processor
    controller = SessionRecoveryController(
        MagicMock(),
        settings,
        ui,
        processing,
    )
    return controller, processor, ui


def test_check_presents_recovery_with_controller_owned_callbacks() -> None:
    controller, processor, ui = _controller()

    with patch("bigocrpdf.ui.session_recovery_controller.Adw.AlertDialog") as dialog_type:
        controller.check()

    processor.get_resumable_session_info.assert_called_once_with()
    dialog = dialog_type.return_value
    dialog.add_response.assert_any_call("cancel", ANY)
    dialog.set_close_response.assert_called_once_with("cancel")
    dialog.connect.assert_called_once_with("response", controller._on_response)
    dialog.present.assert_called_once_with(controller._parent)


def test_check_ignores_absent_session() -> None:
    controller, processor, ui = _controller(resumable=False)

    controller.check()

    processor.get_resumable_session_info.assert_not_called()
    ui.show_toast.assert_not_called()


def test_resume_refreshes_queue_and_reports_restored_count() -> None:
    controller, _processor, ui = _controller()

    with patch("bigocrpdf.ui.session_recovery_controller.ngettext", return_value="{count} files"):
        controller._resume()

    ui.settings_page_manager._populate_file_list.assert_called_once_with()
    ui.custom_header_bar.update_queue_size.assert_called_once_with(2)
    ui.show_toast.assert_called_once_with("2 files")


def test_failed_resume_and_discard_report_distinct_outcomes() -> None:
    controller, processor, ui = _controller(resumed=False)

    controller._resume()
    controller._discard()

    processor.discard_previous_session.assert_called_once_with()
    assert ui.show_toast.call_count == 2


def test_failed_discard_does_not_report_success() -> None:
    controller, processor, ui = _controller(discarded=False)

    controller._discard()

    processor.discard_previous_session.assert_called_once_with()
    ui.show_toast.assert_not_called()


def test_cancel_or_system_close_preserves_resumable_session() -> None:
    controller, processor, ui = _controller()

    controller._on_response(MagicMock(), "cancel")
    controller._on_response(MagicMock(), "close")

    processor.resume_previous_session.assert_not_called()
    processor.discard_previous_session.assert_not_called()
    ui.show_toast.assert_not_called()


def test_explicit_discard_response_discards_resumable_session() -> None:
    controller, processor, ui = _controller()

    controller._on_response(MagicMock(), "discard")

    processor.discard_previous_session.assert_called_once_with()
    ui.show_toast.assert_called_once_with(ANY)
