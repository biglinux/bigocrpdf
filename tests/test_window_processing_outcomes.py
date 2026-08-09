"""Headless contracts for truthful OCR batch completion states."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from bigocrpdf.processing_controller import (
    ProcessingController,
    _BatchCompletionAction,
    _BatchCompletionState,
    _build_batch_completion,
)


@pytest.mark.parametrize(
    ("total", "successful", "state", "action", "message"),
    (
        (
            1,
            1,
            _BatchCompletionState.SUCCESS,
            _BatchCompletionAction.SHOW_RESULTS,
            "OCR processing completed successfully for 1 file",
        ),
        (
            3,
            3,
            _BatchCompletionState.SUCCESS,
            _BatchCompletionAction.SHOW_RESULTS,
            "OCR processing completed successfully for 3 files",
        ),
        (
            3,
            1,
            _BatchCompletionState.PARTIAL,
            _BatchCompletionAction.REVIEW_PARTIAL,
            "Saved 1; 2 failed",
        ),
        (
            3,
            0,
            _BatchCompletionState.FAILURE,
            _BatchCompletionAction.RETRY_FAILED,
            "Saved 0; 3 failed",
        ),
    ),
)
def test_batch_completion_matrix(total, successful, state, action, message):
    with (
        patch("bigocrpdf.processing_controller._", side_effect=lambda text: text),
        patch(
            "bigocrpdf.processing_controller.ngettext",
            side_effect=lambda singular, plural, count: singular if count == 1 else plural,
        ),
    ):
        outcome = _build_batch_completion(total, successful)

    assert outcome.state is state
    assert outcome.action is action
    assert outcome.message == message
    assert outcome.successful_files == successful
    assert outcome.failed_files == total - successful


@pytest.mark.parametrize(("total", "successful"), ((0, 0), (2, -1), (2, 3)))
def test_batch_completion_rejects_unverifiable_counts(total, successful):
    with pytest.raises(ValueError):
        _build_batch_completion(total, successful)


def test_processor_cleanup_preserves_worker_ownership():
    processor = MagicMock()
    controller = SimpleNamespace(
        settings=MagicMock(),
        ocr_processor=processor,
    )

    with patch("bigocrpdf.processing_controller.OcrProcessor") as processor_type:
        ProcessingController._cleanup_ocr_processor(controller)

    processor.force_cleanup.assert_called_once_with()
    processor_type.assert_not_called()
    assert controller.ocr_processor is processor


def test_controller_cleanup_stops_without_replacing_processor():
    processor = MagicMock()
    controller = SimpleNamespace(
        _closed=False,
        ocr_processor=processor,
        _clear_conclusion_timer=MagicMock(),
        _cleanup_ocr_processor=MagicMock(side_effect=processor.force_cleanup),
    )

    with patch("bigocrpdf.processing_controller.OcrProcessor") as processor_type:
        ProcessingController.cleanup(controller)

    assert controller._closed is True
    controller._clear_conclusion_timer.assert_called_once_with()
    processor.force_cleanup.assert_called_once_with()
    processor_type.assert_not_called()


def test_cancelled_state_is_reset_again_after_worker_becomes_idle():
    idle_callbacks = []
    processor = MagicMock()
    processor.run_when_idle.side_effect = idle_callbacks.append
    settings = MagicMock()
    ui = SimpleNamespace(
        terminal_page_manager=SimpleNamespace(stop_progress_monitor=MagicMock()),
        custom_header_bar=SimpleNamespace(start_button=MagicMock()),
        update_file_info=MagicMock(),
    )
    controller = SimpleNamespace(
        conclusion_timer_id=None,
        ocr_processor=processor,
        settings=settings,
        nav_manager=SimpleNamespace(navigate_to_settings=MagicMock()),
        ocr_dependency=SimpleNamespace(is_available=True),
        ui=ui,
        _closed=False,
        _clear_conclusion_timer=MagicMock(),
        _cleanup_ocr_processor=MagicMock(side_effect=processor.force_cleanup),
        _schedule_processing_state_reset=MagicMock(),
        _finalize_processing_state_reset=MagicMock(return_value=False),
    )

    ProcessingController.reset_to_settings(controller)
    idle_callbacks[0]()

    settings.reset_processing_state.assert_called_once_with(full=False)
    controller._clear_conclusion_timer.assert_called_once_with()
    controller._schedule_processing_state_reset.assert_called_once_with()


def test_failed_file_callback_keeps_input_in_queue_and_results_clean():
    window = SimpleNamespace(
        _closed=False,
        ocr_processor=SimpleNamespace(remove_processed_file=MagicMock()),
        settings=SimpleNamespace(processed_files=[]),
        ui=SimpleNamespace(),
    )

    with patch(
        "bigocrpdf.processing_controller.GLib.idle_add", side_effect=lambda callback: callback()
    ):
        ProcessingController._on_file_processed(
            window,
            "failed.pdf",
            "",
            "error: OCR dependency unavailable",
            [],
        )

    assert window.settings.processed_files == []
    window.ocr_processor.remove_processed_file.assert_not_called()


def test_success_releases_generated_input_after_all_consumers_finish():
    events = []
    comparison = SimpleNamespace(
        input_size_mb=1.0,
        output_size_mb=0.5,
        size_change_percent=-50.0,
    )
    processor = SimpleNamespace(
        remove_processed_file=lambda _path: events.append("remove"),
        get_completed_input_count=MagicMock(return_value=1),
        get_total_count=MagicMock(return_value=1),
    )
    settings = SimpleNamespace(
        processed_files=[],
        extracted_text={},
        save_txt=False,
        save_odf=False,
        ocr_boxes={},
        comparison_results=[],
        display_name=lambda _path: events.append("display-name") or "source.pdf",
    )
    ui = SimpleNamespace(
        terminal_page_manager=SimpleNamespace(
            update_processing_status=lambda _path: events.append("terminal")
        )
    )
    window = SimpleNamespace(
        _closed=False,
        ocr_processor=processor,
        settings=settings,
        ui=ui,
    )

    with (
        patch(
            "bigocrpdf.processing_controller.GLib.idle_add",
            side_effect=lambda callback: callback(),
        ),
        patch(
            "bigocrpdf.processing_controller.compare_pdfs",
            side_effect=lambda **_kwargs: events.append("compare") or comparison,
        ),
    ):
        ProcessingController._on_file_processed(
            window,
            "generated-input.pdf",
            "published-output.pdf",
            "text",
            [],
        )

    assert events == ["compare", "display-name", "terminal", "remove"]


def _completion_window(total: int, successful: int):
    output_files = [f"output-{index}.pdf" for index in range(successful)]
    terminal_page_manager = SimpleNamespace(
        show_completion_ui=MagicMock(),
        update_terminal_progress=MagicMock(),
        stop_progress_monitor=MagicMock(),
    )
    conclusion_page_manager = SimpleNamespace(update_conclusion_page=MagicMock())
    ui = SimpleNamespace(
        main_stack=SimpleNamespace(get_visible_child_name=MagicMock(return_value="terminal")),
        terminal_page_manager=terminal_page_manager,
        conclusion_page_manager=conclusion_page_manager,
        show_toast=MagicMock(),
    )
    settings = SimpleNamespace(
        processed_files=output_files,
        cleanup_temp_files=MagicMock(),
    )
    return SimpleNamespace(
        _closed=False,
        ocr_processor=SimpleNamespace(
            get_total_count=MagicMock(return_value=total),
            get_successful_input_count=MagicMock(return_value=successful),
        ),
        ui=ui,
        settings=settings,
        conclusion_timer_id=None,
        _announce_status=MagicMock(),
        _schedule_conclusion_page=MagicMock(),
        _present_batch_completion=MagicMock(),
    )


def test_scheduled_conclusion_replaces_previous_one_shot_timer():
    controller = SimpleNamespace(
        conclusion_timer_id=7,
        _clear_conclusion_timer=MagicMock(),
        _show_scheduled_conclusion_page=MagicMock(),
    )

    with patch("bigocrpdf.processing_controller.GLib.timeout_add", return_value=11) as timeout:
        ProcessingController._schedule_conclusion_page(controller)

    controller._clear_conclusion_timer.assert_called_once_with()
    timeout.assert_called_once_with(2000, controller._show_scheduled_conclusion_page)
    assert controller.conclusion_timer_id == 11


def test_scheduled_conclusion_clears_timer_and_only_opens_from_terminal():
    controller = SimpleNamespace(
        _closed=False,
        conclusion_timer_id=11,
        ui=SimpleNamespace(
            main_stack=SimpleNamespace(get_visible_child_name=MagicMock(return_value="terminal"))
        ),
        nav_manager=SimpleNamespace(navigate_to_conclusion=MagicMock()),
    )

    assert ProcessingController._show_scheduled_conclusion_page(controller) is False

    assert controller.conclusion_timer_id is None
    controller.nav_manager.navigate_to_conclusion.assert_called_once_with()

    controller.conclusion_timer_id = 12
    controller.ui.main_stack.get_visible_child_name.return_value = "main_view"
    assert ProcessingController._show_scheduled_conclusion_page(controller) is False
    controller.nav_manager.navigate_to_conclusion.assert_called_once_with()


def test_closed_controller_drops_deferred_ui_reset():
    controller = SimpleNamespace(
        _closed=True,
        _finalize_processing_state_reset=MagicMock(),
    )

    with patch("bigocrpdf.processing_controller.GLib.idle_add") as idle_add:
        ProcessingController._schedule_processing_state_reset(controller)

    idle_add.assert_not_called()


@pytest.mark.parametrize(
    ("successful", "expected_state"),
    (
        (3, _BatchCompletionState.SUCCESS),
        (1, _BatchCompletionState.PARTIAL),
        (0, _BatchCompletionState.FAILURE),
    ),
)
def test_processing_completion_routes_each_truthful_state(successful, expected_state):
    window = _completion_window(total=3, successful=successful)

    with (
        patch("bigocrpdf.processing_controller._", side_effect=lambda text: text),
        patch(
            "bigocrpdf.processing_controller.ngettext",
            side_effect=lambda singular, plural, count: singular if count == 1 else plural,
        ),
        patch(
            "bigocrpdf.processing_controller.GLib.idle_add",
            side_effect=lambda callback: callback(),
        ),
    ):
        ProcessingController._on_processing_complete(window)

    if successful == 3:
        window.ui.show_toast.assert_called_once_with(
            "OCR processing completed successfully for 3 files"
        )
        window._schedule_conclusion_page.assert_called_once_with()
        window._present_batch_completion.assert_not_called()
    else:
        window.ui.show_toast.assert_not_called()
        outcome = window._present_batch_completion.call_args.args[0]
        assert outcome.state is expected_state

    terminal_page = window.ui.terminal_page_manager
    terminal_page.show_completion_ui.assert_not_called()
    terminal_page.update_terminal_progress.assert_called_once_with(1.0, "100%")
    terminal_page.stop_progress_monitor.assert_called_once_with()

    if successful:
        window.ui.conclusion_page_manager.update_conclusion_page.assert_called_once_with()
    else:
        window.ui.conclusion_page_manager.update_conclusion_page.assert_not_called()
    window.settings.cleanup_temp_files.assert_not_called()


@pytest.mark.parametrize(
    ("successful", "responses", "default_response"),
    (
        (1, [call("back", "Back"), call("results", "Results")], "results"),
        (0, [call("back", "Back")], "back"),
    ),
)
def test_non_success_dialog_has_coherent_title_message_and_actions(
    successful, responses, default_response
):
    dialog = MagicMock()
    parent = MagicMock()
    controller = SimpleNamespace(
        parent=parent,
        _on_batch_completion_response=MagicMock(),
    )
    with (
        patch("bigocrpdf.processing_controller._", side_effect=lambda text: text),
        patch("bigocrpdf.processing_controller.Adw.AlertDialog", return_value=dialog) as alert,
    ):
        outcome = _build_batch_completion(3, successful)
        ProcessingController._present_batch_completion(controller, outcome)

    alert.assert_called_once_with(heading=outcome.title, body=outcome.message)
    assert dialog.add_response.call_args_list == responses
    dialog.set_default_response.assert_called_once_with(default_response)
    dialog.set_close_response.assert_called_once_with("back")
    dialog.connect.assert_called_once_with("response", controller._on_batch_completion_response)
    dialog.present.assert_called_once_with(parent)


@pytest.mark.parametrize(("response", "schedules_results"), (("results", True), ("back", False)))
def test_batch_completion_response_matches_visible_action(response, schedules_results):
    controller = SimpleNamespace(
        _schedule_conclusion_page=MagicMock(),
        reset_to_settings=MagicMock(),
    )

    ProcessingController._on_batch_completion_response(controller, MagicMock(), response)

    assert controller._schedule_conclusion_page.called is schedules_results
    assert controller.reset_to_settings.called is not schedules_results


def test_result_postprocessing_finishes_before_gtk_publication():
    idle_callbacks = []
    events = []
    comparison = SimpleNamespace(
        input_size_mb=1.0,
        output_size_mb=0.5,
        size_change_percent=-50.0,
    )
    settings = SimpleNamespace(
        processed_files=["published-output.pdf"],
        extracted_text={},
        save_txt=True,
        txt_folder="/txt",
        separate_txt_folder=True,
        save_odf=True,
        odf_include_images=True,
        ocr_boxes={},
        comparison_results=[],
        display_name=lambda _path: "source.pdf",
    )
    processor = SimpleNamespace(
        remove_processed_file=MagicMock(),
        get_completed_input_count=MagicMock(return_value=1),
        get_total_count=MagicMock(return_value=1),
    )
    terminal = SimpleNamespace(update_processing_status=MagicMock())
    controller = SimpleNamespace(
        _closed=False,
        ocr_processor=processor,
        settings=settings,
        ui=SimpleNamespace(terminal_page_manager=terminal),
    )

    with (
        patch(
            "bigocrpdf.processing_controller.save_text_file",
            side_effect=lambda *_args: events.append("txt"),
        ),
        patch(
            "bigocrpdf.processing_controller.save_odf_file",
            side_effect=lambda *_args, **_kwargs: events.append("odf"),
        ),
        patch(
            "bigocrpdf.processing_controller.compare_pdfs",
            side_effect=lambda **_kwargs: events.append("compare") or comparison,
        ),
        patch(
            "bigocrpdf.processing_controller.GLib.idle_add",
            side_effect=idle_callbacks.append,
        ),
    ):
        ProcessingController._on_file_processed(
            controller,
            "generated-input.pdf",
            "published-output.pdf",
            "text",
            ["box"],
        )

    assert events == ["txt", "odf", "compare"]
    assert settings.processed_files == ["published-output.pdf"]
    assert settings.extracted_text == {}
    assert settings.ocr_boxes == {}
    assert settings.comparison_results == []
    terminal.update_processing_status.assert_not_called()
    processor.remove_processed_file.assert_not_called()

    assert idle_callbacks[0]() is False

    assert settings.extracted_text == {"published-output.pdf": "text"}
    assert settings.ocr_boxes == {"published-output.pdf": ["box"]}
    assert settings.comparison_results == [comparison]
    terminal.update_processing_status.assert_called_once_with("generated-input.pdf")
    processor.remove_processed_file.assert_called_once_with("generated-input.pdf")


def test_closed_controller_skips_result_postprocessing():
    controller = SimpleNamespace(_closed=True)

    with (
        patch("bigocrpdf.processing_controller.save_text_file") as save_text,
        patch("bigocrpdf.processing_controller.save_odf_file") as save_odf,
        patch("bigocrpdf.processing_controller.compare_pdfs") as compare,
        patch("bigocrpdf.processing_controller.GLib.idle_add") as idle_add,
    ):
        ProcessingController._on_file_processed(
            controller,
            "input.pdf",
            "output.pdf",
            "text",
            [],
        )

    save_text.assert_not_called()
    save_odf.assert_not_called()
    compare.assert_not_called()
    idle_add.assert_not_called()
