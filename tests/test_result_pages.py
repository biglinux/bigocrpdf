from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

from bigocrpdf.ui.conclusion_page import ConclusionPageManager
from bigocrpdf.ui.conclusion_ui_mixin import ConclusionStatsFileListMixin
from bigocrpdf.ui.terminal_page import TerminalPageManager
from bigocrpdf.utils.progress_state import ProgressState


def test_terminal_progress_clamps_fraction() -> None:
    progress_bar = MagicMock()
    manager: Any = SimpleNamespace(
        terminal_progress_bar=progress_bar,
        _progress_state=ProgressState(),
    )

    TerminalPageManager.update_terminal_progress(manager, 1.5)
    TerminalPageManager.update_terminal_progress(manager, -0.5)

    assert progress_bar.set_fraction.call_args_list == [call(1.0), call(0.0)]


def test_terminal_plain_statuses_do_not_enable_markup() -> None:
    status_bar = MagicMock()
    manager: Any = SimpleNamespace(
        terminal_status_bar=status_bar,
        _progress_state=ProgressState(),
    )

    TerminalPageManager._show_simple_progress_status(manager, 1, 2, "3s")
    TerminalPageManager._show_initial_status(manager, 2, "3s")

    assert status_bar.set_text.call_count == 2
    status_bar.set_markup.assert_not_called()


def test_terminal_timer_releases_source_when_page_changes() -> None:
    manager: Any = SimpleNamespace(
        progress_timer_id=42,
        window=SimpleNamespace(
            ui=SimpleNamespace(
                main_stack=SimpleNamespace(
                    get_visible_child_name=MagicMock(return_value="main_view")
                )
            )
        ),
    )

    assert TerminalPageManager._update_ocr_progress(manager) is False
    assert manager.progress_timer_id is None


def test_terminal_timer_stops_after_completion() -> None:
    processor = SimpleNamespace(
        get_progress=MagicMock(return_value=1.0),
        get_completed_input_count=MagicMock(return_value=2),
        get_total_count=MagicMock(return_value=2),
        get_current_file_info=MagicMock(return_value={}),
        is_processing=MagicMock(return_value=False),
    )
    manager: Any = SimpleNamespace(
        progress_timer_id=42,
        terminal_progress_bar=MagicMock(),
        terminal_status_bar=MagicMock(),
        _progress_state=ProgressState(),
        window=SimpleNamespace(
            processing=SimpleNamespace(
                ocr_processor=processor,
                process_start_time=None,
            ),
            ui=SimpleNamespace(
                main_stack=SimpleNamespace(
                    get_visible_child_name=MagicMock(return_value="terminal")
                )
            ),
            announce_status=MagicMock(),
        ),
    )
    manager._update_progress_bar_incremental = (
        TerminalPageManager._update_progress_bar_incremental.__get__(manager)
    )
    manager._update_status_text_incremental = (
        TerminalPageManager._update_status_text_incremental.__get__(manager)
    )
    manager._show_completion_status = TerminalPageManager._show_completion_status.__get__(manager)

    assert TerminalPageManager._update_ocr_progress(manager) is False
    assert manager.progress_timer_id is None
    manager.terminal_status_bar.set_markup.assert_called_once()


def test_conclusion_reset_clears_selection_state() -> None:
    labels = [MagicMock() for _ in range(5)]
    manager: Any = SimpleNamespace(
        result_file_count=labels[0],
        result_page_count=labels[1],
        result_time=labels[2],
        result_file_size=labels[3],
        result_size_change=labels[4],
        _selection_mode=True,
        _selected_files={"old.pdf"},
        _selection_toggle_btn=MagicMock(),
        _selection_action_bar=MagicMock(),
        output_list_box=MagicMock(),
        _refresh_selection_ui=MagicMock(),
        _clear_output_list=MagicMock(),
    )

    ConclusionPageManager.reset_page(manager)

    assert manager._selection_mode is False
    assert manager._selected_files == set()
    manager._selection_toggle_btn.set_active.assert_called_once_with(False)
    manager._selection_action_bar.set_visible.assert_called_once_with(False)
    manager._refresh_selection_ui.assert_called_once_with()
    manager._clear_output_list.assert_called_once_with()


def test_conclusion_reuses_collected_file_statistics() -> None:
    manager: Any = SimpleNamespace(
        window=SimpleNamespace(
            processing=SimpleNamespace(
                ocr_processor=SimpleNamespace(get_successful_input_count=MagicMock(return_value=2))
            )
        ),
        _output_files=[
            ("one.pdf", 3, 1024, None),
            ("two.pdf", 4, 3072, None),
        ],
        result_file_count=MagicMock(),
        result_page_count=MagicMock(),
        result_file_size=MagicMock(),
        _update_processing_time=MagicMock(),
        _update_size_change=MagicMock(),
    )

    ConclusionStatsFileListMixin._update_statistics(manager)

    manager.result_page_count.set_text.assert_called_once_with("7")
    manager._update_size_change.assert_called_once_with(4096)


def test_conclusion_collects_each_output_once(tmp_path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"a" * 10)
    second.write_bytes(b"b" * 20)
    manager: Any = SimpleNamespace(
        window=SimpleNamespace(
            settings=SimpleNamespace(
                processed_files=[str(first), str(second)],
                comparison_results=[],
            )
        ),
        _output_files=[],
        _update_statistics=MagicMock(),
        _update_file_list=MagicMock(),
    )

    with patch(
        "bigocrpdf.ui.conclusion_ui_mixin.get_pdf_page_count",
        side_effect=[2, 5],
    ) as page_count:
        ConclusionStatsFileListMixin.update_conclusion_page(manager)

    assert manager._output_files == [
        (str(first), 2, 10, None),
        (str(second), 5, 20, None),
    ]
    assert page_count.call_count == 2
    manager._update_statistics.assert_called_once_with()
    manager._update_file_list.assert_called_once_with()


def test_conclusion_file_list_drops_stale_selection() -> None:
    row = MagicMock()
    manager: Any = SimpleNamespace(
        _output_files=[("current.pdf", 1, 100, None)],
        _selected_files={"current.pdf", "deleted.pdf"},
        _clear_output_list=MagicMock(),
        _create_file_row=MagicMock(return_value=row),
        output_list_box=MagicMock(),
        _refresh_selection_ui=MagicMock(),
    )

    ConclusionStatsFileListMixin._update_file_list(manager)

    assert manager._selected_files == {"current.pdf"}
    manager.output_list_box.append.assert_called_once_with(row)
    manager._refresh_selection_ui.assert_called_once_with()


def test_conclusion_size_change_clears_stale_css_without_results() -> None:
    label = MagicMock()
    manager: Any = SimpleNamespace(
        result_size_change=label,
        window=SimpleNamespace(settings=SimpleNamespace(comparison_results=[])),
    )

    ConclusionStatsFileListMixin._update_size_change(manager, 100)

    label.remove_css_class.assert_has_calls([call("success"), call("warning")])
    label.set_text.assert_called_once_with("--")


def test_conclusion_keeps_existing_output_after_long_processing(tmp_path) -> None:
    output = tmp_path / "result.pdf"
    output.write_bytes(b"PDF")
    manager: Any = SimpleNamespace(
        window=SimpleNamespace(
            settings=SimpleNamespace(
                processed_files=[str(output)],
                comparison_results=[],
            )
        ),
        _output_files=[],
        _update_statistics=MagicMock(),
        _update_file_list=MagicMock(),
    )

    with patch("bigocrpdf.ui.conclusion_ui_mixin.get_pdf_page_count", return_value=1):
        ConclusionStatsFileListMixin.update_conclusion_page(manager)

    assert manager._output_files == [(str(output), 1, 3, None)]
