"""Pluralization contracts for translated user-visible counts."""

from __future__ import annotations

import ast
from pathlib import Path
from string import Formatter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bigocrpdf import OcrDependencyState
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.ui.terminal_page import TerminalPageManager


def _english_plural(singular: str, plural: str, count: int) -> str:
    return singular if count == 1 else plural


def test_ngettext_calls_use_matching_named_placeholders_and_format() -> None:
    source_root = Path(__file__).parents[1] / "src" / "bigocrpdf"
    formatter = Formatter()

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }

        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "ngettext"
            ):
                continue

            assert len(call.args) == 3, f"{source_path}:{call.lineno}"
            singular = ast.literal_eval(call.args[0])
            plural = ast.literal_eval(call.args[1])
            singular_fields = {field for _, field, _, _ in formatter.parse(singular) if field}
            plural_fields = {field for _, field, _, _ in formatter.parse(plural) if field}

            assert singular_fields == plural_fields, f"{source_path}:{call.lineno}"
            assert all(not field.isdigit() for field in singular_fields), (
                f"{source_path}:{call.lineno} uses positional placeholders"
            )

            if singular_fields:
                attribute = parents[call]
                format_call = parents[attribute]
                assert isinstance(attribute, ast.Attribute) and attribute.attr == "format", (
                    f"{source_path}:{call.lineno} must format the selected translation"
                )
                assert isinstance(format_call, ast.Call)


@pytest.mark.parametrize(("count", "expected"), ((1, "1 file"), (3, "3 files")))
def test_header_queue_count_uses_singular_and_plural(count: int, expected: str) -> None:
    header = SimpleNamespace(
        window=SimpleNamespace(ocr_dependency=OcrDependencyState(is_available=True)),
        queue_size_label=MagicMock(),
        clear_queue_button=MagicMock(),
        view_toggle_button=MagicMock(),
        start_button=MagicMock(),
    )
    header._apply_ocr_availability_to_button = lambda button: (
        HeaderBar._apply_ocr_availability_to_button(header, button)
    )

    with patch("bigocrpdf.ui.header_bar.ngettext", side_effect=_english_plural):
        HeaderBar.update_queue_size(header, count)

    header.queue_size_label.set_text.assert_called_once_with(expected)


@pytest.mark.parametrize(
    ("count", "expected_markup", "expected_announcement"),
    (
        (
            1,
            "<b>OCR processing complete!</b> 1 file processed · Total time: 2s",
            "OCR processing complete. 1 file processed.",
        ),
        (
            2,
            "<b>OCR processing complete!</b> 2 files processed · Total time: 2s",
            "OCR processing complete. 2 files processed.",
        ),
    ),
)
def test_terminal_completion_uses_matching_visual_and_a11y_plural(
    count: int,
    expected_markup: str,
    expected_announcement: str,
) -> None:
    manager = SimpleNamespace(
        _progress_state=SimpleNamespace(update_status=lambda _text: True),
        terminal_status_bar=MagicMock(),
        window=SimpleNamespace(announce_status=MagicMock()),
        stop_progress_monitor=MagicMock(),
    )

    with patch("bigocrpdf.ui.terminal_page.ngettext", side_effect=_english_plural):
        TerminalPageManager._show_completion_status(manager, count, "2s")

    manager.terminal_status_bar.set_markup.assert_called_once_with(expected_markup)
    manager.window.announce_status.assert_called_once_with(expected_announcement)


def test_header_queue_update_restores_the_idle_button_label() -> None:
    """Publishing a queue count is what returns the button to its resting state.

    It already restored visibility and sensitivity but not the label, so a
    button hidden while it still read "Starting…" came back with that text the
    moment files were added again -- which is exactly what a user sees after a
    finished batch.
    """
    header = SimpleNamespace(
        window=SimpleNamespace(ocr_dependency=OcrDependencyState(is_available=True)),
        queue_size_label=MagicMock(),
        clear_queue_button=MagicMock(),
        view_toggle_button=MagicMock(),
        start_button=MagicMock(),
    )
    header._apply_ocr_availability_to_button = lambda button: (
        HeaderBar._apply_ocr_availability_to_button(header, button)
    )

    HeaderBar.update_queue_size(header, 0)

    header.start_button.set_label.assert_called_once_with("Start OCR")
