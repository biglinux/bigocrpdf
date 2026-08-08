import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin


def test_unique_path_never_falls_back_to_overwrite() -> None:
    real_descriptor = __import__("os").open("/dev/null", __import__("os").O_RDONLY)
    failures = [FileExistsError()] * 1000 + [real_descriptor]

    with patch("bigocrpdf.ui.conclusion_export_mixin.os.open", side_effect=failures):
        result = ConclusionExportMixin._reserve_unique_path("document.md")

    assert result == "document (1000).md"


def test_unknown_export_format_is_rejected() -> None:
    owner: Any = ConclusionExportMixin.__new__(ConclusionExportMixin)
    with pytest.raises(ValueError, match="unsupported export format"):
        owner._bulk_convert_one(
            "input.pdf",
            "output.txt",
            "txt",
            {},
            threading.Event(),
        )


def test_cancelled_bulk_export_reports_original_total() -> None:
    toast = MagicMock()
    owner: Any = ConclusionExportMixin.__new__(ConclusionExportMixin)
    owner.window = SimpleNamespace(ui=SimpleNamespace(show_toast=toast))
    dialog = MagicMock()

    result = owner._on_bulk_export_finished(
        dialog,
        saved=1,
        failed=0,
        cancelled=True,
        total=4,
        dest_folder="/tmp/exports",
    )

    assert result is False
    dialog.force_close.assert_called_once_with()
    toast.assert_called_once_with("Cancelled — saved 1 of 4")


def test_bulk_worker_marks_preexisting_cancellation() -> None:
    idle_calls = []
    cancel_event = threading.Event()
    cancel_event.set()
    owner: Any = ConclusionExportMixin.__new__(ConclusionExportMixin)
    owner._on_bulk_export_finished = MagicMock()

    with patch(
        "bigocrpdf.ui.conclusion_export_mixin.GLib.idle_add",
        side_effect=lambda *args: idle_calls.append(args),
    ):
        owner._bulk_export_worker(
            ["one.pdf", "two.pdf"],
            "/tmp",
            "md",
            {},
            cancel_event,
            MagicMock(),
            MagicMock(),
        )

    assert len(idle_calls) == 1
    assert idle_calls[0][2:6] == (0, 0, True, 2)


def test_single_export_failure_preserves_existing_target(tmp_path) -> None:
    target = tmp_path / "existing.md"
    target.write_text("KEEP", encoding="utf-8")
    owner: Any = ConclusionExportMixin.__new__(ConclusionExportMixin)
    owner._build_progress_dialog = MagicMock(
        return_value=(MagicMock(), MagicMock(), threading.Event())
    )
    owner._on_single_export_finished = MagicMock(return_value=False)

    class ImmediateThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self) -> None:
            self.target()

    def fail(_event: threading.Event) -> None:
        raise RuntimeError("conversion failed")

    with (
        patch("bigocrpdf.ui.conclusion_export_mixin.threading.Thread", ImmediateThread),
        patch(
            "bigocrpdf.ui.conclusion_export_mixin.GLib.idle_add",
            side_effect=lambda callback, *args: callback(*args),
        ),
    ):
        owner._run_single_export(
            str(target),
            "input.pdf",
            "Exporting…",
            "Markdown",
            False,
            fail,
        )

    assert target.read_text(encoding="utf-8") == "KEEP"
