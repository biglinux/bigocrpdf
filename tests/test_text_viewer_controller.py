"""Contracts for the composed extracted-text viewer and dialog manager."""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bigocrpdf.ui.dialogs_manager import DialogsManager
from bigocrpdf.ui.text_viewer_controller import Gtk as ViewerGtk
from bigocrpdf.ui.text_viewer_controller import TextViewerController, TextViewerState


def _controller(settings=None) -> TextViewerController:
    if settings is None:
        settings = SimpleNamespace(extracted_text={})
    return TextViewerController(MagicMock(), settings, MagicMock(), MagicMock())


def test_cached_extracted_text_is_used_before_file_fallbacks(tmp_path) -> None:
    file_path = str(tmp_path / "cached.pdf")
    (tmp_path / "cached.pdf").write_bytes(b"%PDF")
    settings = SimpleNamespace(extracted_text={file_path: "cached text"})
    controller = _controller(settings)

    with patch("bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text") as convert:
        assert controller._get_extracted_text_for_file(file_path) == "cached text"

    convert.assert_not_called()


def test_structured_text_is_cached_after_sidecar_fallbacks(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    settings = SimpleNamespace(extracted_text={})
    controller = _controller(settings)

    with (
        patch("bigocrpdf.ui.text_viewer_controller.read_text_from_sidecar", return_value=None),
        patch(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
            return_value="structured text",
        ),
    ):
        assert controller._get_extracted_text_for_file(str(pdf_path)) == "structured text"

    assert settings.extracted_text[str(pdf_path)] == "structured text"


def test_whitespace_sidecars_do_not_hide_structured_text(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    settings = SimpleNamespace(extracted_text={})
    controller = _controller(settings)

    with (
        patch(
            "bigocrpdf.ui.text_viewer_controller.read_text_from_sidecar",
            side_effect=["  \n", "\t"],
        ),
        patch(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
            return_value="structured text",
        ),
    ):
        assert controller._get_extracted_text_for_file(str(pdf_path)) == "structured text"

    assert settings.extracted_text[str(pdf_path)] == "structured text"


def test_expected_structured_extraction_failure_uses_uncached_fallback(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    settings = SimpleNamespace(extracted_text={})
    controller = _controller(settings)

    with (
        patch("bigocrpdf.ui.text_viewer_controller.read_text_from_sidecar", return_value=None),
        patch(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
            side_effect=subprocess.CalledProcessError(1, ["pdftotext"]),
        ),
    ):
        result = controller._get_extracted_text_for_file(str(pdf_path))

    assert "could not be found" in result
    assert str(pdf_path) not in settings.extracted_text


def test_programming_error_from_structured_extractor_is_not_hidden(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF")
    controller = _controller()

    with (
        patch("bigocrpdf.ui.text_viewer_controller.read_text_from_sidecar", return_value=None),
        patch(
            "bigocrpdf.utils.tsv_odf_converter.convert_pdf_to_text",
            side_effect=TypeError("invalid parser state"),
        ),
        pytest.raises(TypeError, match="invalid parser state"),
    ):
        controller._get_extracted_text_for_file(str(pdf_path))


def test_unicode_search_uses_gtk_match_offsets() -> None:
    buffer = ViewerGtk.TextBuffer()
    buffer.set_text("Straße STRASSE")
    buffer.create_tag("search_highlight")
    buffer.create_tag("current_match")
    text_view = MagicMock(spec=ViewerGtk.TextView)
    text_view.get_buffer.return_value = buffer
    state = TextViewerState()

    count = TextViewerController._highlight_text_matches(text_view, state, "straße")

    assert count == 2
    assert state.matches == [(0, 6), (7, 14)]


def test_pending_search_source_is_removed_when_viewer_closes() -> None:
    state = TextViewerState(debounce_id=42)

    with patch("bigocrpdf.ui.text_viewer_controller.GLib.source_remove") as source_remove:
        assert TextViewerController._clear_pending_search(state) is False

    source_remove.assert_called_once_with(42)
    assert state.debounce_id == 0


def test_font_zoom_reuses_one_css_provider() -> None:
    provider = MagicMock()
    state = TextViewerState(font_provider=provider)
    zoom_out = MagicMock()
    zoom_in = MagicMock()

    TextViewerController._apply_text_viewer_font_size(zoom_out, zoom_in, state, 14)
    TextViewerController._apply_text_viewer_font_size(zoom_out, zoom_in, state, 16)

    assert provider.load_from_string.call_count == 2
    assert state.font_size == 16


def test_copy_text_uses_text_clipboard_api() -> None:
    controller = _controller()
    clipboard = MagicMock()

    with patch(
        "bigocrpdf.ui.text_viewer_controller.get_default_clipboard",
        return_value=clipboard,
    ):
        controller._copy_text_to_clipboard("OCR text")

    clipboard.set_text.assert_called_once_with("OCR text")


def test_dialogs_manager_passes_parent_to_text_viewer() -> None:
    parent = MagicMock()
    settings = MagicMock()
    toast = MagicMock()

    with (
        patch("bigocrpdf.ui.dialogs_manager.FileSaveController") as file_save,
        patch("bigocrpdf.ui.dialogs_manager.TextViewerController") as text_viewer,
        patch("bigocrpdf.ui.dialogs_manager.PDFOptionsController"),
    ):
        DialogsManager(parent, settings, toast)

    text_viewer.assert_called_once_with(parent, settings, toast, file_save.return_value)


def test_dialogs_manager_delegates_text_viewer_contract() -> None:
    manager = DialogsManager(MagicMock(), MagicMock(), MagicMock())
    manager._text_viewer = MagicMock()

    manager.show_extracted_text("processed.pdf")

    manager._text_viewer.show_extracted_text.assert_called_once_with("processed.pdf")


def test_empty_image_merge_request_completes_without_opening_dialog() -> None:
    manager = DialogsManager(MagicMock(), MagicMock(), MagicMock())
    complete = MagicMock()

    with patch("bigocrpdf.ui.dialogs_manager.Adw.AlertDialog") as alert_dialog:
        manager.show_image_merge_dialog([], heading="Heading", body="Body", on_complete=complete)

    alert_dialog.assert_not_called()
    complete.assert_called_once_with()


def test_merge_failure_reports_error_and_completes() -> None:
    manager = DialogsManager(MagicMock(), MagicMock(), MagicMock())
    manager._convert_and_queue_images = MagicMock(return_value=False)
    manager._show_toast = MagicMock()
    complete = MagicMock()
    dialog = MagicMock()

    with patch("bigocrpdf.ui.dialogs_manager.Adw.AlertDialog", return_value=dialog):
        manager.show_image_merge_dialog(
            ["one.png", "two.png"],
            heading="Heading",
            body="Body",
            on_complete=complete,
        )

    response_callback = dialog.connect.call_args.args[1]
    response_callback(dialog, "merge")

    manager._show_toast.assert_called_once_with("Error merging images")
    complete.assert_called_once_with()
