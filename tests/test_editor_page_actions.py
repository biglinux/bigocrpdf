"""Regression tests for editor actions that own undo snapshots."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import gi

gi.require_version("Gdk", "4.0")
from gi.repository import Gdk

from bigocrpdf.ui.pdf_editor.editor_page_actions_mixin import EditorPageActionsMixin
from bigocrpdf.ui.pdf_editor.editor_tools_controller import EditorToolsController
from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail


def _owner(document: PDFDocument, *, selected: set[int] | None = None) -> Any:
    thumbnails = [SimpleNamespace(page_state=page) for page in document.pages]
    return cast(
        Any,
        SimpleNamespace(
            _document=document,
            _page_grid=SimpleNamespace(
                thumbnails=thumbnails,
                selected_indices=selected or set(),
                refresh=MagicMock(),
                swap_pages_in_grid=MagicMock(),
            ),
            _push_undo=MagicMock(),
            _show_error=MagicMock(),
            _update_status_bar=MagicMock(),
        ),
    )


def test_rotate_and_flip_skip_invalid_or_deleted_targets() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    document.pages[0].deleted = True
    owner = _owner(document, selected={-1, 4})

    EditorPageActionsMixin._rotate_selected_pages(owner, 90)
    EditorPageActionsMixin._flip_selected_pages(owner)

    owner._push_undo.assert_not_called()
    assert document.modified is False
    assert document.pages[0].rotation == 0
    assert document.pages[0].flip_horizontal is False


def test_rotate_real_target_records_one_snapshot() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document, selected={0})

    EditorPageActionsMixin._rotate_selected_pages(owner, 90)

    owner._push_undo.assert_called_once_with()
    assert document.modified is True
    assert document.pages[0].rotation == 90


def test_add_files_records_snapshot_only_after_pages_are_discovered() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document)
    renderer = MagicMock()
    renderer.get_page_count.side_effect = [0, 2]

    with patch(
        "bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.get_thumbnail_renderer",
        return_value=renderer,
    ):
        added = EditorPageActionsMixin._add_files_to_document(owner, ["empty.pdf", "added.pdf"])

    assert added == 2
    owner._push_undo.assert_called_once_with()
    assert document.total_pages == 3
    assert [page.position for page in document.pages] == [0, 1, 2]
    assert [page.source_file for page in document.pages[1:]] == ["added.pdf", "added.pdf"]


def test_add_files_without_pages_does_not_record_snapshot() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document)
    renderer = MagicMock()
    renderer.get_page_count.return_value = 0

    with patch(
        "bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.get_thumbnail_renderer",
        return_value=renderer,
    ):
        added = EditorPageActionsMixin._add_files_to_document(owner, ["empty.pdf"])

    assert added == 0
    owner._push_undo.assert_not_called()
    assert document.modified is False
    assert document.total_pages == 1


def test_external_drop_reports_failure_when_no_pages_are_added() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document)
    renderer = MagicMock()
    renderer.get_page_count.return_value = 0

    with (
        patch(
            "bigocrpdf.ui.pdf_editor.editor_page_actions_mixin._valid_external_drop_paths",
            return_value=["empty.pdf"],
        ),
        patch(
            "bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.get_thumbnail_renderer",
            return_value=renderer,
        ),
    ):
        handled = EditorPageActionsMixin._on_external_file_drop(
            owner, MagicMock(), MagicMock(), 0, 0
        )

    assert handled is False
    owner._push_undo.assert_not_called()


def test_rejected_clipboard_texture_removes_owned_png() -> None:
    owner = _owner(PDFDocument(path="source.pdf", total_pages=1))
    clipboard = MagicMock()
    texture = MagicMock()
    texture.save_to_png_bytes.return_value.get_data.return_value = b"png"
    clipboard.read_texture_finish.return_value = texture
    owner._add_files_to_document = MagicMock(return_value=0)

    with (
        patch(
            "bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.mkstemp",
            return_value=(10, "/tmp/paste.png"),
        ),
        patch("bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.os.write"),
        patch("bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.os.close"),
        patch("bigocrpdf.ui.pdf_editor.editor_page_actions_mixin.remove_file") as remove_file,
    ):
        EditorPageActionsMixin._on_editor_clipboard_texture_ready(owner, clipboard, MagicMock())

    remove_file.assert_called_once_with("/tmp/paste.png")


def test_clipboard_uri_stream_closes_when_data_is_missing() -> None:
    owner = _owner(PDFDocument(path="source.pdf", total_pages=1))
    clipboard = MagicMock()
    stream = MagicMock()
    stream.read_bytes.return_value.get_data.return_value = None
    clipboard.read_finish.return_value = (stream, "text/uri-list")

    EditorPageActionsMixin._on_editor_clipboard_uris_ready(owner, clipboard, MagicMock())

    stream.close.assert_called_once_with(None)


def test_move_at_document_boundary_does_not_record_snapshot() -> None:
    document = PDFDocument(path="source.pdf", total_pages=2)
    owner = _owner(document, selected={0})

    EditorPageActionsMixin._move_selected_pages(owner, -1)

    owner._push_undo.assert_not_called()
    assert document.modified is False


def test_reverse_single_page_does_not_record_snapshot() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document)
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner

    controller.reverse(None, None)

    owner._push_undo.assert_not_called()
    assert document.modified is False


def test_reverse_preserves_unique_positions_for_excluded_pages() -> None:
    document = PDFDocument(path="source.pdf", total_pages=3)
    document.pages[1].deleted = True
    owner = _owner(document)
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner

    controller.reverse(None, None)

    owner._push_undo.assert_called_once_with()
    assert [page.position for page in document.pages] == [2, 1, 0]
    assert document.pages[1].deleted is True
    assert document.modified is True


def test_page_drop_inside_selected_block_does_not_record_snapshot() -> None:
    document = PDFDocument(path="source.pdf", total_pages=4)
    grid = PageGrid.__new__(PageGrid)
    grid._document = document
    grid._selected_indices = {1, 2}
    grid.on_before_mutate = MagicMock()

    changed = grid._move_selected_pages(3)

    assert changed is False
    grid.on_before_mutate.assert_not_called()
    assert [page.page_number for page in document.pages] == [1, 2, 3, 4]


def test_invalid_single_page_drop_does_not_record_snapshot() -> None:
    document = PDFDocument(path="source.pdf", total_pages=2)
    grid = PageGrid.__new__(PageGrid)
    grid._document = document
    grid.on_before_mutate = MagicMock()

    changed = grid._move_single_page(-1, 0)

    assert changed is False
    grid.on_before_mutate.assert_not_called()


def test_thumbnail_ocr_notification_skips_unchanged_state() -> None:
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._page_state = PageState(page_number=1)
    thumbnail.on_before_mutate = MagicMock()
    thumbnail.emit = MagicMock()
    thumbnail._update_appearance = MagicMock()
    check = MagicMock()
    check.get_active.return_value = True

    thumbnail._on_ocr_toggled(check)

    thumbnail.on_before_mutate.assert_not_called()
    thumbnail.emit.assert_not_called()


def test_delete_shortcut_delegates_single_snapshot_to_grid() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document, selected={0})
    owner._page_grid.toggle_ocr_for_selected = MagicMock(side_effect=owner._push_undo)

    handled = EditorPageActionsMixin._on_key_pressed(
        owner,
        MagicMock(),
        Gdk.KEY_Delete,
        0,
        Gdk.ModifierType(0),
    )

    assert handled is True
    owner._push_undo.assert_called_once_with()


def test_compress_save_copies_atomically_then_removes_owned_temp() -> None:
    owner = _owner(PDFDocument(path="source.pdf", total_pages=1))
    owner._show_info = MagicMock()
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner
    dialog = MagicMock()
    dialog.save_finish.return_value.get_path.return_value = "/output/compressed.pdf"

    with (
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_document_io.copy_pdf_with_ocr_invalidation"
        ) as copy_pdf,
        patch("bigocrpdf.utils.temp_manager.remove_tracked_file") as remove_file,
    ):
        controller._finish_compress_save(dialog, MagicMock(), "/tmp/compressed.pdf", "done")

    copy_pdf.assert_called_once_with(
        "/tmp/compressed.pdf",
        "/output/compressed.pdf",
        overwrite=True,
    )
    remove_file.assert_called_once_with("/tmp/compressed.pdf")
    owner._show_info.assert_called_once_with("done")


def test_compress_save_dialog_uses_editor_window_as_parent() -> None:
    owner = _owner(PDFDocument(path="source.pdf", total_pages=1))
    owner._pdf_path = "source.pdf"
    owner._default_save_dir = MagicMock(return_value="/output")
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner
    save_dialog = MagicMock()
    compression_result = SimpleNamespace(success=True, message="compressed")

    with (
        patch(
            "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
            return_value=True,
        ),
        patch(
            "bigocrpdf.services.pdf_operations.compress_pdf",
            return_value=compression_result,
        ),
        patch(
            "bigocrpdf.utils.temp_manager.mkstemp",
            side_effect=[
                (10, "/tmp/edited.pdf"),
                (11, "/tmp/compressed.pdf"),
            ],
        ),
        patch("bigocrpdf.ui.pdf_editor.editor_tools_controller.os.close"),
        patch("bigocrpdf.utils.temp_manager.remove_file"),
        patch(
            "bigocrpdf.ui.pdf_editor.editor_tools_controller.Gtk.FileDialog",
            return_value=save_dialog,
        ),
    ):
        controller._do_compress(60, 150)

    save_dialog.save.assert_called_once()
    assert save_dialog.save.call_args.args[0] is owner


def test_compress_save_failure_removes_owned_temp() -> None:
    owner = _owner(PDFDocument(path="source.pdf", total_pages=1))
    owner._show_error = MagicMock()
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner
    dialog = MagicMock()
    dialog.save_finish.return_value.get_path.return_value = "/output/compressed.pdf"

    with (
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_document_io.copy_pdf_with_ocr_invalidation",
            side_effect=OSError("disk full"),
        ),
        patch("bigocrpdf.utils.temp_manager.remove_tracked_file") as remove_file,
    ):
        controller._finish_compress_save(dialog, MagicMock(), "/tmp/compressed.pdf", "done")

    remove_file.assert_called_once_with("/tmp/compressed.pdf")
    owner._show_error.assert_called_once()
