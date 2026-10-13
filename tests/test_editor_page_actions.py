"""Regression tests for editor actions that own undo snapshots."""

import dataclasses
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

import gi

gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gio, Gtk

from bigocrpdf.ui.pdf_editor.editor_help_controller import EditorHelpController
from bigocrpdf.ui.pdf_editor.editor_page_actions_mixin import EditorPageActionsMixin
from bigocrpdf.ui.pdf_editor.editor_tools_controller import EditorToolsController
from bigocrpdf.ui.pdf_editor.editor_window_layout import EditorWindowLayoutMixin
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
            _saving=False,
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
        patch("bigocrpdf.utils.durable_writes.copy_file_atomically") as copy_pdf,
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


def _compress(
    compress_pdf: MagicMock,
    *,
    mapped: bool = True,
    size_after: int = 1,
    response: str = "save",
) -> tuple[Any, MagicMock, MagicMock, MagicMock]:
    """Run the editor's compression with the worker thread inlined, the user
    editing while it runs, and the result dialog answered with ``response``."""
    document = PDFDocument(path="source.pdf", total_pages=2)
    document.pages[0].thumbnail_pixbuf = object()
    pages_at_start = [dataclasses.replace(page, thumbnail_pixbuf=None) for page in document.pages]
    owner = _owner(document)
    owner._pdf_path = "source.pdf"
    owner._default_save_dir = MagicMock(return_value="/output")
    owner._show_busy = MagicMock()
    owner._end_busy = MagicMock()
    owner.get_mapped = MagicMock(return_value=mapped)
    action = MagicMock()
    owner._actions = {"compress": action}
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner
    alert = MagicMock()
    save_dialog = MagicMock()
    sizes = {"/tmp/edited.pdf": 2, "/tmp/compressed.pdf": size_after}

    with (
        patch(
            "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
            return_value=True,
        ) as apply_changes,
        patch("bigocrpdf.services.pdf_operations.compress_pdf", compress_pdf),
        patch(
            "bigocrpdf.utils.temp_manager.mkstemp",
            side_effect=[
                (10, "/tmp/edited.pdf"),
                (11, "/tmp/compressed.pdf"),
            ],
        ),
        patch("bigocrpdf.ui.pdf_editor.editor_tools_controller.os.close"),
        patch(
            "bigocrpdf.ui.pdf_editor.editor_tools_controller.os.path.getsize",
            side_effect=sizes.__getitem__,
        ),
        patch("bigocrpdf.utils.temp_manager.remove_file") as remove_file,
        patch(
            "bigocrpdf.ui.pdf_editor.editor_tools_controller.Adw.AlertDialog",
            return_value=alert,
        ),
        patch(
            "bigocrpdf.ui.pdf_editor.editor_tools_controller.Gtk.FileDialog",
            return_value=save_dialog,
        ),
        patch("bigocrpdf.ui.pdf_editor.page_thumbnail_export.run_in_worker") as run_in_worker,
    ):

        def edit_while_compressing(work, on_done):
            action.set_enabled.assert_called_with(False)
            # The document is written in the worker, not on the main loop.
            apply_changes.assert_not_called()
            document.pages[0].deleted = True
            document.pages.append(PageState(page_number=3))
            on_done(work())

        run_in_worker.side_effect = edit_while_compressing
        controller._do_compress(60, 150)
        if alert.connect.called:
            alert.connect.call_args.args[1](alert, response)

    action.set_enabled.assert_called_with(True)
    owner._end_busy.assert_called_once()
    snapshot, written = apply_changes.call_args.args
    assert written == "/tmp/edited.pdf"
    assert snapshot is not document
    assert snapshot.pages == pages_at_start
    remove_file.assert_any_call("/tmp/edited.pdf")
    return owner, alert, save_dialog, remove_file


def test_compress_shows_size_before_saving() -> None:
    owner, alert, save_dialog, remove_file = _compress(
        MagicMock(return_value=SimpleNamespace(success=True, message="compressed"))
    )

    assert "→" in alert.set_body.call_args.args[0]
    alert.present.assert_called_once_with(owner)
    save_dialog.save.assert_called_once()
    assert save_dialog.save.call_args.args[0] is owner
    assert call("/tmp/compressed.pdf") not in remove_file.call_args_list


def test_cancelled_compress_removes_its_result() -> None:
    _, _, save_dialog, remove_file = _compress(
        MagicMock(return_value=SimpleNamespace(success=True, message="compressed")),
        response="cancel",
    )

    save_dialog.save.assert_not_called()
    remove_file.assert_any_call("/tmp/compressed.pdf")


def test_unshrunk_compress_offers_no_save() -> None:
    _, alert, save_dialog, remove_file = _compress(
        MagicMock(return_value=SimpleNamespace(success=True, message="compressed")),
        size_after=3,
        response="cancel",
    )

    responses = [args.args[0] for args in alert.add_response.call_args_list]
    assert responses == ["cancel"]
    save_dialog.save.assert_not_called()
    remove_file.assert_any_call("/tmp/compressed.pdf")


def test_compress_crash_is_reported() -> None:
    owner, alert, _, remove_file = _compress(
        MagicMock(side_effect=RuntimeError("1-bit RGB is not supported"))
    )

    alert.present.assert_not_called()
    remove_file.assert_any_call("/tmp/compressed.pdf")
    assert "1-bit RGB" in owner._show_error.call_args.args[0]


def test_closed_editor_gets_no_save_dialog() -> None:
    _, alert, _, remove_file = _compress(
        MagicMock(return_value=SimpleNamespace(success=True, message="compressed")), mapped=False
    )

    alert.present.assert_not_called()
    remove_file.assert_any_call("/tmp/compressed.pdf")


def test_compress_save_failure_removes_owned_temp() -> None:
    owner = _owner(PDFDocument(path="source.pdf", total_pages=1))
    owner._show_error = MagicMock()
    controller = EditorToolsController.__new__(EditorToolsController)
    controller._owner = owner
    dialog = MagicMock()
    dialog.save_finish.return_value.get_path.return_value = "/output/compressed.pdf"

    with (
        patch(
            "bigocrpdf.utils.durable_writes.copy_file_atomically",
            side_effect=OSError("disk full"),
        ),
        patch("bigocrpdf.utils.temp_manager.remove_tracked_file") as remove_file,
    ):
        controller._finish_compress_save(dialog, MagicMock(), "/tmp/compressed.pdf", "done")

    remove_file.assert_called_once_with("/tmp/compressed.pdf")
    owner._show_error.assert_called_once()


def _banner_window() -> Any:
    from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

    label = SimpleNamespace(text="")
    label.set_text = lambda text: setattr(label, "text", text)
    label.get_text = lambda: label.text
    window = SimpleNamespace(
        _notification_timer_id=None,
        _busy_message=None,
        _notification_icon=MagicMock(),
        _notification_label=label,
        _notification_box=MagicMock(),
        _notification_revealer=MagicMock(),
        announce=MagicMock(),
    )
    for name in (
        "_show_notification",
        "_on_notification_timeout",
        "_hide_notification",
        "_show_info",
        "_show_busy",
        "_end_busy",
    ):
        setattr(window, name, getattr(PDFEditorWindow, name).__get__(window))
    return window


def test_busy_banner_outlasts_other_messages() -> None:
    window = _banner_window()
    timers: list[Any] = []

    with (
        patch("bigocrpdf.ui.pdf_editor.editor_window.set_a11y_label"),
        patch(
            "bigocrpdf.ui.pdf_editor.editor_window.GLib.timeout_add_seconds",
            side_effect=lambda _seconds, callback: timers.append(callback) or len(timers),
        ),
        patch("bigocrpdf.ui.pdf_editor.editor_window.GLib.source_remove") as source_remove,
    ):
        window._show_busy("Compressing")
        window._show_info("Saved")
        timers[-1]()
        assert window._notification_label.text == "Compressing"
        window._notification_revealer.set_reveal_child.assert_called_with(True)

        window._show_info("Saved again")
        window._end_busy()
        assert window._notification_label.text == "Saved again"
        source_remove.assert_not_called()
        timers[-1]()
        window._notification_revealer.set_reveal_child.assert_called_with(False)

        window._show_busy("Compressing")
        window._end_busy()
        window._notification_revealer.set_reveal_child.assert_called_with(False)


def test_edits_wait_for_running_save() -> None:
    document = PDFDocument(path="source.pdf", total_pages=1)
    owner = _owner(document)
    owner._saving = True
    owner._undo = MagicMock()
    owner._page_grid.select_all = MagicMock()
    owner._paste_from_clipboard = MagicMock()

    assert EditorPageActionsMixin._handle_ctrl_shortcut(owner, Gdk.KEY_z)
    assert EditorPageActionsMixin._add_files_to_document(owner, ["added.pdf"]) == 0

    owner._undo.assert_not_called()
    assert document.total_pages == 1


def test_help_lists_every_ctrl_shortcut() -> None:
    content = Gtk.Box()
    EditorHelpController._append_help_shortcuts(content)
    listed = set()
    row = content.get_last_child().get_first_child()
    while row is not None:
        for key in row.get_first_child().get_label().split(" / "):
            if key.startswith("Ctrl+"):
                name = key.removeprefix("Ctrl+")
                listed.add(Gdk.keyval_from_name(name.lower() if len(name) == 1 else name))
        row = row.get_next_sibling()

    owner = cast(Any, MagicMock(_saving=True))
    candidates = [Gdk.keyval_from_name(chr(c)) for c in range(ord("a"), ord("z") + 1)]
    candidates += [Gdk.KEY_Up, Gdk.KEY_Down, Gdk.KEY_Left, Gdk.KEY_Right]
    handled = {
        key for key in candidates if EditorPageActionsMixin._handle_ctrl_shortcut(owner, key)
    }

    assert listed == handled


def test_sidebar_rows_offer_activate_action() -> None:
    Adw.init()
    reversed_pages = MagicMock()
    selected = MagicMock()
    editor = Gio.SimpleActionGroup()
    reverse = Gio.SimpleAction.new("reverse", None)
    reverse.connect("activate", lambda *_: reversed_pages())
    editor.add_action(reverse)
    sidebar = Gtk.Box()
    sidebar.insert_action_group("editor", editor)
    layout = cast(Any, None)
    action_row = EditorWindowLayoutMixin._document_action_row(
        layout, "Reverse", "icon", "editor.reverse"
    )
    signal_row = EditorWindowLayoutMixin._page_action_row(layout, "Select", "icon", selected)
    sidebar.append(action_row)
    sidebar.append(signal_row)

    assert action_row.activate_action("row.activate", None)
    assert signal_row.activate_action("row.activate", None)
    reversed_pages.assert_called_once_with()
    selected.assert_called_once_with(signal_row)


def test_include_all_row_claims_no_shortcut() -> None:
    Adw.init()

    class Sidebar(EditorWindowLayoutMixin):
        _on_include_all = MagicMock()

    sidebar = Sidebar()
    group = sidebar._create_page_actions_group()
    rows = []
    stack = [group]
    while stack:
        child = stack.pop()
        if isinstance(child, Adw.ActionRow):
            rows.append(child)
        widget = child.get_first_child()
        while widget is not None:
            stack.append(widget)
            widget = widget.get_next_sibling()
    include_all = next(row for row in rows if "Include" in row.get_title())

    include_all.activate_action("row.activate", None)

    # No shortcut includes every page: Ctrl+A selects them instead.
    assert "Ctrl+" not in include_all.get_tooltip_text()
    Sidebar._on_include_all.assert_called_once()
