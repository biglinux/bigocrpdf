"""Focused tests for recoverable UI action state."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from bigocrpdf import OcrDependencyState
from bigocrpdf.processing_controller import ProcessingController
from bigocrpdf.services.screen_capture import ImageOcrOutcome, ImageOcrStatus
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.ui.image_ocr_window import ImageOcrWindow
from bigocrpdf.ui.settings_reset_controller import SettingsResetController
from bigocrpdf.ui.widgets import parse_clipboard_file_paths
from bigocrpdf.ui.window_ui import BigOcrPdfUI, Gtk
from bigocrpdf.window_controller import Gdk, WindowController

_WINDOW_ACTION_NAMES = (
    "add-files",
    "start-processing",
    "remove-all-files",
    "paste-clipboard",
    "cancel-processing",
)
_OCR_AVAILABLE = OcrDependencyState(is_available=True)
_OCR_UNAVAILABLE = OcrDependencyState(
    is_available=False,
    error="RapidOCR could not load ONNX Runtime.",
)


def _make_window_controller(
    page_name: str = "main_view",
    ocr_dependency: OcrDependencyState = _OCR_AVAILABLE,
) -> WindowController:
    parent = MagicMock()
    settings = MagicMock()
    settings.selected_files = ["queued.pdf"]
    ui = MagicMock()
    ui.main_stack.get_visible_child_name.return_value = page_name
    return WindowController(
        parent=parent,
        settings=settings,
        ui=ui,
        file_manager=MagicMock(),
        processing=MagicMock(),
        ocr_dependency=ocr_dependency,
        show_ocr_unavailable=MagicMock(),
        welcome_config_path="",
    )


def test_start_button_remains_owned_by_processing_validation():
    button = MagicMock()
    start = MagicMock(return_value=False)
    header = SimpleNamespace(window=SimpleNamespace(processing=SimpleNamespace(start=start)))

    HeaderBar._on_start_clicked(header, button)

    start.assert_called_once_with(button)
    button.set_sensitive.assert_not_called()


def test_rejected_processing_validation_does_not_disable_start():
    processor = MagicMock()
    processor.has_active_worker.return_value = False
    events = []
    processing = SimpleNamespace(
        ocr_dependency=_OCR_AVAILABLE,
        ocr_processor=processor,
        _get_settings_from_ui=MagicMock(side_effect=lambda: events.append("sync")),
        _validate_ocr_settings=MagicMock(side_effect=lambda: events.append("validate") or False),
    )

    started = ProcessingController.start(processing)

    assert started is False
    assert events == ["sync", "validate"]


def test_failed_processing_start_restores_start_button():
    start_button = MagicMock()
    processor = MagicMock()
    processor.has_active_worker.return_value = False
    processor.process_with_api.return_value = False
    processing = SimpleNamespace(
        ocr_dependency=_OCR_AVAILABLE,
        _validate_ocr_settings=MagicMock(return_value=True),
        _get_settings_from_ui=MagicMock(),
        ui=SimpleNamespace(
            custom_header_bar=SimpleNamespace(start_button=start_button),
            show_toast=MagicMock(),
        ),
        _cleanup_ocr_processor=MagicMock(),
        process_start_time=0.0,
        ocr_processor=processor,
        _on_file_processed=MagicMock(),
        _on_processing_complete=MagicMock(),
    )

    started = ProcessingController.start(processing)

    assert started is False
    assert start_button.set_sensitive.call_args_list == [call(False), call(True)]


def test_processing_start_rejects_worker_that_is_still_stopping():
    processor = MagicMock()
    processor.has_active_worker.return_value = True
    processing = SimpleNamespace(
        ocr_dependency=_OCR_AVAILABLE,
        ocr_processor=processor,
        ui=SimpleNamespace(show_toast=MagicMock()),
        _get_settings_from_ui=MagicMock(),
        _validate_ocr_settings=MagicMock(),
    )

    assert ProcessingController.start(processing) is False

    processing._validate_ocr_settings.assert_not_called()
    processing._get_settings_from_ui.assert_not_called()
    processor.process_with_api.assert_not_called()
    processing.ui.show_toast.assert_called_once()


def test_unavailable_processing_stops_before_validation_or_processor_io():
    processing = SimpleNamespace(
        ocr_dependency=_OCR_UNAVAILABLE,
        _show_ocr_unavailable=MagicMock(),
        _get_settings_from_ui=MagicMock(),
        _validate_ocr_settings=MagicMock(),
        ocr_processor=MagicMock(),
    )

    started = ProcessingController.start(processing)

    assert started is False
    processing._show_ocr_unavailable.assert_called_once_with()
    processing._get_settings_from_ui.assert_not_called()
    processing._validate_ocr_settings.assert_not_called()
    processing.ocr_processor.process_with_api.assert_not_called()


def test_settings_sync_clears_stale_custom_destination():
    settings = SimpleNamespace(
        lang="en",
        destination_folder="/old/output",
        save_settings=MagicMock(),
    )
    settings_page = SimpleNamespace(
        lang_dropdown=None,
        dest_entry=SimpleNamespace(get_text=MagicMock(return_value="")),
        folder_combo=SimpleNamespace(get_selected=MagicMock(return_value=1)),
    )
    processing = SimpleNamespace(
        settings=settings,
        ui=SimpleNamespace(settings_page_manager=settings_page),
        ocr_processor=MagicMock(),
    )
    processing.get_save_in_same_folder = ProcessingController.get_save_in_same_folder.__get__(
        processing
    )

    ProcessingController._get_settings_from_ui(processing)

    assert settings.destination_folder == ""
    settings.save_settings.assert_called_once_with("en", "", False)


def test_cancelled_capture_restores_stable_page_without_showing_error():
    text_buffer = MagicMock()
    text_buffer.get_bounds.return_value = (object(), object())
    text_buffer.get_text.return_value = "previous result"
    window = SimpleNamespace(
        _show_error=MagicMock(),
        _stack=MagicMock(),
        _stable_page_name="welcome",
        _text_buffer=text_buffer,
        _copy_button=MagicMock(),
        _result_copyable=True,
    )
    window._sync_copy_button_state = ImageOcrWindow._sync_copy_button_state.__get__(window)

    ImageOcrWindow._on_processing_complete(
        window,
        ImageOcrOutcome(ImageOcrStatus.CANCELLED),
    )

    window._show_error.assert_not_called()
    window._stack.set_visible_child_name.assert_called_once_with("welcome")
    window._copy_button.set_sensitive.assert_called_once_with(True)


def test_failed_image_ocr_restores_copy_for_previous_result():
    text_buffer = MagicMock()
    text_buffer.get_bounds.return_value = (object(), object())
    text_buffer.get_text.return_value = "previous result"
    window = SimpleNamespace(
        _show_error=MagicMock(),
        _stack=MagicMock(),
        _stable_page_name="results",
        _text_buffer=text_buffer,
        _copy_button=MagicMock(),
        _result_copyable=True,
    )
    window._sync_copy_button_state = ImageOcrWindow._sync_copy_button_state.__get__(window)

    ImageOcrWindow._on_processing_complete(
        window,
        ImageOcrOutcome(ImageOcrStatus.ERROR, message="failed"),
    )

    window._show_error.assert_called_once_with("failed")
    window._stack.set_visible_child_name.assert_called_once_with("results")
    window._copy_button.set_sensitive.assert_called_once_with(True)


@pytest.mark.parametrize("status", (ImageOcrStatus.CANCELLED, ImageOcrStatus.ERROR))
def test_empty_result_message_never_becomes_copyable_after_later_failure(status):
    text_buffer = MagicMock()
    text_buffer.get_bounds.return_value = (object(), object())
    text_buffer.get_text.return_value = "No text extracted."
    window = SimpleNamespace(
        _show_error=MagicMock(),
        _stack=MagicMock(),
        _stable_page_name="results",
        _text_buffer=text_buffer,
        _copy_button=MagicMock(),
        _result_copyable=False,
    )
    window._sync_copy_button_state = ImageOcrWindow._sync_copy_button_state.__get__(window)

    ImageOcrWindow._on_processing_complete(
        window,
        ImageOcrOutcome(status, message="failed" if status == ImageOcrStatus.ERROR else None),
    )

    window._copy_button.set_sensitive.assert_called_once_with(False)


@pytest.mark.parametrize(
    ("page_name", "enabled_actions"),
    (
        (
            "main_view",
            {"add-files", "start-processing", "remove-all-files", "paste-clipboard"},
        ),
        ("terminal", {"cancel-processing"}),
        ("conclusion", set()),
    ),
)
def test_window_action_matrix_follows_main_stack(page_name, enabled_actions):
    actions = {name: MagicMock() for name in _WINDOW_ACTION_NAMES}
    controller = _make_window_controller(page_name)
    controller.parent.lookup_action.side_effect = actions.get

    controller.sync_for_page()

    for action_name, action in actions.items():
        action.set_enabled.assert_called_once_with(action_name in enabled_actions)


@pytest.mark.parametrize(
    ("page_name", "is_queue_page"),
    (("main_view", True), ("terminal", False), ("conclusion", False)),
)
def test_queue_shortcuts_and_drop_follow_main_stack(page_name, is_queue_page):
    settings_page_manager = SimpleNamespace(
        _remove_all_files=MagicMock(),
        _on_drop=MagicMock(return_value=True),
    )
    controller = _make_window_controller(page_name)
    controller.ui.settings_page_manager = settings_page_manager

    controller._on_add_files_action(MagicMock(), None)
    controller._on_start_processing_action(MagicMock(), None)
    controller._on_remove_all_files_action(MagicMock(), None)
    drop_handled = controller._on_global_drop(MagicMock(), MagicMock(), 0, 0)
    drop_action = controller._on_global_drop_enter(MagicMock(), 0, 0)

    assert controller.file_manager.show_open_files_dialog.called is is_queue_page
    assert controller.processing.start.called is is_queue_page
    assert settings_page_manager._remove_all_files.called is is_queue_page
    assert drop_handled is is_queue_page
    assert bool(drop_action & Gdk.DragAction.COPY) is is_queue_page


def test_unavailable_window_action_disables_start_but_preserves_queue_actions():
    actions = {name: MagicMock() for name in _WINDOW_ACTION_NAMES}
    controller = _make_window_controller(ocr_dependency=_OCR_UNAVAILABLE)
    controller.parent.lookup_action.side_effect = actions.get

    controller.sync_for_page()

    actions["start-processing"].set_enabled.assert_called_once_with(False)
    for action_name in ("add-files", "remove-all-files", "paste-clipboard"):
        actions[action_name].set_enabled.assert_called_once_with(True)


def test_unavailable_start_shortcut_reports_cause_without_starting():
    controller = _make_window_controller(ocr_dependency=_OCR_UNAVAILABLE)

    controller._on_start_processing_action(MagicMock(), None)

    controller._show_ocr_unavailable.assert_called_once_with()
    controller.processing.start.assert_not_called()


def test_clear_file_queue_repaints_even_when_the_model_was_already_empty():
    """The screen and the model go out of step, so one cannot stand for the other.

    Files are removed one by one as they finish, while the terminal page is up
    and nothing repaints. By the time the user comes back the model is empty
    and ``_clear_files`` reports "nothing to do" -- and returning early on that
    left the finished batch still listed with the header button stuck on the
    label it took when processing started.
    """
    controller = _make_window_controller()
    controller.settings.selected_files = []
    controller.settings._clear_files.return_value = False

    controller.clear_file_queue()

    controller.ui.update_file_info.assert_called_once_with()


def test_clear_file_queue_repaints_after_a_persisted_clear():
    controller = _make_window_controller()
    controller.settings._clear_files.return_value = True

    controller.clear_file_queue()

    controller.ui.update_file_info.assert_called_once_with()


def test_clear_file_queue_repaints_from_the_model_not_from_zero():
    """``_clear_files`` also reports False after a failed save rolled the queue back.

    Asserting an empty header there would claim the files were gone while they
    are still queued, so the repaint has to read the model.
    """
    controller = _make_window_controller()
    controller.settings._clear_files.return_value = False

    controller.clear_file_queue()

    controller.ui.custom_header_bar.update_queue_size.assert_not_called()


@pytest.mark.parametrize(
    ("layout", "buttons_on_left"),
    (
        ("close,minimize,maximize:", True),
        ("menu:minimize,maximize,close", False),
    ),
)
def test_window_button_position_uses_gtk_decoration_layout(layout, buttons_on_left):
    controller = _make_window_controller()
    gtk_settings = MagicMock()
    gtk_settings.get_property.return_value = layout

    with patch(
        "bigocrpdf.window_controller.Gtk.Settings.get_default",
        return_value=gtk_settings,
    ):
        assert controller.window_buttons_on_left() is buttons_on_left

    gtk_settings.get_property.assert_called_once_with("gtk-decoration-layout")


@pytest.mark.parametrize("page_name", ("terminal", "conclusion"))
def test_paste_shortcut_does_not_read_clipboard_outside_queue(page_name):
    controller = _make_window_controller(page_name)

    with patch("bigocrpdf.window_controller.get_default_clipboard") as get_clipboard:
        controller._on_paste_clipboard_action(MagicMock(), None)

    get_clipboard.assert_not_called()


def test_pending_clipboard_read_cannot_mutate_queue_after_page_transition():
    clipboard = MagicMock()
    result = MagicMock()
    controller = _make_window_controller("terminal")
    controller._read_clipboard_uri_text = MagicMock(return_value="file:///tmp/queued.pdf")

    with patch("bigocrpdf.window_controller.parse_clipboard_file_paths") as parse_paths:
        controller._on_clipboard_uris_ready(clipboard, result)

    controller._read_clipboard_uri_text.assert_called_once_with(clipboard, result)
    parse_paths.assert_not_called()


def test_clipboard_texture_discards_png_after_conversion_failure():
    clipboard = MagicMock()
    texture = clipboard.read_texture_finish.return_value
    texture.save_to_png_bytes.return_value.get_data.return_value = b"png"
    controller = _make_window_controller()
    controller._is_queue_page = MagicMock(return_value=True)
    opened_file = mock_open()

    with (
        patch(
            "bigocrpdf.window_controller.mkstemp",
            return_value=(10, "/tmp/paste.png"),
        ),
        patch("bigocrpdf.window_controller.os.fdopen", opened_file),
        patch(
            "bigocrpdf.window_controller.images_to_pdf",
            side_effect=OSError("conversion failed"),
        ),
        patch("bigocrpdf.window_controller.remove_file") as remove_file,
    ):
        controller._on_clipboard_texture_ready(clipboard, MagicMock())

    opened_file.assert_called_once_with(10, "wb")
    opened_file().write.assert_called_once_with(b"png")
    remove_file.assert_called_once_with("/tmp/paste.png")


def test_clipboard_uri_toast_reports_only_files_actually_queued():
    controller = _make_window_controller()
    controller._read_clipboard_uri_text = MagicMock(return_value="clipboard payload")
    controller._supported_clipboard_files = MagicMock(
        return_value=["one.pdf", "duplicate.pdf", "scan.png"]
    )
    controller._add_clipboard_supported_files = MagicMock(return_value=1)

    with patch(
        "bigocrpdf.window_controller.parse_clipboard_file_paths",
        return_value=["one.pdf", "duplicate.pdf", "scan.png"],
    ):
        controller._on_clipboard_uris_ready(MagicMock(), MagicMock())

    controller.ui.update_file_info.assert_called_once_with()
    controller.ui.show_toast.assert_called_once_with("1 file added from clipboard")


def test_multiple_clipboard_images_wait_for_merge_choice_before_counting():
    controller = _make_window_controller()
    controller.settings.add_files.return_value = 1

    with patch(
        "bigocrpdf.window_controller.is_image_file",
        side_effect=lambda path: path.endswith(".png"),
    ):
        added = controller._add_clipboard_supported_files(
            ["document.pdf", "first.png", "second.png"]
        )

    assert added == 1
    controller.settings.add_files.assert_called_once_with(["document.pdf"])
    controller.ui.dialogs_manager.show_image_merge_dialog.assert_called_once()


def test_clipboard_png_is_removed_when_writing_it_fails():
    clipboard = MagicMock()
    texture = clipboard.read_texture_finish.return_value
    texture.save_to_png_bytes.return_value.get_data.return_value = b"png"
    controller = _make_window_controller()
    controller._is_queue_page = MagicMock(return_value=True)
    opened_file = mock_open()
    opened_file().write.side_effect = OSError("disk full")

    with (
        patch(
            "bigocrpdf.window_controller.mkstemp",
            return_value=(10, "/tmp/paste.png"),
        ),
        patch("bigocrpdf.window_controller.os.fdopen", opened_file),
        patch("bigocrpdf.window_controller.remove_file") as remove_file,
    ):
        controller._on_clipboard_texture_ready(clipboard, MagicMock())

    remove_file.assert_called_once_with("/tmp/paste.png")


def test_clipboard_uri_stream_closes_when_data_is_missing():
    clipboard = MagicMock()
    stream = MagicMock()
    stream.read_bytes.return_value.get_data.return_value = None
    clipboard.read_finish.return_value = (stream, "text/uri-list")
    controller = _make_window_controller()

    assert controller._read_clipboard_uri_text(clipboard, MagicMock()) is None
    stream.close.assert_called_once_with(None)


def test_clipboard_file_parser_rejects_remote_file_authority(tmp_path):
    local_file = tmp_path / "scan.png"
    local_file.write_bytes(b"png")
    payload = f"file://remote-host{local_file}\n{local_file.as_uri()}"

    assert parse_clipboard_file_paths(payload) == [str(local_file)]


def test_main_stack_transition_resynchronizes_window_actions():
    stack = SimpleNamespace(get_visible_child_name=MagicMock(return_value="terminal"))
    window = SimpleNamespace(actions=SimpleNamespace(sync_for_page=MagicMock()))
    ui = SimpleNamespace(window=window)

    BigOcrPdfUI._on_main_stack_changed(ui, stack, None)

    window.actions.sync_for_page.assert_called_once_with("terminal")


def test_ui_cleanup_continues_after_terminal_cleanup_failure():
    settings_cleanup = MagicMock()
    terminal_cleanup = MagicMock(side_effect=RuntimeError("failed"))
    conclusion_reset = MagicMock()
    ui = SimpleNamespace(
        settings_page_manager=SimpleNamespace(cleanup=settings_cleanup),
        terminal_page_manager=SimpleNamespace(cleanup=terminal_cleanup),
        conclusion_page_manager=SimpleNamespace(reset_page=conclusion_reset),
    )

    BigOcrPdfUI.cleanup(ui)

    settings_cleanup.assert_called_once_with()
    terminal_cleanup.assert_called_once_with()
    conclusion_reset.assert_called_once_with()


def test_reset_syncs_settings_owner_before_toast_and_preserves_queue():
    settings = SimpleNamespace(
        selected_files=["scan.pdf"],
        original_file_paths={"scan.pdf": "original.png"},
        page_ranges={"scan.pdf": (2, 4)},
        file_modifications={"scan.pdf": {"rotation": 90}},
    )

    def reset_to_defaults():
        settings.selected_files = []
        settings.original_file_paths = {}
        settings.page_ranges = {}
        settings.file_modifications = {}

    settings.reset_to_defaults = MagicMock(side_effect=reset_to_defaults)
    event_order = []

    def sync_ui_to_settings():
        event_order.append("sync")
        assert settings.selected_files == ["scan.pdf"]
        assert settings.original_file_paths == {"scan.pdf": "original.png"}
        assert settings.page_ranges == {"scan.pdf": (2, 4)}
        assert settings.file_modifications == {"scan.pdf": {"rotation": 90}}

    settings_page_manager = SimpleNamespace(
        sync_ui_to_settings=MagicMock(side_effect=sync_ui_to_settings)
    )
    controller = _make_window_controller()
    controller.settings = settings
    controller.ui = SimpleNamespace(
        settings_page_manager=settings_page_manager,
        show_toast=MagicMock(side_effect=lambda _message: event_order.append("toast")),
    )

    reset = SettingsResetController(
        MagicMock(),
        settings,
        settings_page_manager.sync_ui_to_settings,
        controller.ui.show_toast,
    )
    reset._on_response(MagicMock(), "reset")

    settings.reset_to_defaults.assert_called_once_with()
    settings_page_manager.sync_ui_to_settings.assert_called_once_with()
    assert event_order == ["sync", "toast"]


def test_failed_reset_preserves_queue_and_reports_error():
    settings = SimpleNamespace(
        selected_files=["scan.pdf"],
        original_file_paths={"scan.pdf": "original.png"},
        page_ranges={"scan.pdf": (2, 4)},
        file_modifications={"scan.pdf": {"rotation": 90}},
        reset_to_defaults=MagicMock(side_effect=OSError("disk full")),
    )

    def sync_ui_to_settings():
        assert settings.selected_files == ["scan.pdf"]
        assert settings.original_file_paths == {"scan.pdf": "original.png"}
        assert settings.page_ranges == {"scan.pdf": (2, 4)}
        assert settings.file_modifications == {"scan.pdf": {"rotation": 90}}

    sync = MagicMock(side_effect=sync_ui_to_settings)
    show_toast = MagicMock()
    reset = SettingsResetController(MagicMock(), settings, sync, show_toast)

    reset._on_response(MagicMock(), "reset")

    sync.assert_called_once_with()
    show_toast.assert_called_once_with("Error saving settings: disk full")


def test_full_width_header_uses_system_decoration_layout():
    header = MagicMock()
    toolbar = MagicMock()
    controller = SimpleNamespace(window=SimpleNamespace())

    with (
        patch("bigocrpdf.ui.window_ui.Adw.HeaderBar", return_value=header),
        patch("bigocrpdf.ui.window_ui.Adw.ToolbarView", return_value=toolbar),
        patch("bigocrpdf.ui.window_ui.Gtk.Label", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gtk.MenuButton", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gio.Menu", return_value=MagicMock()),
    ):
        result = BigOcrPdfUI._create_full_width_page_with_header(
            controller,
            is_terminal=True,
        )

    assert result is toolbar
    header.set_decoration_layout.assert_not_called()


def test_toast_message_is_plain_text():
    toast = MagicMock()
    overlay = MagicMock()
    ui = SimpleNamespace(toast_overlay=overlay)

    with patch("bigocrpdf.ui.window_ui.Adw.Toast.new", return_value=toast):
        BigOcrPdfUI.show_toast(ui, "<scan & copy>", timeout=5)

    toast.set_use_markup.assert_called_once_with(False)
    toast.set_timeout.assert_called_once_with(5)
    overlay.add_toast.assert_called_once_with(toast)


def test_left_sidebar_preserves_system_button_layout():
    toolbar = MagicMock()
    header = MagicMock()
    controller = SimpleNamespace(
        window=SimpleNamespace(
            actions=SimpleNamespace(window_buttons_on_left=MagicMock(return_value=True))
        ),
        split_view=MagicMock(),
    )

    with (
        patch("bigocrpdf.ui.window_ui.Adw.ToolbarView", return_value=toolbar),
        patch("bigocrpdf.ui.window_ui.Adw.HeaderBar", return_value=header),
        patch("bigocrpdf.ui.window_ui.Gtk.Box", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gtk.Label", return_value=MagicMock()),
    ):
        BigOcrPdfUI._create_left_sidebar(controller)

    header.set_decoration_layout.assert_not_called()


def test_conclusion_back_icon_is_decorative():
    header = MagicMock()
    toolbar = MagicMock()
    back_icon = MagicMock()
    controller = SimpleNamespace(window=SimpleNamespace())

    with (
        patch("bigocrpdf.ui.window_ui.Adw.HeaderBar", return_value=header),
        patch("bigocrpdf.ui.window_ui.Adw.ToolbarView", return_value=toolbar),
        patch("bigocrpdf.ui.window_ui.Gtk.Button", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gtk.Box", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gtk.Image.new_from_icon_name", return_value=back_icon),
        patch("bigocrpdf.ui.window_ui.Gtk.Label", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gtk.MenuButton", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.Gio.Menu", return_value=MagicMock()),
        patch("bigocrpdf.ui.window_ui.set_a11y_label"),
    ):
        BigOcrPdfUI._create_full_width_page_with_header(controller)

    back_icon.set_accessible_role.assert_called_once_with(Gtk.AccessibleRole.PRESENTATION)
