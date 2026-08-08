"""Contracts for propagating and enforcing resolved OCR availability."""

from __future__ import annotations

import os
import sys
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

import pytest

import bigocrpdf
import bigocrpdf.config as config
from bigocrpdf import OcrDependencyState
from bigocrpdf.application import BigOcrPdfApp
from bigocrpdf.image_application import ImageOcrApp
from bigocrpdf.services.rapidocr_service.config import OCRConfig
from bigocrpdf.services.screen_capture import ImageOcrOutcome, ImageOcrStatus
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.ui.image_ocr_window import ImageOcrWindow
from bigocrpdf.window import BigOcrPdfWindow

_OCR_AVAILABLE = OcrDependencyState(is_available=True)
_OCR_UNAVAILABLE = OcrDependencyState(
    is_available=False,
    error="RapidOCR could not load ONNX Runtime.",
)


def _fake_gi_modules(
    gtk_version: tuple[int, int, int],
    adw_version: tuple[int, int, int],
) -> dict[str, ModuleType]:
    gi_module = cast(Any, ModuleType("gi"))
    repository = cast(Any, ModuleType("gi.repository"))
    gi_module.require_version = MagicMock()
    gi_module.repository = repository
    repository.Gtk = SimpleNamespace(
        get_major_version=lambda: gtk_version[0],
        get_minor_version=lambda: gtk_version[1],
        get_micro_version=lambda: gtk_version[2],
    )
    repository.Adw = SimpleNamespace(
        get_major_version=lambda: adw_version[0],
        get_minor_version=lambda: adw_version[1],
        get_micro_version=lambda: adw_version[2],
    )
    return {"gi": gi_module, "gi.repository": repository}


def _bind_image_lifecycle(window: SimpleNamespace) -> None:
    """Attach the private lifecycle methods used by focused namespace tests."""
    window._alive = True
    window._active_request = getattr(window, "_active_request", None)
    window._processing_generation = getattr(window, "_processing_generation", 0)
    window._hidden_capture_generation = getattr(
        window,
        "_hidden_capture_generation",
        None,
    )
    window._is_hidden_for_capture = getattr(window, "_is_hidden_for_capture", False)
    window._capture_delay_source_id = 0
    window._focus_idle_source_id = 0
    window._input_cancellable = None
    window._stable_page_name = "welcome"
    window._remove_source = MethodType(ImageOcrWindow._remove_source, window)
    window._set_cancel_enabled = MagicMock()
    if not hasattr(window, "_restore_after_capture"):
        window._restore_after_capture = MethodType(
            ImageOcrWindow._restore_after_capture,
            window,
        )
    window._begin_operation = MethodType(ImageOcrWindow._begin_operation, window)
    window._is_current_operation = MethodType(
        ImageOcrWindow._is_current_operation,
        window,
    )


@pytest.mark.parametrize(
    ("gtk_version", "adw_version", "expected_error"),
    [
        ((4, 13, 9), (1, 5, 0), "GTK 4.14 or newer is required"),
        ((4, 14, 0), (1, 4, 9), "libadwaita 1.5 or newer is required"),
    ],
)
def test_gtk_probe_rejects_unsupported_runtime(
    gtk_version: tuple[int, int, int],
    adw_version: tuple[int, int, int],
    expected_error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch.dict(sys.modules, _fake_gi_modules(gtk_version, adw_version)):
        assert not bigocrpdf._check_gtk_dependencies()

    assert expected_error in capsys.readouterr().err


@pytest.mark.parametrize(
    ("gtk_version", "adw_version"),
    [
        # The stack an AppImage built on Ubuntu 24.04 carries.
        ((4, 14, 5), (1, 5, 0)),
        # A current desktop.
        ((4, 22, 0), (1, 9, 3)),
    ],
)
def test_gtk_probe_accepts_supported_runtime(
    gtk_version: tuple[int, int, int],
    adw_version: tuple[int, int, int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch.dict(sys.modules, _fake_gi_modules(gtk_version, adw_version)):
        assert bigocrpdf._check_gtk_dependencies()

    assert capsys.readouterr().err == ""


def test_pdf_application_registers_expected_shortcuts() -> None:
    app = SimpleNamespace(set_accels_for_action=MagicMock())

    BigOcrPdfApp._setup_keyboard_shortcuts(app)

    assert app.set_accels_for_action.call_args_list == [
        call("app.quit", [config.SHORTCUTS.get("quit", "<Control>q")]),
        call("app.about", [config.SHORTCUTS.get("about", "F1")]),
        call("app.shortcuts", ["<Control>question"]),
        call("win.add-files", [config.SHORTCUTS.get("add-files", "<Control>o")]),
        call(
            "win.start-processing",
            [config.SHORTCUTS.get("start-processing", "<Control>Return")],
        ),
        call(
            "win.cancel-processing",
            [config.SHORTCUTS.get("cancel-processing", "Escape")],
        ),
        call(
            "win.remove-all-files",
            [config.SHORTCUTS.get("remove-all-files", "<Control>r")],
        ),
        call(
            "win.paste-clipboard",
            [config.SHORTCUTS.get("paste-clipboard", "<Control>v")],
        ),
    ]


def test_pdf_application_does_not_hide_shortcut_registration_errors() -> None:
    app = SimpleNamespace(
        set_accels_for_action=MagicMock(side_effect=RuntimeError("invalid accelerator"))
    )

    with pytest.raises(RuntimeError, match="invalid accelerator"):
        BigOcrPdfApp._setup_keyboard_shortcuts(app)


def test_pdf_application_uses_libadwaita_shortcuts_dialog() -> None:
    class FakeDialog:
        def __init__(self) -> None:
            self.sections = []

        def add(self, section) -> None:
            self.sections.append(section)

    class FakeSection:
        def __init__(self, *, title: str) -> None:
            self.title = title
            self.items = []

        def add(self, item) -> None:
            self.items.append(item)

    class FakeItem:
        @staticmethod
        def new(title: str, accelerator: str) -> tuple[str, str]:
            return title, accelerator

    fake_adw = SimpleNamespace(
        ShortcutsDialog=FakeDialog,
        ShortcutsSection=FakeSection,
        ShortcutsItem=FakeItem,
    )

    # The widgets now live behind utils.adw_compat, which picks the native
    # implementation whenever libadwaita exposes it.
    with (
        patch("bigocrpdf.utils.adw_compat.Adw", fake_adw),
        patch("bigocrpdf.utils.adw_compat.HAS_ADW_SHORTCUTS_DIALOG", True),
    ):
        dialog = BigOcrPdfApp._build_shortcuts_dialog(SimpleNamespace())

    assert [len(section.items) for section in dialog.sections] == [3, 3, 2]
    assert dialog.sections[0].items[0][1] == config.SHORTCUTS.get("add-files", "<Control>o")
    assert dialog.sections[2].items[0][1] == "<Control>question"


def test_pdf_application_shortcuts_dialog_degrades_on_old_libadwaita() -> None:
    """The same shortcuts must survive on the stack an AppImage may carry."""
    import gi

    gi.require_version("Gtk", "4.0")
    from gi.repository import Adw, Gtk

    with patch("bigocrpdf.utils.adw_compat.HAS_ADW_SHORTCUTS_DIALOG", False):
        dialog = BigOcrPdfApp._build_shortcuts_dialog(SimpleNamespace())

    assert isinstance(dialog, Adw.PreferencesDialog)

    accelerators = []

    def walk(widget) -> None:
        if isinstance(widget, Gtk.ShortcutLabel):
            accelerators.append(widget.get_property("accelerator"))
        child = widget.get_first_child()
        while child is not None:
            walk(child)
            child = child.get_next_sibling()

    walk(dialog.get_child() or dialog)

    # Same eight shortcuts the native dialog shows, none dropped.
    assert len(accelerators) == 8
    assert config.SHORTCUTS.get("add-files", "<Control>o") in accelerators
    assert "<Control>question" in accelerators


def test_main_checks_runtime_before_importing_pdf_application(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "SELECTED_FILE_PATH", str(tmp_path / "selected-files"))
    monkeypatch.setattr(
        config,
        "setup_environment",
        MagicMock(return_value=SimpleNamespace(image_mode=False, files=[])),
    )
    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bigocrpdf.application":
            raise AssertionError("graphical application imported before runtime validation")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(bigocrpdf, "setup_i18n"),
        patch.object(bigocrpdf, "_check_gtk_dependencies", return_value=False),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        assert bigocrpdf.main() == 1


def test_main_image_checks_runtime_before_importing_application() -> None:
    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bigocrpdf.image_application":
            raise AssertionError("image application imported before runtime validation")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(bigocrpdf, "setup_i18n"),
        patch.object(bigocrpdf, "_check_gtk_dependencies", return_value=False),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        assert bigocrpdf.main_image() == 1


def test_main_propagates_failed_probe_to_pdf_application(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_file = tmp_path / "selected-files"
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "SELECTED_FILE_PATH", str(selected_file))
    monkeypatch.setattr(
        config,
        "setup_environment",
        MagicMock(return_value=SimpleNamespace(image_mode=False, files=[])),
    )

    with (
        patch.object(bigocrpdf, "setup_i18n"),
        patch.object(bigocrpdf, "_check_gtk_dependencies", return_value=True),
        patch.object(
            bigocrpdf,
            "_check_ocr_dependencies",
            return_value=(False, _OCR_UNAVAILABLE.error),
        ),
        patch("bigocrpdf.application.BigOcrPdfApp") as application_class,
    ):
        application_class.return_value.run.return_value = 17
        result = bigocrpdf.main()

    assert result == 17
    state = application_class.call_args.kwargs["ocr_dependency"]
    assert state == _OCR_UNAVAILABLE


def test_main_image_propagates_successful_probe_to_image_application() -> None:
    with (
        patch.object(bigocrpdf, "setup_i18n"),
        patch.object(bigocrpdf, "_check_gtk_dependencies", return_value=True),
        patch.object(bigocrpdf, "_check_ocr_dependencies", return_value=(True, "")),
        patch("bigocrpdf.image_application.ImageOcrApp") as application_class,
    ):
        application_class.return_value.run.return_value = 23
        result = bigocrpdf.main_image()

    assert result == 23
    state = application_class.call_args.kwargs["ocr_dependency"]
    assert state == _OCR_AVAILABLE


def test_pdf_application_presents_before_scheduling_unavailable_dialog() -> None:
    events: list[str] = []
    window = SimpleNamespace(
        present=MagicMock(side_effect=lambda: events.append("present")),
        show_ocr_unavailable_dialog=MagicMock(),
    )
    application = SimpleNamespace(
        ocr_dependency=_OCR_UNAVAILABLE,
        _edit_mode=False,
        get_active_window=MagicMock(return_value=None),
    )
    window_constructor = MagicMock(return_value=window)

    class FakePdfWindow:
        def __new__(cls, *args, **kwargs):
            return window_constructor(*args, **kwargs)

    with (
        patch("bigocrpdf.application.init_config"),
        patch("bigocrpdf.application.load_css"),
        patch("bigocrpdf.application.BigOcrPdfWindow", FakePdfWindow),
        patch(
            "bigocrpdf.application.GLib.idle_add",
            side_effect=lambda _callback: events.append("schedule"),
        ),
    ):
        BigOcrPdfApp.on_activate(application, application)

    window_constructor.assert_called_once_with(application, ocr_dependency=_OCR_UNAVAILABLE)
    assert events == ["present", "schedule"]


def test_file_categorization_rejects_unknown_inputs_instead_of_treating_them_as_pdfs() -> None:
    files = [
        SimpleNamespace(get_path=MagicMock(return_value="/tmp/document.pdf")),
        SimpleNamespace(get_path=MagicMock(return_value="/tmp/photo.png")),
        SimpleNamespace(get_path=MagicMock(return_value="/tmp/archive.pdf.gz")),
        SimpleNamespace(get_path=MagicMock(return_value="/tmp/notes.txt")),
    ]

    pdfs, images, unsupported = BigOcrPdfApp._categorize_files(files)

    assert pdfs == ["/tmp/document.pdf"]
    assert images == ["/tmp/photo.png"]
    assert unsupported == ["/tmp/archive.pdf.gz", "/tmp/notes.txt"]


def test_image_editor_bootstrap_failure_removes_owned_pdf(tmp_path) -> None:
    temp_pdf = tmp_path / "bootstrap.pdf"
    descriptor = os.open(temp_pdf, os.O_CREAT | os.O_RDWR, 0o600)
    application = SimpleNamespace()

    with (
        patch(
            "bigocrpdf.ui.pdf_editor.thumbnail_renderer.get_thumbnail_renderer",
            return_value=MagicMock(),
        ),
        patch(
            "bigocrpdf.utils.temp_manager.mkstemp",
            return_value=(descriptor, str(temp_pdf)),
        ),
        patch(
            "bigocrpdf.utils.temp_manager.remove_file",
            wraps=lambda path: os.remove(path),
        ) as remove_file,
        patch("PIL.Image.open", side_effect=OSError("invalid image")),
    ):
        BigOcrPdfApp._open_images_in_editor(cast(Any, application), MagicMock(), ["broken.png"])

    remove_file.assert_called_once_with(str(temp_pdf))
    assert not temp_pdf.exists()


def test_unsupported_input_dialog_names_rejected_paths() -> None:
    parent = MagicMock()
    application = SimpleNamespace(get_active_window=MagicMock(return_value=parent))

    with (
        patch("bigocrpdf.application._", side_effect=lambda text: text),
        patch("bigocrpdf.application.Adw.AlertDialog") as alert_class,
    ):
        result = BigOcrPdfApp._show_unsupported_files_dialog(
            application,
            ["/tmp/archive.pdf.gz", "/tmp/notes.txt"],
        )

    assert result is False
    assert "/tmp/archive.pdf.gz\n/tmp/notes.txt" in alert_class.call_args.kwargs["body"]
    alert_class.return_value.present.assert_called_once_with(parent)


def test_image_application_presents_before_scheduling_unavailable_dialog() -> None:
    events: list[str] = []
    window = SimpleNamespace(
        present=MagicMock(side_effect=lambda: events.append("present")),
        show_ocr_unavailable_dialog=MagicMock(),
    )
    application = SimpleNamespace(
        ocr_dependency=_OCR_UNAVAILABLE,
        get_active_window=MagicMock(return_value=None),
    )
    window_constructor = MagicMock(return_value=window)

    class FakeImageWindow:
        def __new__(cls, *args, **kwargs):
            return window_constructor(*args, **kwargs)

    with (
        patch("bigocrpdf.image_application.load_css"),
        patch("bigocrpdf.image_application.ImageOcrWindow", FakeImageWindow),
        patch(
            "bigocrpdf.image_application.GLib.idle_add",
            side_effect=lambda _callback: events.append("schedule"),
        ),
    ):
        ImageOcrApp.on_activate(application, application)

    window_constructor.assert_called_once_with(application, ocr_dependency=_OCR_UNAVAILABLE)
    assert events == ["present", "schedule"]


def test_image_application_does_not_hide_activation_errors() -> None:
    application = SimpleNamespace()

    with (
        patch("bigocrpdf.image_application.load_css", side_effect=RuntimeError("invalid CSS")),
        pytest.raises(RuntimeError, match="invalid CSS"),
    ):
        ImageOcrApp.on_activate(cast(Any, application), cast(Any, application))


def test_image_application_uses_window_scoped_clipboard_shortcuts() -> None:
    actions: list[str] = []
    app = SimpleNamespace(
        ocr_dependency=_OCR_AVAILABLE,
        on_about_action=MagicMock(),
        _on_quit_action=MagicMock(),
        add_action=lambda action: actions.append(action.name),
        set_accels_for_action=MagicMock(),
    )

    class FakeAction:
        def __init__(self, name: str) -> None:
            self.name = name

        def connect(self, *_args: object) -> None:
            pass

    with patch(
        "bigocrpdf.image_application.Gio.SimpleAction.new",
        side_effect=lambda name, _parameter_type: FakeAction(name),
    ):
        ImageOcrApp._setup_actions(cast(Any, app))

    assert actions == ["about", "quit"]
    assert app.set_accels_for_action.call_args_list == [
        call("app.quit", [config.SHORTCUTS.get("quit", "<Control>q")]),
        call("app.about", [config.SHORTCUTS.get("about", "F1")]),
        call(
            "win.paste-clipboard",
            [config.SHORTCUTS.get("paste-clipboard", "<Control>v")],
        ),
        call(
            "win.cancel-processing",
            [config.SHORTCUTS.get("cancel-processing", "Escape")],
        ),
    ]


@pytest.mark.parametrize(
    "window_type",
    (
        BigOcrPdfWindow,
        ImageOcrWindow,
    ),
)
def test_unavailable_dialog_contains_probe_cause_and_recovery_action(
    window_type: type,
) -> None:
    window = SimpleNamespace(
        ocr_dependency=_OCR_UNAVAILABLE,
        _ocr_unavailable_dialog=None,
        _on_ocr_unavailable_response=MagicMock(),
    )

    with (
        patch("bigocrpdf.ui.widgets._", side_effect=lambda text: text),
        patch("bigocrpdf.ui.widgets.Adw.AlertDialog") as alert_class,
    ):
        window_type.show_ocr_unavailable_dialog(window)
        window_type.show_ocr_unavailable_dialog(window)

    alert_class.assert_called_once()
    assert _OCR_UNAVAILABLE.error in alert_class.call_args.kwargs["body"]
    assert "restart the application" in alert_class.call_args.kwargs["body"]
    alert_class.return_value.present.assert_called_once_with(window)


@pytest.mark.parametrize(
    ("state", "expected_sensitive"),
    ((_OCR_AVAILABLE, True), (_OCR_UNAVAILABLE, False)),
)
def test_pdf_start_button_follows_resolved_ocr_state(
    state: OcrDependencyState,
    expected_sensitive: bool,
) -> None:
    button = MagicMock()
    header = SimpleNamespace(window=SimpleNamespace(ocr_dependency=state))

    with patch("bigocrpdf.ui.header_bar._", side_effect=lambda text: text):
        HeaderBar._apply_ocr_availability_to_button(header, button)

    button.set_sensitive.assert_called_once_with(expected_sensitive)


@pytest.mark.parametrize(
    ("state", "expected_started"),
    ((_OCR_AVAILABLE, True), (_OCR_UNAVAILABLE, False)),
)
def test_image_processing_allow_deny_stops_before_service_io(
    state: OcrDependencyState,
    expected_started: bool,
) -> None:
    config = OCRConfig(
        use_textline_cls=True,
        detection_full_resolution=True,
    )
    settings = SimpleNamespace(
        lang="en",
        load_settings=MagicMock(),
        _snapshot_ocr_config=MagicMock(return_value=config),
    )
    window = SimpleNamespace(
        ocr_dependency=state,
        show_ocr_unavailable_dialog=MagicMock(),
        _stack=MagicMock(),
        _copy_button=MagicMock(),
        _screen_capture_service=MagicMock(),
        _settings=settings,
        _on_processing_complete=MagicMock(),
    )
    window._require_ocr_available = MethodType(ImageOcrWindow._require_ocr_available, window)
    _bind_image_lifecycle(window)

    started = ImageOcrWindow._start_processing(window, "/tmp/scan.png")

    assert started is expected_started
    if expected_started:
        window._screen_capture_service.process_image_file.assert_called_once()
        call_kwargs = window._screen_capture_service.process_image_file.call_args.kwargs
        assert call_kwargs["config"] is config
        assert "language" not in call_kwargs
        settings.load_settings.assert_called_once_with()
        settings._snapshot_ocr_config.assert_called_once_with()
        window.show_ocr_unavailable_dialog.assert_not_called()
    else:
        window._screen_capture_service.process_image_file.assert_not_called()
        settings.load_settings.assert_not_called()
        settings._snapshot_ocr_config.assert_not_called()
        window._stack.set_visible_child_name.assert_not_called()
        window.show_ocr_unavailable_dialog.assert_called_once_with()


def test_image_processing_removes_owned_input_after_async_callback() -> None:
    config = OCRConfig()
    window = SimpleNamespace(
        _require_ocr_available=MagicMock(return_value=True),
        _stack=MagicMock(),
        _copy_button=MagicMock(),
        _screen_capture_service=MagicMock(),
        _settings=SimpleNamespace(
            load_settings=MagicMock(),
            _snapshot_ocr_config=MagicMock(return_value=config),
        ),
        _on_processing_complete=MagicMock(),
    )
    _bind_image_lifecycle(window)

    with patch("bigocrpdf.ui.image_ocr_window.remove_file") as remove_file:
        assert ImageOcrWindow._start_processing(
            window, "/tmp/paste.png", cleanup_path="/tmp/paste.png"
        )
        callback = window._screen_capture_service.process_image_file.call_args.kwargs["callback"]
        remove_file.assert_not_called()
        outcome = ImageOcrOutcome(ImageOcrStatus.SUCCESS, text="text")
        callback(outcome)

    remove_file.assert_called_once_with("/tmp/paste.png")
    window._on_processing_complete.assert_called_once_with(outcome)


def test_latest_image_processing_request_owns_the_visible_result() -> None:
    config = OCRConfig()
    window = SimpleNamespace(
        _require_ocr_available=MagicMock(return_value=True),
        _stack=MagicMock(),
        _copy_button=MagicMock(),
        _screen_capture_service=MagicMock(),
        _settings=SimpleNamespace(
            load_settings=MagicMock(),
            _snapshot_ocr_config=MagicMock(return_value=config),
        ),
        _on_processing_complete=MagicMock(),
        _processing_generation=0,
    )
    _bind_image_lifecycle(window)

    assert ImageOcrWindow._start_processing(window, "first.png")
    first_request = window._active_request
    first_callback = window._screen_capture_service.process_image_file.call_args.kwargs["callback"]
    assert ImageOcrWindow._start_processing(window, "second.png")
    second_callback = window._screen_capture_service.process_image_file.call_args.kwargs["callback"]

    new_outcome = ImageOcrOutcome(ImageOcrStatus.SUCCESS, text="new")
    second_callback(new_outcome)
    first_callback(ImageOcrOutcome(ImageOcrStatus.SUCCESS, text="stale"))

    window._on_processing_complete.assert_called_once_with(new_outcome)
    first_request.cancel.assert_called_once_with()


def test_newer_image_request_supersedes_pending_capture_result() -> None:
    config = OCRConfig()
    window = SimpleNamespace(
        _require_ocr_available=MagicMock(return_value=True),
        _restore_after_capture=MagicMock(),
        _on_capture_taken=MagicMock(),
        _on_processing_complete=MagicMock(),
        _screen_capture_service=MagicMock(),
        _settings=SimpleNamespace(
            load_settings=MagicMock(),
            _snapshot_ocr_config=MagicMock(return_value=config),
        ),
        _stack=MagicMock(),
        _copy_button=MagicMock(),
        _processing_generation=0,
    )
    _bind_image_lifecycle(window)

    assert ImageOcrWindow._trigger_capture(window) is False
    capture_request = window._active_request
    capture_callbacks = window._screen_capture_service.capture_screen_region.call_args.kwargs
    assert ImageOcrWindow._start_processing(window, "newer.png")
    image_callback = window._screen_capture_service.process_image_file.call_args.kwargs["callback"]

    capture_callbacks["on_processing"]()
    capture_callbacks["callback"](ImageOcrOutcome(ImageOcrStatus.SUCCESS, text="stale"))
    new_outcome = ImageOcrOutcome(ImageOcrStatus.SUCCESS, text="new")
    image_callback(new_outcome)

    window._on_capture_taken.assert_not_called()
    window._on_processing_complete.assert_called_once_with(new_outcome)
    capture_request.cancel.assert_called_once_with()


def test_image_started_during_capture_delay_prevents_stale_capture_start() -> None:
    window = SimpleNamespace(
        _require_ocr_available=MagicMock(return_value=True),
        _stack=MagicMock(),
        _copy_button=MagicMock(),
        _screen_capture_service=MagicMock(),
        _settings=SimpleNamespace(
            load_settings=MagicMock(),
            _snapshot_ocr_config=MagicMock(return_value=OCRConfig()),
        ),
        _on_processing_complete=MagicMock(),
        set_visible=MagicMock(),
        present=MagicMock(),
        _trigger_capture=MagicMock(return_value=False),
    )
    _bind_image_lifecycle(window)
    window._on_capture_delay = MethodType(ImageOcrWindow._on_capture_delay, window)
    scheduled = {}

    def fake_timeout_add(_delay, callback, generation):
        scheduled["callback"] = callback
        scheduled["generation"] = generation
        return 41

    with (
        patch(
            "bigocrpdf.ui.image_ocr_window.GLib.timeout_add",
            side_effect=fake_timeout_add,
        ),
        patch("bigocrpdf.ui.image_ocr_window.safe_remove_source"),
    ):
        ImageOcrWindow._on_new_capture_clicked(window)
        assert ImageOcrWindow._start_processing(window, "newer.png")
        scheduled["callback"](scheduled["generation"])

    window._trigger_capture.assert_not_called()
    window.set_visible.assert_has_calls([call(False), call(True)])


def test_stale_capture_cannot_restore_a_newer_hidden_capture() -> None:
    window = SimpleNamespace(
        _alive=True,
        _is_hidden_for_capture=True,
        _hidden_capture_generation=7,
        set_visible=MagicMock(),
        present=MagicMock(),
    )

    ImageOcrWindow._restore_after_capture(window, 6)

    assert window._is_hidden_for_capture is True
    assert window._hidden_capture_generation == 7
    window.set_visible.assert_not_called()


def test_image_window_close_cancels_owned_async_work_once() -> None:
    request = MagicMock()
    cancellable = MagicMock()
    window = SimpleNamespace(
        _alive=True,
        _processing_generation=4,
        _input_generation=2,
        _capture_delay_source_id=11,
        _focus_idle_source_id=12,
        _active_request=request,
        _input_cancellable=cancellable,
        _screen_capture_service=MagicMock(),
        _hidden_capture_generation=4,
        _is_hidden_for_capture=True,
        _set_cancel_enabled=MagicMock(),
        _save_window_size=MagicMock(),
    )
    window._remove_source = MethodType(ImageOcrWindow._remove_source, window)

    with patch("bigocrpdf.ui.image_ocr_window.safe_remove_source") as remove_source:
        ImageOcrWindow.prepare_close(window)
        ImageOcrWindow.prepare_close(window)

    assert window._alive is False
    request.cancel.assert_called_once_with()
    cancellable.cancel.assert_called_once_with()
    window._screen_capture_service.shutdown.assert_called_once_with(wait=False)
    window._save_window_size.assert_called_once_with()
    assert remove_source.call_args_list == [call(11), call(12)]
    assert ImageOcrWindow._is_current_operation(window, 5) is False


@pytest.mark.parametrize(("maximized", "fullscreen"), ((True, False), (False, True)))
def test_image_window_does_not_save_transient_toplevel_size(maximized, fullscreen) -> None:
    window = SimpleNamespace(
        is_maximized=MagicMock(return_value=maximized),
        is_fullscreen=MagicMock(return_value=fullscreen),
        get_width=MagicMock(),
        get_height=MagicMock(),
    )

    with patch("bigocrpdf.ui.image_ocr_window.get_config_manager") as get_config_manager:
        ImageOcrWindow._save_window_size(window)

    get_config_manager.assert_not_called()
    window.get_width.assert_not_called()
    window.get_height.assert_not_called()


def test_pdf_window_close_is_idempotent_and_continues_after_cleanup_failure() -> None:
    window = SimpleNamespace(
        _close_prepared=False,
        _save_window_size=MagicMock(),
        processing=SimpleNamespace(cleanup=MagicMock(side_effect=RuntimeError("failed"))),
        ui=SimpleNamespace(cleanup=MagicMock()),
        settings=SimpleNamespace(reset_processing_state=MagicMock()),
    )

    BigOcrPdfWindow.prepare_close(window)
    BigOcrPdfWindow.prepare_close(window)

    window._save_window_size.assert_called_once_with()
    window.processing.cleanup.assert_called_once_with()
    window.ui.cleanup.assert_called_once_with()
    window.settings.reset_processing_state.assert_called_once_with()


@pytest.mark.parametrize(("maximized", "fullscreen"), ((True, False), (False, True)))
def test_pdf_window_does_not_save_transient_toplevel_size(maximized, fullscreen) -> None:
    window = SimpleNamespace(
        is_maximized=MagicMock(return_value=maximized),
        is_fullscreen=MagicMock(return_value=fullscreen),
        get_width=MagicMock(),
        get_height=MagicMock(),
    )

    with patch("bigocrpdf.window.get_config_manager") as get_config_manager:
        BigOcrPdfWindow._save_window_size(window)

    get_config_manager.assert_not_called()
    window.get_width.assert_not_called()
    window.get_height.assert_not_called()


def test_pdf_window_close_request_prepares_resources_and_allows_close() -> None:
    window = SimpleNamespace(prepare_close=MagicMock())

    assert BigOcrPdfWindow._on_close_request(window, MagicMock()) is False
    window.prepare_close.assert_called_once_with()


def test_pdf_application_prepares_main_window_resources() -> None:
    class FakeMainWindow:
        def __init__(self) -> None:
            self.prepare_close = MagicMock()

    main_window = FakeMainWindow()
    app = SimpleNamespace(get_windows=MagicMock(return_value=[main_window]))

    with (
        patch("bigocrpdf.application.BigOcrPdfWindow", FakeMainWindow),
        patch("bigocrpdf.application.ImageOcrWindow", type("FakeImageWindow", (), {})),
        patch(
            "bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow",
            type("FakeEditorWindow", (), {}),
        ),
    ):
        BigOcrPdfApp._prepare_windows_for_shutdown(app)

    main_window.prepare_close.assert_called_once_with()


def test_image_application_quit_prepares_windows_before_quitting() -> None:
    app = SimpleNamespace(
        _prepare_image_windows_for_shutdown=MagicMock(),
        quit=MagicMock(),
    )

    ImageOcrApp._on_quit_action(app)

    assert app._prepare_image_windows_for_shutdown.call_args_list == [call()]
    app.quit.assert_called_once_with()


def test_pdf_application_quit_prepares_windows_before_quitting() -> None:
    app = SimpleNamespace(
        _prepare_windows_for_shutdown=MagicMock(),
        quit=MagicMock(),
    )

    BigOcrPdfApp._on_quit_action(app)

    app._prepare_windows_for_shutdown.assert_called_once_with()
    app.quit.assert_called_once_with()


def test_pdf_application_shutdown_prepares_windows_before_releasing_renderer() -> None:
    events: list[str] = []
    app = SimpleNamespace(
        _prepare_windows_for_shutdown=MagicMock(side_effect=lambda: events.append("windows"))
    )

    with patch(
        "bigocrpdf.application.shutdown_thumbnail_renderer",
        side_effect=lambda **_kwargs: events.append("renderer"),
    ) as shutdown_renderer:
        BigOcrPdfApp._on_shutdown(app)

    app._prepare_windows_for_shutdown.assert_called_once_with()
    shutdown_renderer.assert_called_once_with(wait=True)
    assert events == ["windows", "renderer"]


def test_pdf_application_prepares_editor_window_thumbnail_requests() -> None:
    class FakeEditorWindow:
        def __init__(self) -> None:
            self._prepare_close = MagicMock()

    editor = FakeEditorWindow()
    app = SimpleNamespace(get_windows=MagicMock(return_value=[editor]))
    with (
        patch(
            "bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow",
            FakeEditorWindow,
        ),
        patch("bigocrpdf.application.ImageOcrWindow", type("FakeImageWindow", (), {})),
    ):
        BigOcrPdfApp._prepare_windows_for_shutdown(app)

    editor._prepare_close.assert_called_once_with()


def test_image_clipboard_texture_save_failure_removes_owned_png() -> None:
    window = SimpleNamespace(_alive=True, _toast_overlay=MagicMock())
    window._is_current_input_operation = MethodType(
        ImageOcrWindow._is_current_input_operation,
        window,
    )
    window._finish_input_operation = MethodType(
        ImageOcrWindow._finish_input_operation,
        window,
    )
    window._save_clipboard_texture_worker = MethodType(
        ImageOcrWindow._save_clipboard_texture_worker,
        window,
    )
    window._on_clipboard_texture_saved = MethodType(
        ImageOcrWindow._on_clipboard_texture_saved,
        window,
    )
    clipboard = MagicMock()
    texture = clipboard.read_texture_finish.return_value
    texture.get_width.return_value = 100
    texture.get_height.return_value = 100
    texture.save_to_png.side_effect = OSError("full")

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            assert daemon is True
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    with (
        patch(
            "bigocrpdf.ui.image_ocr_window.mkstemp",
            return_value=(10, "/tmp/paste.png"),
        ),
        patch("os.close"),
        patch(
            "bigocrpdf.ui.image_ocr_window.threading.Thread",
            ImmediateThread,
        ),
        patch(
            "bigocrpdf.ui.image_ocr_window.GLib.idle_add",
            side_effect=lambda callback, *args: callback(*args),
        ),
        patch("bigocrpdf.ui.image_ocr_window.remove_file") as remove_file,
    ):
        ImageOcrWindow._on_clipboard_texture_ready(window, clipboard, MagicMock())

    remove_file.assert_called_once_with("/tmp/paste.png")


def test_oversized_clipboard_texture_is_rejected_before_encoding() -> None:
    window = SimpleNamespace(_alive=True, _toast_overlay=MagicMock())
    window._is_current_input_operation = MethodType(
        ImageOcrWindow._is_current_input_operation,
        window,
    )
    window._finish_input_operation = MethodType(
        ImageOcrWindow._finish_input_operation,
        window,
    )
    clipboard = MagicMock()
    texture = clipboard.read_texture_finish.return_value
    texture.get_width.return_value = 6000
    texture.get_height.return_value = 5000

    with patch("bigocrpdf.ui.image_ocr_window.mkstemp") as make_temp:
        ImageOcrWindow._on_clipboard_texture_ready(window, clipboard, MagicMock())

    make_temp.assert_not_called()
    texture.save_to_png.assert_not_called()


def test_clipboard_texture_encoding_starts_in_worker_thread() -> None:
    window = SimpleNamespace(_alive=True, _toast_overlay=MagicMock())
    window._is_current_input_operation = MethodType(
        ImageOcrWindow._is_current_input_operation,
        window,
    )
    window._finish_input_operation = MethodType(
        ImageOcrWindow._finish_input_operation,
        window,
    )
    window._save_clipboard_texture_worker = MethodType(
        ImageOcrWindow._save_clipboard_texture_worker,
        window,
    )
    clipboard = MagicMock()
    texture = clipboard.read_texture_finish.return_value
    texture.get_width.return_value = 100
    texture.get_height.return_value = 100

    with (
        patch(
            "bigocrpdf.ui.image_ocr_window.mkstemp",
            return_value=(10, "/tmp/paste.png"),
        ),
        patch("bigocrpdf.ui.image_ocr_window.os.close"),
        patch("bigocrpdf.ui.image_ocr_window.threading.Thread") as thread_class,
    ):
        ImageOcrWindow._on_clipboard_texture_ready(window, clipboard, MagicMock())

    thread_class.assert_called_once()
    thread_class.return_value.start.assert_called_once_with()
    texture.save_to_png.assert_not_called()


def test_clipboard_encoder_keeps_only_the_latest_pending_texture() -> None:
    pending_threads = []

    class DeferredThread:
        def __init__(self, *, target, args, daemon):
            assert args == ()
            assert daemon is True
            self.target = target
            pending_threads.append(self)

        def start(self):
            return None

    window = SimpleNamespace(_alive=True)
    first_texture = MagicMock()
    second_texture = MagicMock()
    first_cancellable = MagicMock()
    second_cancellable = MagicMock()

    with patch(
        "bigocrpdf.ui.image_ocr_window.threading.Thread",
        DeferredThread,
    ):
        assert ImageOcrWindow._queue_clipboard_texture(
            window,
            first_texture,
            1,
            first_cancellable,
        )
        assert ImageOcrWindow._queue_clipboard_texture(
            window,
            second_texture,
            2,
            second_cancellable,
        )

    assert len(pending_threads) == 1
    assert window._clipboard_encode_pending.texture is second_texture
    first_cancellable.cancel.assert_called_once_with()
    second_cancellable.cancel.assert_not_called()


def test_clipboard_encoder_start_failure_rolls_back_and_allows_retry() -> None:
    starts = 0
    encoded = []

    class FlakyThread:
        def __init__(self, *, target, args, daemon):
            assert args == ()
            assert daemon is True
            self.target = target

        def start(self):
            nonlocal starts
            starts += 1
            if starts == 1:
                raise RuntimeError("thread unavailable")
            self.target()

    window = SimpleNamespace(_alive=True)
    window._save_clipboard_texture_worker = encoded.append
    first_texture = MagicMock()
    second_texture = MagicMock()

    with patch("bigocrpdf.ui.image_ocr_window.threading.Thread", FlakyThread):
        with pytest.raises(RuntimeError, match="thread unavailable"):
            ImageOcrWindow._queue_clipboard_texture(
                window,
                first_texture,
                1,
                MagicMock(),
            )

        assert window._clipboard_encode_thread is None
        assert window._clipboard_encode_pending is None

        assert ImageOcrWindow._queue_clipboard_texture(
            window,
            second_texture,
            2,
            MagicMock(),
        )

    assert starts == 2
    assert [job.texture for job in encoded] == [second_texture]


def test_capture_start_failure_restores_hidden_window() -> None:
    window = SimpleNamespace(
        _require_ocr_available=MagicMock(return_value=True),
        _restore_after_capture=MagicMock(),
        _show_error=MagicMock(),
        _stack=MagicMock(),
        _copy_button=MagicMock(),
        _settings=SimpleNamespace(
            load_settings=MagicMock(side_effect=OSError("settings unavailable")),
            _snapshot_ocr_config=MagicMock(),
        ),
        _screen_capture_service=MagicMock(),
        _processing_generation=3,
    )
    _bind_image_lifecycle(window)
    window._processing_generation = 3

    with patch("bigocrpdf.ui.image_ocr_window._", side_effect=lambda text: text):
        assert ImageOcrWindow._trigger_capture(window, 3) is False

    window._restore_after_capture.assert_called_once_with(3)
    window._stack.set_visible_child_name.assert_called_once_with("welcome")
    window._set_cancel_enabled.assert_called_once_with(False)
    window._show_error.assert_called_once()
    window._screen_capture_service.capture_screen_region.assert_not_called()


def test_image_capture_deny_keeps_window_visible_and_does_not_schedule_capture() -> None:
    window = SimpleNamespace(
        _require_ocr_available=MagicMock(return_value=False),
        _is_hidden_for_capture=False,
        set_visible=MagicMock(),
    )

    with patch("bigocrpdf.ui.image_ocr_window.GLib.timeout_add") as timeout_add:
        ImageOcrWindow._on_new_capture_clicked(window)

    assert window._is_hidden_for_capture is False
    window.set_visible.assert_not_called()
    timeout_add.assert_not_called()


def test_image_clipboard_deny_does_not_read_clipboard() -> None:
    window = SimpleNamespace(_require_ocr_available=MagicMock(return_value=False))

    with patch("bigocrpdf.ui.image_ocr_window.Gdk.Display.get_default") as get_display:
        ImageOcrWindow.paste_from_clipboard(window)

    get_display.assert_not_called()


def test_image_window_exposes_paste_as_a_window_action() -> None:
    cancel_action = MagicMock()
    paste_action = MagicMock()
    window = SimpleNamespace(
        ocr_dependency=_OCR_AVAILABLE,
        _on_cancel_processing=MagicMock(),
        paste_from_clipboard=MagicMock(),
        add_action=MagicMock(),
    )

    with patch(
        "bigocrpdf.ui.image_ocr_window.Gio.SimpleAction.new",
        side_effect=[cancel_action, paste_action],
    ) as new_action:
        ImageOcrWindow._setup_window_actions(window)

    assert [call.args[0] for call in new_action.call_args_list] == [
        "cancel-processing",
        "paste-clipboard",
    ]
    paste_action.set_enabled.assert_called_once_with(True)
    assert window.add_action.call_args_list == [call(cancel_action), call(paste_action)]


def test_image_clipboard_uri_stream_reads_and_closes_asynchronously() -> None:
    window = SimpleNamespace(
        _alive=True,
        _toast_overlay=MagicMock(),
        _SUPPORTED_IMAGE_EXTENSIONS=frozenset((".png",)),
        _MAX_CLIPBOARD_URI_BYTES=64 * 1024,
    )
    window._is_current_input_operation = MethodType(
        ImageOcrWindow._is_current_input_operation,
        window,
    )
    window._finish_input_operation = MethodType(
        ImageOcrWindow._finish_input_operation,
        window,
    )
    window._read_clipboard_uri_chunk = MethodType(
        ImageOcrWindow._read_clipboard_uri_chunk,
        window,
    )
    window._on_clipboard_uri_chunk_ready = MethodType(
        ImageOcrWindow._on_clipboard_uri_chunk_ready,
        window,
    )
    window._close_stream_async = ImageOcrWindow._close_stream_async
    clipboard = MagicMock()
    stream = MagicMock()
    clipboard.read_finish.return_value = (stream, "text/uri-list")

    ImageOcrWindow._on_clipboard_uri_ready(window, clipboard, MagicMock())
    stream.read_bytes_async.assert_called_once()
    callback = stream.read_bytes_async.call_args.args[3]
    callback_args = stream.read_bytes_async.call_args.args[4:]
    stream.read_bytes_finish.return_value.get_data.return_value = b""
    callback(stream, MagicMock(), *callback_args)

    stream.close_async.assert_called_once()
    stream.read_bytes.assert_not_called()


def test_image_copy_reports_failure_when_clipboard_rejects_content() -> None:
    clipboard = MagicMock()
    clipboard.set_content.return_value = False
    text_buffer = MagicMock()
    text_buffer.get_bounds.return_value = (MagicMock(), MagicMock())
    text_buffer.get_text.return_value = "recognized text"
    window = SimpleNamespace(
        _text_buffer=text_buffer,
        _toast_overlay=MagicMock(),
    )

    with (
        patch(
            "bigocrpdf.ui.image_ocr_window.get_default_clipboard",
            return_value=clipboard,
        ),
        patch(
            "bigocrpdf.ui.image_ocr_window.Adw.Toast",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch("bigocrpdf.ui.image_ocr_window._", side_effect=lambda text: text),
    ):
        ImageOcrWindow._on_copy_clicked(window, MagicMock())

    toast = window._toast_overlay.add_toast.call_args.args[0]
    assert toast.title == "Could not copy text to the clipboard"
    clipboard.set_content.assert_called_once()
