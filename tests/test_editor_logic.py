import os
import sys
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch


# Stands in for Adw.Window so PDFEditorWindow can subclass it without GTK.
# Public widget calls the tests do not observe are no-ops; private names stay
# strict so an attribute the window forgot to set still raises.
class DummyWindow:
    def __init__(self, **kwargs):
        pass

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


# Save originals and mock Gtk/Adw before importing modules
_MOCKED_MODULES = [
    "gi",
    "gi.repository",
    "bigocrpdf.utils.logger",
    "bigocrpdf.utils.i18n",
    "bigocrpdf.ui.pdf_editor.thumbnail_renderer",
]
_saved_modules = {m: sys.modules.get(m) for m in _MOCKED_MODULES}
_EDITOR_MODULE_PREFIX = "bigocrpdf.ui.pdf_editor."
_EDITOR_WINDOW_MODULE = f"{_EDITOR_MODULE_PREFIX}editor_window"
_saved_editor_modules = {
    name: module for name, module in sys.modules.items() if name.startswith(_EDITOR_MODULE_PREFIX)
}

mock_gi = MagicMock()
mock_adw = MagicMock()
mock_adw.Window = DummyWindow
mock_adw.ToolbarView = MagicMock()
mock_adw.HeaderBar = MagicMock()

mock_gtk = MagicMock()
mock_gtk.Box = MagicMock()
mock_gtk.Button = MagicMock()

sys.modules["gi"] = MagicMock()
sys.modules["gi.repository"] = mock_gi
sys.modules["gi.repository"].Adw = mock_adw
sys.modules["gi.repository"].Gtk = mock_gtk
sys.modules["gi.repository"].Gdk = MagicMock()
sys.modules["gi.repository"].Gio = MagicMock()
sys.modules["gi.repository"].GLib = MagicMock()
sys.modules["gi.repository"].GObject = MagicMock()

# Mock internal modules
sys.modules["bigocrpdf.utils.logger"] = MagicMock()
sys.modules["bigocrpdf.utils.i18n"] = MagicMock()
sys.modules["bigocrpdf.utils.i18n"]._ = lambda x: x
sys.modules["bigocrpdf.ui.pdf_editor.thumbnail_renderer"] = MagicMock()

# Now import the modules to test
from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow, requires_materialization
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument

# Restore original modules to avoid contaminating other test files
for _mod_name, _original in _saved_modules.items():
    if _original is not None:
        sys.modules[_mod_name] = _original
    else:
        sys.modules.pop(_mod_name, None)
for _mod_name in tuple(sys.modules):
    if not _mod_name.startswith(_EDITOR_MODULE_PREFIX):
        continue
    if _mod_name == _EDITOR_WINDOW_MODULE:
        continue
    if _mod_name in _saved_editor_modules:
        sys.modules[_mod_name] = _saved_editor_modules[_mod_name]
    else:
        sys.modules.pop(_mod_name, None)
del (
    _saved_modules,
    _saved_editor_modules,
    _MOCKED_MODULES,
    _EDITOR_MODULE_PREFIX,
    _EDITOR_WINDOW_MODULE,
)


class TestEditorWindowLogic(unittest.TestCase):
    def setUp(self):
        self.mock_app = MagicMock()

        # Setup page grid mock
        self.mock_grid = MagicMock()
        self.mock_status_bar = MagicMock()

        # Patch PageGrid and other UI components
        patcher = patch(
            "bigocrpdf.ui.pdf_editor.editor_window.PageGrid", return_value=self.mock_grid
        )
        self.addCleanup(patcher.stop)
        self.mock_page_grid_cls = patcher.start()

    def test_save_behavior(self):
        # Verify OK button saves and clears modifications
        with (
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_ui"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_actions"),
            patch(
                "bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_keyboard_shortcuts"
            ),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_drag_drop"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._load_document"),
            patch("os.path.basename", return_value="test.pdf"),
        ):
            mock_save_callback = MagicMock()
            window = PDFEditorWindow(self.mock_app, "test.pdf", on_save_callback=mock_save_callback)

            # Setup doc with modifications
            doc = PDFDocument(path="test.pdf", total_pages=1)
            doc.modified = True
            window._document = doc
            window._close_window = MagicMock()

            # Call OK
            window._on_ok_clicked(cast(Any, MagicMock()))

            # Verify callback called
            mock_save_callback.assert_called_once()
            # Verify modifications cleared
            self.assertFalse(doc.modified)
            window._close_window.assert_called_once_with()

    def test_rejected_save_callback_keeps_editor_open_and_modified(self):
        with (
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_ui"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_actions"),
            patch(
                "bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_keyboard_shortcuts"
            ),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_drag_drop"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._load_document"),
            patch("os.path.basename", return_value="test.pdf"),
        ):
            save_callback = MagicMock(return_value=False)
            window = PDFEditorWindow(self.mock_app, "test.pdf", on_save_callback=save_callback)

        document = PDFDocument(path="test.pdf", total_pages=1)
        document.modified = True
        window._document = document
        window._close_window = MagicMock()
        window._show_error = MagicMock()

        window._on_ok_clicked(cast(Any, MagicMock()))

        save_callback.assert_called_once_with(document)
        self.assertTrue(document.modified)
        window._close_window.assert_not_called()
        window._show_error.assert_called_once_with("Error saving changes.")

    def _editor(self, save_callback=None, standalone=False, save_in_place=False):
        with (
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_ui"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_actions"),
            patch(
                "bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_keyboard_shortcuts"
            ),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_drag_drop"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._load_document"),
            patch("os.path.basename", return_value="test.pdf"),
        ):
            window = PDFEditorWindow(
                self.mock_app,
                "test.pdf",
                on_save_callback=save_callback,
                standalone=standalone,
                save_in_place=save_in_place,
            )
        for name in (
            "_sidebar_box",
            "_page_grid",
            "_add_button_header",
            "_apply_button",
            "_show_busy",
            "_end_busy",
            "_show_info",
            "_show_error",
            "_close_window",
            "_save_editor_window_size",
        ):
            setattr(window, name, MagicMock())
        return window

    def _merge(self, save_callback, *, write_error=None, close_during_save=False):
        """Apply a merge with the worker held until the main-loop checks ran."""
        window = self._editor(save_callback)
        document = PDFDocument(path="test.pdf", total_pages=1)
        document.pages[0].source_file = "added.pdf"
        document.modified = True
        window._document = document
        jobs = []
        temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        fd, temp_path = tempfile.mkstemp(suffix=".pdf", dir=temp_dir)

        with (
            patch(
                "bigocrpdf.ui.pdf_editor.page_thumbnail_export.run_in_worker",
                side_effect=lambda work, done: jobs.append((work, done)),
            ),
            patch(
                "bigocrpdf.ui.pdf_editor.page_operations.write_changes_to_pdf",
                side_effect=write_error,
            ) as write,
            patch("bigocrpdf.utils.temp_manager.mkstemp", return_value=(fd, temp_path)),
        ):
            window._on_ok_clicked(None)
            window._on_ok_clicked(None)

            self.assertEqual(len(jobs), 1)
            write.assert_not_called()
            save_callback.assert_not_called()
            window._close_window.assert_not_called()
            window._apply_button.set_sensitive.assert_called_with(False)
            window._page_grid.set_sensitive.assert_called_with(False)
            window._show_busy.assert_called_once_with("Saving…")
            if close_during_save:
                self.assertTrue(window._on_close_request(window))
                window._close_window.assert_not_called()

            work, done = jobs[0]
            done(work())

        snapshot, written_path = write.call_args.args
        self.assertIsNot(snapshot, document)
        self.assertEqual(written_path, temp_path)
        window._apply_button.set_sensitive.assert_called_with(True)
        window._end_busy.assert_called_once_with()
        return window, document, temp_path

    def test_merge_writes_in_worker_then_closes(self):
        save_callback = MagicMock()
        window, document, temp_path = self._merge(save_callback, close_during_save=True)

        merged = save_callback.call_args.args[0]
        self.assertEqual(merged.path, temp_path)
        self.assertEqual(merged.total_pages, 1)
        self.assertFalse(document.modified)
        window._close_window.assert_called_once_with()
        window._show_error.assert_not_called()

    def test_failed_merge_reports_cause_and_stays_open(self):
        save_callback = MagicMock()
        window, document, temp_path = self._merge(
            save_callback,
            write_error=OSError("No space left on device"),
            close_during_save=True,
        )

        save_callback.assert_not_called()
        self.assertTrue(document.modified)
        self.assertFalse(os.path.exists(temp_path))
        window._close_window.assert_not_called()
        self.assertIn("No space left on device", window._show_error.call_args.args[0])
        self.assertFalse(window._saving)
        self.assertFalse(window._close_after_save)

    def test_rejected_merge_removes_temp_and_stays_open(self):
        window, document, temp_path = self._merge(MagicMock(return_value=False))

        self.assertTrue(document.modified)
        self.assertFalse(os.path.exists(temp_path))
        window._close_window.assert_not_called()
        window._show_error.assert_called_once_with("Error saving changes.")

    def test_close_during_save_as_waits_for_it(self):
        window = self._editor(standalone=True)
        document = PDFDocument(path="test.pdf", total_pages=1)
        document.modified = True
        window._document = document
        dialog = MagicMock()
        dialog.save_finish.return_value.get_path.return_value = "/output/edited.pdf"
        jobs = []

        with (
            patch(
                "bigocrpdf.ui.pdf_editor.page_thumbnail_export.run_in_worker",
                side_effect=lambda work, done: jobs.append((work, done)),
            ),
            patch("bigocrpdf.ui.pdf_editor.page_operations.publish_changes_to_pdf") as publish,
        ):
            window._on_save_as_response(dialog, MagicMock(), False)

            publish.assert_not_called()
            self.assertTrue(window._on_close_request(window))
            window._close_window.assert_not_called()

            work, done = jobs[0]
            done(work())

        self.assertEqual(publish.call_args.args[1], "/output/edited.pdf")
        self.assertFalse(document.modified)
        window._show_info.assert_called_once_with("Saved: edited.pdf")
        window._close_window.assert_called_once_with()

    def _apply_in_place(self, publish_error=None):
        """Apply to the opened file with the worker held until the main-loop checks ran."""
        window = self._editor(save_in_place=True)
        document = PDFDocument(path="test.pdf", total_pages=1)
        document.pages[0].rotation = 90
        document.modified = True
        window._document = document
        jobs = []

        with (
            patch(
                "bigocrpdf.ui.pdf_editor.page_thumbnail_export.run_in_worker",
                side_effect=lambda work, done: jobs.append((work, done)),
            ),
            patch(
                "bigocrpdf.ui.pdf_editor.page_operations.publish_changes_to_pdf",
                side_effect=publish_error,
            ) as publish,
        ):
            window._on_ok_clicked(None)

            publish.assert_not_called()
            self.assertEqual(len(jobs), 1)
            self.assertTrue(window._on_close_request(window))
            window._close_window.assert_not_called()

            work, done = jobs[0]
            done(work())

        snapshot, target = publish.call_args.args
        self.assertIsNot(snapshot, document)
        self.assertEqual(target, "test.pdf")
        return window, document

    def test_in_place_apply_saves_then_closes(self):
        window, document = self._apply_in_place()

        self.assertFalse(document.modified)
        window._close_window.assert_called_once_with()
        window._show_error.assert_not_called()

    def test_failed_in_place_apply_stays_open(self):
        window, document = self._apply_in_place(OSError("Permission denied"))

        self.assertTrue(document.modified)
        window._close_window.assert_not_called()
        self.assertIn("Permission denied", window._show_error.call_args.args[0])

    def test_flipped_page_needs_merge(self):
        save_callback = MagicMock()
        window = self._editor(save_callback)
        window._apply_merged = MagicMock()
        document = PDFDocument(path="test.pdf", total_pages=1)
        document.pages[0].flip_horizontal = True
        document.modified = True
        window._document = document

        window._on_ok_clicked(None)

        window._apply_merged.assert_called_once_with(document)
        save_callback.assert_not_called()

    def test_reordered_pages_need_merge(self):
        save_callback = MagicMock()
        window = self._editor(save_callback)
        window._original_page_count = 2
        window._apply_merged = MagicMock()
        document = PDFDocument(path="test.pdf", total_pages=2)
        document.pages[0].position = 1
        document.pages[1].position = 0
        window._document = document

        window._on_ok_clicked(None)

        window._apply_merged.assert_called_once_with(document)
        save_callback.assert_not_called()

    def test_excluded_foreign_page_needs_merge(self):
        save_callback = MagicMock()
        window = self._editor(save_callback)
        window._original_page_count = 1
        window._apply_merged = MagicMock()
        document = PDFDocument(path="test.pdf", total_pages=1)
        document.pages.append(
            PageState(
                page_number=1,
                position=1,
                source_file="foreign.pdf",
                included_for_ocr=False,
            )
        )
        document.total_pages = 2
        window._document = document

        window._on_ok_clicked(None)

        window._apply_merged.assert_called_once_with(document)
        save_callback.assert_not_called()

    def test_unsaved_changes_save_follows_editor_mode(self):
        for standalone in (False, True):
            window = MagicMock()
            window._standalone = standalone

            PDFEditorWindow._on_save_dialog_response(window, MagicMock(), "save")

            if standalone:
                window._save_as.assert_called_once_with(close_after=True)
                window._apply_and_close.assert_not_called()
            else:
                window._apply_and_close.assert_called_once_with()
                window._save_as.assert_not_called()
            window._close_window.assert_not_called()

    def test_back_behavior(self):
        # Verify Back button discards and clears modifications
        with (
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_ui"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_actions"),
            patch(
                "bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_keyboard_shortcuts"
            ),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._setup_drag_drop"),
            patch("bigocrpdf.ui.pdf_editor.editor_window.PDFEditorWindow._load_document"),
            patch("os.path.basename", return_value="test.pdf"),
        ):
            mock_save_callback = MagicMock()
            window = PDFEditorWindow(self.mock_app, "test.pdf", on_save_callback=mock_save_callback)

            # Setup doc with modifications
            doc = PDFDocument(path="test.pdf", total_pages=1)
            doc.modified = True
            window._document = doc

            # Call Back
            window._on_back_clicked(cast(Any, MagicMock()))

            # The save callback stays uncalled
            mock_save_callback.assert_not_called()
            # Verify modifications cleared (explicit discard)
            self.assertFalse(doc.modified)

    def test_undo_stack_ignores_consecutive_duplicate_snapshots(self):
        window = PDFEditorWindow.__new__(PDFEditorWindow)
        window._document = PDFDocument(path="test.pdf", total_pages=1)
        window._undo_stack = []

        window._push_undo()
        window._push_undo()

        self.assertEqual(len(window._undo_stack), 1)


class TestMaterializationDecision(unittest.TestCase):
    def test_rotation_soft_deletion_and_ocr_exclusion_remain_metadata_only(self):
        document = PDFDocument(path="test.pdf", total_pages=3)
        document.pages[0].rotation = 90
        document.pages[1].deleted = True
        document.pages[2].included_for_ocr = False

        self.assertFalse(requires_materialization(document, "test.pdf", 3))

    def test_missing_original_page_requires_materialization(self):
        document = PDFDocument(path="test.pdf", total_pages=2)
        document.pages.pop()
        document.total_pages = 1

        self.assertTrue(requires_materialization(document, "test.pdf", 2))

    def test_equivalent_symlink_source_does_not_require_materialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original = os.path.join(temp_dir, "source.pdf")
            alias = os.path.join(temp_dir, "alias.pdf")
            with open(original, "wb") as stream:
                stream.write(b"%PDF-1.7\n")
            os.symlink(original, alias)
            document = PDFDocument(path=original, total_pages=1)
            document.pages[0].source_file = alias

            self.assertFalse(requires_materialization(document, original, 1))

    def test_flip_on_excluded_page_does_not_require_materialization(self):
        document = PDFDocument(path="test.pdf", total_pages=1)
        document.pages[0].included_for_ocr = False
        document.pages[0].flip_horizontal = True

        self.assertFalse(requires_materialization(document, "test.pdf", 1))
