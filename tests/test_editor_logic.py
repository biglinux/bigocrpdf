import os
import sys
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch


# Define dummy classes for Gtk/Adw to avoid inheritance issues
class DummyWindow:
    def __init__(self, **kwargs):
        pass

    def set_content(self, *args):
        pass

    def set_default_size(self, *args):
        pass

    def set_title(self, *args):
        pass

    def set_modal(self, *args):
        pass

    def connect(self, *args):
        pass

    def add_controller(self, *args):
        pass

    def insert_action_group(self, *args):
        pass

    def close(self):
        pass


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
            print("Save behavior test passed!")

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

    def test_failed_merge_keeps_editor_open_and_modified(self):
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

            doc = PDFDocument(path="test.pdf", total_pages=1)
            doc.pages[0].source_file = "added.pdf"
            doc.modified = True
            window._document = doc
            window._save_merged_pdf = MagicMock(return_value=False)
            window._close_window = MagicMock()

            window._on_ok_clicked(cast(Any, MagicMock()))

            window._save_merged_pdf.assert_called_once_with("test.pdf")
            mock_save_callback.assert_not_called()
            self.assertTrue(doc.modified)
            window._close_window.assert_not_called()

    def test_flip_is_materialized_before_the_document_enters_ocr(self):
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
            save_callback = MagicMock()
            window = PDFEditorWindow(self.mock_app, "test.pdf", on_save_callback=save_callback)

        document = PDFDocument(path="test.pdf", total_pages=1)
        document.pages[0].flip_horizontal = True
        document.modified = True
        window._document = document
        window._save_merged_pdf = MagicMock(return_value=True)

        saved = window._save_and_callback()

        self.assertTrue(saved)
        window._save_merged_pdf.assert_called_once_with("test.pdf")
        save_callback.assert_not_called()

    def test_reordered_pages_are_materialized_before_the_document_enters_ocr(self):
        save_callback = MagicMock()
        window = PDFEditorWindow.__new__(PDFEditorWindow)
        window._on_save_callback = save_callback
        window._original_page_count = 2
        window._show_error = MagicMock()
        window._save_merged_pdf = MagicMock(return_value=True)
        document = PDFDocument(path="test.pdf", total_pages=2)
        document.pages[0].position = 1
        document.pages[1].position = 0
        window._document = document

        saved = window._save_and_callback()

        self.assertTrue(saved)
        window._save_merged_pdf.assert_called_once_with("test.pdf")
        save_callback.assert_not_called()

    def test_excluded_foreign_page_is_materialized_before_ocr(self):
        save_callback = MagicMock()
        window = PDFEditorWindow.__new__(PDFEditorWindow)
        window._on_save_callback = save_callback
        window._original_page_count = 1
        window._show_error = MagicMock()
        window._save_merged_pdf = MagicMock(return_value=True)
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

        saved = window._save_and_callback()

        self.assertTrue(saved)
        window._save_merged_pdf.assert_called_once_with("test.pdf")
        save_callback.assert_not_called()

    def test_save_dialog_keeps_editor_open_when_save_fails(self):
        window = MagicMock()
        window._save_and_callback.return_value = False

        PDFEditorWindow._on_save_dialog_response(window, MagicMock(), "save")

        window._save_and_callback.assert_called_once_with()
        window._close_window.assert_not_called()

    def test_failed_merge_preserves_modifications_and_reports_error(self):
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
            window = PDFEditorWindow(self.mock_app, "test.pdf", on_save_callback=MagicMock())

        doc = PDFDocument(path="test.pdf", total_pages=1)
        doc.modified = True
        window._document = doc
        window._show_error = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            fd, temp_path = tempfile.mkstemp(suffix=".pdf", dir=temp_dir)
            with (
                patch(
                    "bigocrpdf.utils.temp_manager.mkstemp",
                    return_value=(fd, temp_path),
                ),
                patch(
                    "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
                    return_value=False,
                ),
                patch(
                    "bigocrpdf.utils.temp_manager.remove_file",
                    wraps=lambda path: os.remove(path),
                ) as remove_file,
            ):
                saved = window._save_merged_pdf("test.pdf")

            remove_file.assert_called_once_with(temp_path)
            self.assertFalse(os.path.exists(temp_path))

        self.assertFalse(saved)
        self.assertTrue(doc.modified)
        window._show_error.assert_called_once_with("Failed to merge PDF pages.")

    def test_rejected_merged_save_callback_removes_temp_and_preserves_changes(self):
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
        window._show_error = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            fd, temp_path = tempfile.mkstemp(suffix=".pdf", dir=temp_dir)
            with (
                patch(
                    "bigocrpdf.utils.temp_manager.mkstemp",
                    return_value=(fd, temp_path),
                ),
                patch(
                    "bigocrpdf.ui.pdf_editor.page_operations.apply_changes_to_pdf",
                    return_value=True,
                ),
                patch(
                    "bigocrpdf.utils.temp_manager.remove_file",
                    wraps=lambda path: os.remove(path),
                ) as remove_file,
            ):
                saved = window._save_merged_pdf("test.pdf")

            remove_file.assert_called_once_with(temp_path)
            self.assertFalse(os.path.exists(temp_path))

        self.assertFalse(saved)
        self.assertTrue(document.modified)
        window._show_error.assert_called_once_with("Error saving changes.")

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

            # Verify callback NOT called
            mock_save_callback.assert_not_called()
            # Verify modifications cleared (explicit discard)
            self.assertFalse(doc.modified)
            print("Back/Discard behavior test passed!")

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
