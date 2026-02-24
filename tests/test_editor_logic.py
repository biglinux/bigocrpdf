import sys
import unittest
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
from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow
from bigocrpdf.ui.pdf_editor.page_model import PDFDocument

# Restore original modules to avoid contaminating other test files
for _mod_name, _original in _saved_modules.items():
    if _original is not None:
        sys.modules[_mod_name] = _original
    else:
        sys.modules.pop(_mod_name, None)
del _saved_modules, _MOCKED_MODULES


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

            # Call OK
            window._on_ok_clicked(None)

            # Verify callback called
            mock_save_callback.assert_called_once()
            # Verify modifications cleared
            self.assertFalse(doc.modified)
            print("Save behavior test passed!")

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
            window._on_back_clicked(None)

            # Verify callback NOT called
            mock_save_callback.assert_not_called()
            # Verify modifications cleared (explicit discard)
            self.assertFalse(doc.modified)
            print("Back/Discard behavior test passed!")


if __name__ == "__main__":
    unittest.main()
