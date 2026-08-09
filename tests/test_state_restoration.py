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
from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

_mocked_editor_window_module = sys.modules[_EDITOR_WINDOW_MODULE]

# Restore original modules to avoid contaminating other test files
for _mod_name, _original in _saved_modules.items():
    if _original is not None:
        sys.modules[_mod_name] = _original
    else:
        sys.modules.pop(_mod_name, None)
for _mod_name in tuple(sys.modules):
    if not _mod_name.startswith(_EDITOR_MODULE_PREFIX):
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


class TestStateRestoration(unittest.TestCase):
    def setUp(self):
        self.mock_app = MagicMock()

        # Setup page grid mock
        self.mock_grid = MagicMock()
        self.mock_status_bar = MagicMock()

        # Patch PageGrid and other UI components
        patcher = patch.object(
            _mocked_editor_window_module,
            "PageGrid",
            return_value=self.mock_grid,
        )
        self.addCleanup(patcher.stop)
        self.mock_page_grid_cls = patcher.start()

    def test_restore_state(self):
        # Create a "saved" state
        saved_state = {
            "path": "test.pdf",
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "rotation": 90,
                    "deleted": False,
                    "included_for_ocr": True,
                    "position": 0,
                    "source_file": "test.pdf",
                }
            ],
            "split_points": [],
        }

        # Verify Window restores this state
        with (
            patch.object(PDFEditorWindow, "_setup_ui"),
            patch.object(PDFEditorWindow, "_setup_actions"),
            patch.object(PDFEditorWindow, "_setup_keyboard_shortcuts"),
            patch.object(PDFEditorWindow, "_setup_drag_drop"),
            patch("os.path.basename", return_value="test.pdf"),
        ):
            # Initialize with initial_state
            window = PDFEditorWindow(self.mock_app, "test.pdf", initial_state=saved_state)

            # Check if document was loaded from state
            document = window.document
            self.assertIsNotNone(document)
            assert document is not None
            self.assertEqual(len(document.pages), 1)
            self.assertEqual(document.pages[0].rotation, 90)
