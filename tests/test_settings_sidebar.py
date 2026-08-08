"""Settings sidebar behavior contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from bigocrpdf.ui.settings_sidebar_mixin import SettingsSidebarMixin


def test_invalid_image_quality_selection_is_ignored() -> None:
    settings = SimpleNamespace(
        image_export_format="original",
        image_export_quality=95,
        force_bilevel_compression=False,
        _save_all_settings=MagicMock(),
    )
    manager = SettingsSidebarMixin.__new__(SettingsSidebarMixin)
    manager.window = SimpleNamespace(settings=settings)
    combo = MagicMock()
    combo.get_selected.return_value = Gtk.INVALID_LIST_POSITION

    manager._on_image_quality_changed(combo, None)

    assert settings.image_export_format == "original"
    assert settings.image_export_quality == 95
    assert settings.force_bilevel_compression is False
    settings._save_all_settings.assert_not_called()
