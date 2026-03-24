"""
BigOcrPdf - Settings Page Module

Thin orchestrator that inherits sidebar and queue mixins.
"""

from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk

from bigocrpdf.ui.settings_queue_mixin import SettingsQueueMixin
from bigocrpdf.ui.settings_sidebar_mixin import SettingsSidebarMixin

if TYPE_CHECKING:
    from window import BigOcrPdfWindow


class SettingsPageManager(SettingsSidebarMixin, SettingsQueueMixin):
    """Manages the settings page UI and interactions."""

    def __init__(self, window: "BigOcrPdfWindow"):
        self.window = window

        # UI component references
        self.lang_dropdown = None
        self.dest_entry = None
        self.folder_combo = None
        self.folder_entry_box = None
        self.file_list_box = None
        self.placeholder = None

        # Preprocessing UI components
        self.deskew_switch = None
        self.dewarp_switch = None
        self.perspective_switch = None
        self.orientation_switch = None
        self.scanner_switch = None
        self.replace_ocr_switch = None
        self._preprocessing_signal_connected = False

    def create_settings_page(self) -> Gtk.Widget:
        """Create the settings page (file queue only)."""
        return self._create_file_queue_panel()

    def create_sidebar_content(self) -> Gtk.Widget:
        """Create the sidebar content with OCR settings."""
        return self._create_config_panel()

    def sync_ui_to_settings(self) -> None:
        """Re-sync all UI widgets to current OcrSettings values."""
        settings = self.window.settings

        if self.lang_dropdown:
            languages = self.window.ocr_processor.get_available_ocr_languages()
            for i, (code, _name) in enumerate(languages):
                if code == settings.lang:
                    self.lang_dropdown.set_selected(i)
                    break

        if self.folder_combo:
            self.folder_combo.set_selected(0 if settings.save_in_same_folder else 1)
        if self.dest_entry:
            self.dest_entry.set_text(settings.destination_folder or "")

        self._load_preprocessing_settings()
        self._load_advanced_ocr_settings()
        self._load_image_export_settings()

        if self.file_list_box:
            self._populate_file_list()

    def refresh_queue_status(self) -> None:
        """Update the queue status without rebuilding the entire settings page."""
        file_count = len(self.window.settings.selected_files)

        if self.folder_combo and self.folder_entry_box:
            use_custom = self.folder_combo.get_selected() == 1
            self.folder_entry_box.set_visible(use_custom)

        if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
            self.window.custom_header_bar.update_queue_size(file_count)

        if self.file_list_box:
            self._populate_file_list()
