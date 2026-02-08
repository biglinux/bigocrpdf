"""PDF Options Dialog Callbacks Mixin."""

import os
import time
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class PDFOptionsCallbacksMixin:
    """Mixin providing PDF options dialog callback/logic methods."""

    def _setup_pdf_options_callbacks(
        self,
        dialog: Adw.Window,
        prefs_page: Adw.PreferencesPage,
        header_bar: Adw.HeaderBar,
        callback: Callable,
    ) -> None:
        """Set up all callbacks for PDF options dialog

        Args:
            dialog: The dialog window
            prefs_page: The preferences page
            header_bar: The header bar with save button
            callback: The completion callback
        """
        # Get all groups and their components
        file_group = prefs_page.file_group
        text_group = prefs_page.text_group
        odf_group = prefs_page.odf_group
        date_group = prefs_page.date_group
        preview_group = prefs_page.preview_group

        # Set up preview update function
        def update_preview(*_args):
            self._update_filename_preview(file_group, date_group, preview_group)

        def update_date_options_sensitivity(*_args):
            is_date_enabled = date_group.include_date_row.get_active()
            date_group.year_row.set_sensitive(is_date_enabled)
            date_group.month_row.set_sensitive(is_date_enabled)
            date_group.day_row.set_sensitive(is_date_enabled)
            date_group.time_row.set_sensitive(is_date_enabled)
            date_group.format_row.set_sensitive(is_date_enabled)
            update_preview()

        def update_text_options_sensitivity(*_args):
            is_save_txt = text_group.save_txt_row.get_active()
            is_separate_folder = text_group.separate_folder_row.get_active()

            text_group.separate_folder_row.set_sensitive(is_save_txt)
            text_group.text_folder_row.set_sensitive(is_save_txt and is_separate_folder)
            text_group.text_folder_row.folder_button.set_sensitive(
                is_save_txt and is_separate_folder
            )

        def update_odf_options_sensitivity(*_args):
            is_save_odf = odf_group.save_odf_row.get_active()
            odf_group.include_images_row.set_sensitive(is_save_odf)
            odf_group.use_formatting_row.set_sensitive(is_save_odf)

        # Connect all signals
        file_group.use_original_name_row.connect("notify::active", update_preview)
        file_group.suffix_row.connect("changed", update_preview)

        date_group.include_date_row.connect("notify::active", update_date_options_sensitivity)
        date_group.year_row.connect("notify::active", update_preview)
        date_group.month_row.connect("notify::active", update_preview)
        date_group.day_row.connect("notify::active", update_preview)
        date_group.time_row.connect("notify::active", update_preview)
        date_group.format_row.connect("notify::selected", update_preview)

        text_group.save_txt_row.connect("notify::active", update_text_options_sensitivity)
        text_group.separate_folder_row.connect("notify::active", update_text_options_sensitivity)

        odf_group.save_odf_row.connect("notify::active", update_odf_options_sensitivity)

        # Folder selection callback
        def select_text_folder(*_args):
            self._show_folder_selection_dialog(text_group.text_folder_row.folder_label)

        text_group.text_folder_row.folder_button.connect("clicked", select_text_folder)

        # Save button callback
        def on_save_button_clicked(_button):
            self._save_pdf_options(dialog, file_group, text_group, odf_group, date_group, callback)

        # Connect save button (header_bar passed as parameter)
        header_bar.save_button.connect("clicked", on_save_button_clicked)

        # Initial updates
        update_preview()
        update_date_options_sensitivity()
        update_text_options_sensitivity()
        update_odf_options_sensitivity()

    def _update_filename_preview(
        self,
        file_group: Adw.PreferencesGroup,
        date_group: Adw.PreferencesGroup,
        preview_group: Adw.PreferencesGroup,
    ) -> None:
        """Update the filename preview

        Args:
            file_group: File settings group
            date_group: Date settings group
            preview_group: Preview group
        """
        now = time.localtime()
        suffix = file_group.suffix_row.get_text() or "ocr"
        use_original = file_group.use_original_name_row.get_active()

        # Update UI sensitivity
        file_group.suffix_row.set_sensitive(not use_original)
        file_group.warning_row.set_visible(use_original)

        # Format date parts if enabled
        date_str = ""
        if date_group.include_date_row.get_active():
            date_str = self._format_date_for_preview(date_group, now)

        # Create sample filename
        if use_original:
            preview_text = "original_document.pdf"
        else:
            sample_name = "document"
            if date_str:
                preview_text = f"{sample_name}-{suffix}-{date_str}.pdf"
            else:
                preview_text = f"{sample_name}-{suffix}.pdf"

        preview_group.preview_value.set_text(preview_text)

    def _format_date_for_preview(
        self, date_group: Adw.PreferencesGroup, now: time.struct_time
    ) -> str:
        """Format date string for preview.

        Args:
            date_group: Date settings group
            now: Current time struct

        Returns:
            Formatted date string
        """
        date_parts = self._build_date_parts(date_group, now)
        date_str = "-".join(date_parts)
        return self._append_time_if_enabled(date_group, date_str, now)

    def _build_date_parts(
        self, date_group: Adw.PreferencesGroup, now: time.struct_time
    ) -> list[str]:
        """Build date parts list based on format and enabled components.

        Args:
            date_group: Date settings group
            now: Current time struct

        Returns:
            List of formatted date components
        """
        selected_format = date_group.format_row.get_selected()
        component_order = self._get_component_order(selected_format)
        component_values = self._get_component_values(date_group, now)

        return [
            component_values[component]
            for component in component_order
            if component_values.get(component)
        ]

    def _get_component_order(self, selected_format: int) -> tuple[str, ...]:
        """Get the component order based on selected date format.

        Args:
            selected_format: Index of selected format (0=ISO, 1=Europe, 2=US)

        Returns:
            Tuple of component names in display order
        """
        format_orders = {
            0: ("year", "month", "day"),  # YYYY-MM-DD (ISO)
            1: ("day", "month", "year"),  # DD-MM-YYYY (Europe)
            2: ("month", "day", "year"),  # MM-DD-YYYY (US)
        }
        return format_orders.get(selected_format, format_orders[0])

    def _get_component_values(
        self, date_group: Adw.PreferencesGroup, now: time.struct_time
    ) -> dict[str, str | None]:
        """Get formatted values for each date component if enabled.

        Args:
            date_group: Date settings group
            now: Current time struct

        Returns:
            Dictionary mapping component names to formatted values (or None if disabled)
        """
        return {
            "year": f"{now.tm_year}" if date_group.year_row.get_active() else None,
            "month": f"{now.tm_mon:02d}" if date_group.month_row.get_active() else None,
            "day": f"{now.tm_mday:02d}" if date_group.day_row.get_active() else None,
        }

    def _append_time_if_enabled(
        self, date_group: Adw.PreferencesGroup, date_str: str, now: time.struct_time
    ) -> str:
        """Append time to date string if time option is enabled.

        Args:
            date_group: Date settings group
            date_str: Current date string
            now: Current time struct

        Returns:
            Date string with optional time appended
        """
        if not date_group.time_row.get_active():
            return date_str

        time_str = f"{now.tm_hour:02d}{now.tm_min:02d}"
        if date_str:
            return f"{date_str}-{time_str}"
        return time_str

    def _show_folder_selection_dialog(self, folder_label: Gtk.Label) -> None:
        """Show folder selection dialog

        Args:
            folder_label: Label to update with selected folder
        """

        def on_folder_selected(dialog, result):
            try:
                file = dialog.select_folder_finish(result)
                if file:
                    path = file.get_path()
                    folder_label.set_label(path)
            except Exception as e:
                logger.error(f"Error selecting folder: {e}")

        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Select Folder for Text Files"))

        current_folder = folder_label.get_label()
        if current_folder and current_folder != _("Not set") and os.path.exists(current_folder):
            dialog.set_initial_folder(Gio.File.new_for_path(current_folder))

        dialog.select_folder(parent=self.window, cancellable=None, callback=on_folder_selected)

    def _save_pdf_options(
        self,
        dialog: Adw.Window,
        file_group: Adw.PreferencesGroup,
        text_group: Adw.PreferencesGroup,
        odf_group: Adw.PreferencesGroup,
        date_group: Adw.PreferencesGroup,
        callback: Callable,
    ) -> None:
        """Save PDF options and close dialog

        Args:
            dialog: The dialog window
            file_group: File settings group
            text_group: Text extraction group
            odf_group: ODF export group
            date_group: Date settings group
            callback: Completion callback
        """
        # Save file settings
        self.window.settings.use_original_filename = file_group.use_original_name_row.get_active()
        self.window.settings.pdf_suffix = file_group.suffix_row.get_text() or "ocr"
        self.window.settings.overwrite_existing = file_group.overwrite_row.get_active()

        # Save text extraction settings
        self.window.settings.save_txt = text_group.save_txt_row.get_active()
        self.window.settings.separate_txt_folder = text_group.separate_folder_row.get_active()
        folder_text = text_group.text_folder_row.folder_label.get_label()
        self.window.settings.txt_folder = folder_text if folder_text != _("Not set") else ""

        # Save ODF export settings
        self.window.settings.save_odf = odf_group.save_odf_row.get_active()
        self.window.settings.odf_include_images = odf_group.include_images_row.get_active()
        self.window.settings.odf_use_formatting = odf_group.use_formatting_row.get_active()

        # Save date settings
        self.window.settings.include_date = date_group.include_date_row.get_active()
        self.window.settings.include_year = date_group.year_row.get_active()
        self.window.settings.include_month = date_group.month_row.get_active()
        self.window.settings.include_day = date_group.day_row.get_active()
        self.window.settings.include_time = date_group.time_row.get_active()

        # Save date format order
        selected_format = date_group.format_row.get_selected()
        if not hasattr(self.window.settings, "date_format_order"):
            self.window.settings.date_format_order = {}

        if selected_format == 1:  # DD-MM-YYYY
            self.window.settings.date_format_order = {"day": 1, "month": 2, "year": 3}
        elif selected_format == 2:  # MM-DD-YYYY
            self.window.settings.date_format_order = {"month": 1, "day": 2, "year": 3}
        else:  # YYYY-MM-DD
            self.window.settings.date_format_order = {"year": 1, "month": 2, "day": 3}

        # Close dialog and call callback
        dialog.destroy()
        callback(True)
