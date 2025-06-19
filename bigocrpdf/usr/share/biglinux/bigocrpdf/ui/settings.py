"""
BigOcrPdf - Settings Configuration

This module contains the settings configuration dialog and functionality.
"""

import gi
import os
import time

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, Pango

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.logger import logger
from utils.i18n import _


class SettingsDialog:
    """Settings configuration dialog for BigOcrPdf"""

    def __init__(self, parent_window: "BigOcrPdfWindow"):
        """Initialize settings dialog

        Args:
            parent_window: The parent window
        """
        self.parent_window = parent_window
        self.dialog = None

    def show_pdf_options_dialog(self, callback) -> None:
        """Show PDF options configuration dialog

        Args:
            callback: Callback function to call after dialog closes
        """
        # Create the dialog using Adw.Window for full control
        self.dialog = Adw.Window()
        self.dialog.set_title(_("PDF Output Options"))
        self.dialog.set_default_size(600, 580)
        self.dialog.set_modal(True)
        self.dialog.set_transient_for(self.parent_window)

        # Set up the Adwaita toolbar view structure (main container)
        toolbar_view = Adw.ToolbarView()

        # Create a header bar
        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(False)
        header_bar.set_title_widget(Gtk.Label(label=_("PDF Output Settings")))

        # Add cancel button to header
        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda b: self.dialog.destroy())
        header_bar.pack_start(cancel_button)

        # Add save button to header
        save_button = Gtk.Button(label=_("Save"))
        save_button.add_css_class("suggested-action")
        header_bar.pack_end(save_button)

        # Add the header to the toolbar view
        toolbar_view.add_top_bar(header_bar)

        # Create scrolled window for content
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)

        # Create a preferences page
        prefs_page = Adw.PreferencesPage()

        # File Settings group
        main_group = Adw.PreferencesGroup()
        main_group.set_title(_("File Settings"))

        # Use original filename switch row
        use_original_name_row = Adw.SwitchRow()
        use_original_name_row.set_title(_("Use Original Filename"))
        use_original_name_row.set_subtitle(_("Use the same name as the original file"))

        # Check if the setting exists, if not, default to False
        use_orig_name = getattr(
            self.parent_window.settings, "use_original_filename", False
        )
        use_original_name_row.set_active(use_orig_name)

        # Add a warning for users about potential file replacement
        warning_row = Adw.ActionRow()
        warning_row.set_title(_("Warning"))
        warning_row.set_subtitle(
            _(
                "If saving to the same folder as original, this will replace the original files"
            )
        )
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        warning_icon.set_pixel_size(16)
        warning_row.add_prefix(warning_icon)
        warning_row.add_css_class("warning-row")
        warning_row.set_visible(use_orig_name)

        main_group.add(use_original_name_row)
        main_group.add(warning_row)

        # Suffix entry row
        suffix_row = Adw.EntryRow()
        suffix_row.set_title(_("Filename Suffix"))
        suffix_row.set_tooltip_text(
            _("Text added to the filename (e.g. document-ocr.pdf)")
        )
        suffix_row.set_text(self.parent_window.settings.pdf_suffix or "ocr")
        suffix_row.set_show_apply_button(False)
        suffix_row.set_sensitive(not use_orig_name)
        main_group.add(suffix_row)

        # Overwrite files switch row
        overwrite_row = Adw.SwitchRow()
        overwrite_row.set_title(_("Overwrite Existing Files"))
        overwrite_row.set_subtitle(_("Replace files with the same name"))
        overwrite_row.set_active(self.parent_window.settings.overwrite_existing)
        main_group.add(overwrite_row)

        # Text extraction group
        text_group = Adw.PreferencesGroup()
        text_group.set_title(_("Text Extraction"))

        # Save TXT files switch row
        save_txt_row = Adw.SwitchRow()
        save_txt_row.set_title(_("Save Text Files"))
        save_txt_row.set_subtitle(_("Automatically save extracted text as .txt files"))
        save_txt = getattr(self.parent_window.settings, "save_txt", False)
        save_txt_row.set_active(save_txt)
        text_group.add(save_txt_row)

        # Separate folder for TXT files switch row
        separate_folder_row = Adw.SwitchRow()
        separate_folder_row.set_title(_("Use Separate Folder for Text Files"))
        separate_folder_row.set_subtitle(_("Save text files to a different folder"))
        use_separate_folder = getattr(
            self.parent_window.settings, "separate_txt_folder", False
        )
        separate_folder_row.set_active(use_separate_folder)
        separate_folder_row.set_sensitive(save_txt)
        text_group.add(separate_folder_row)

        # Text folder selection button row
        text_folder_row = Adw.ActionRow()
        text_folder_row.set_title(_("Text Files Folder"))
        text_folder_row.set_subtitle(_("Select where to save text files"))
        text_folder_row.set_sensitive(save_txt and use_separate_folder)

        # Get current text folder or empty string if not set
        txt_folder = getattr(self.parent_window.settings, "txt_folder", "")
        folder_label = Gtk.Label(label=txt_folder or _("Not set"))
        folder_label.set_ellipsize(Pango.EllipsizeMode.START)
        folder_label.set_halign(Gtk.Align.END)
        folder_label.set_margin_end(8)

        # Add button to select folder
        folder_button = Gtk.Button()
        folder_button.set_icon_name("folder-symbolic")
        folder_button.set_valign(Gtk.Align.CENTER)
        folder_button.add_css_class("flat")
        folder_button.set_tooltip_text(_("Select folder"))
        folder_button.set_sensitive(save_txt and use_separate_folder)

        text_folder_row.add_suffix(folder_label)
        text_folder_row.add_suffix(folder_button)
        text_group.add(text_folder_row)

        # Date format options
        date_group = Adw.PreferencesGroup()
        date_group.set_title(_("Date and Time"))

        # Date switch row
        include_date_row = Adw.SwitchRow()
        include_date_row.set_title(_("Add Date to Filename"))
        include_date_row.set_subtitle(_("Include date elements in the filename"))
        include_date_row.set_active(self.parent_window.settings.include_date)
        date_group.add(include_date_row)

        # Date format row with dropdown
        format_row = Adw.ComboRow()
        format_row.set_title(_("Date Format"))
        format_model = Gtk.StringList()
        format_model.append(_("YYYY-MM-DD (ISO)"))
        format_model.append(_("DD-MM-YYYY (Europe)"))
        format_model.append(_("MM-DD-YYYY (US)"))
        format_row.set_model(format_model)
        format_row.set_sensitive(include_date_row.get_active())

        # Set default format
        order = getattr(
            self.parent_window.settings,
            "date_format_order",
            {"year": 1, "month": 2, "day": 3},
        )
        if order["day"] < order["month"] and order["month"] < order["year"]:
            format_row.set_selected(1)  # DD-MM-YYYY
        elif order["month"] < order["day"] and order["day"] < order["year"]:
            format_row.set_selected(2)  # MM-DD-YYYY
        else:
            format_row.set_selected(0)  # ISO format
        date_group.add(format_row)

        # Year switch row
        year_row = Adw.SwitchRow()
        year_row.set_title(_("Include Year"))
        year_row.set_subtitle(_("Add YYYY to the date"))
        year_row.set_active(self.parent_window.settings.include_year)
        year_row.set_sensitive(include_date_row.get_active())
        date_group.add(year_row)

        # Month switch row
        month_row = Adw.SwitchRow()
        month_row.set_title(_("Include Month"))
        month_row.set_subtitle(_("Add MM to the date"))
        month_row.set_active(self.parent_window.settings.include_month)
        month_row.set_sensitive(include_date_row.get_active())
        date_group.add(month_row)

        # Day switch row
        day_row = Adw.SwitchRow()
        day_row.set_title(_("Include Day"))
        day_row.set_subtitle(_("Add DD to the date"))
        day_row.set_active(self.parent_window.settings.include_day)
        day_row.set_sensitive(include_date_row.get_active())
        date_group.add(day_row)

        # Time switch row
        time_row = Adw.SwitchRow()
        time_row.set_title(_("Include Time"))
        time_row.set_subtitle(_("Add HHMM to the filename"))
        time_row.set_active(self.parent_window.settings.include_time)
        time_row.set_sensitive(include_date_row.get_active())
        date_group.add(time_row)

        # Sample file preview (will be used in bottom bar)
        preview_value = Gtk.Label()
        preview_value.add_css_class("monospace")
        preview_value.add_css_class("caption")
        preview_value.add_css_class("dim-label")

        # Add groups to preferences page (without preview group)
        prefs_page.add(main_group)
        prefs_page.add(text_group)
        prefs_page.add(date_group)

        # Add preferences page to scrolled window
        scrolled.set_child(prefs_page)
        toolbar_view.set_content(scrolled)

        # Create bottom bar for preview (fixed at bottom like absolute positioning)
        bottom_bar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        bottom_bar.add_css_class("toolbar")

        # Add separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        bottom_bar.append(separator)

        # Create preview container with padding
        preview_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        preview_container.set_margin_top(12)
        preview_container.set_margin_bottom(12)
        preview_container.set_margin_start(16)
        preview_container.set_margin_end(16)

        # Preview label
        preview_label = Gtk.Label()
        preview_label.set_markup(f"<b>{_('Filename Example:')}</b>")
        preview_label.set_halign(Gtk.Align.START)
        preview_label.set_hexpand(True)

        # Sample file preview (reuse existing preview_value)
        preview_value.set_halign(Gtk.Align.END)
        preview_value.set_hexpand(False)

        preview_container.append(preview_label)
        preview_container.append(preview_value)
        bottom_bar.append(preview_container)

        # Add bottom bar to toolbar view
        toolbar_view.add_bottom_bar(bottom_bar)
        preview_container.set_margin_bottom(12)
        preview_container.set_margin_start(16)
        preview_container.set_margin_end(16)

        # Preview value (reuse the existing one)
        preview_value.set_halign(Gtk.Align.END)
        preview_value.set_hexpand(False)

        preview_container.append(preview_label)
        preview_container.append(preview_value)
        bottom_bar.append(preview_container)

        # Add bottom bar to toolbar view
        toolbar_view.add_bottom_bar(bottom_bar)
        self.dialog.set_content(toolbar_view)

        # Setup event handlers and connections
        self._setup_event_handlers(
            use_original_name_row,
            suffix_row,
            overwrite_row,
            save_txt_row,
            separate_folder_row,
            text_folder_row,
            folder_label,
            folder_button,
            include_date_row,
            format_row,
            year_row,
            month_row,
            day_row,
            time_row,
            preview_value,
            warning_row,
            save_button,
            callback,
        )

        # Show the dialog
        self.dialog.present()

    def _setup_event_handlers(
        self,
        use_original_name_row,
        suffix_row,
        overwrite_row,
        save_txt_row,
        separate_folder_row,
        text_folder_row,
        folder_label,
        folder_button,
        include_date_row,
        format_row,
        year_row,
        month_row,
        day_row,
        time_row,
        preview_value,
        warning_row,
        save_button,
        callback,
    ):
        """Setup all event handlers for the settings dialog"""

        # Function to update the preview filename
        def update_preview(*args):
            now = time.localtime()
            suffix = suffix_row.get_text() or "ocr"
            use_original = use_original_name_row.get_active()

            # Format date parts if enabled
            if include_date_row.get_active():
                date_parts = []
                selected_format = format_row.get_selected()

                # Get date components to include
                need_year = year_row.get_active()
                need_month = month_row.get_active()
                need_day = day_row.get_active()

                # Get format based on dropdown selection
                if selected_format == 0:  # YYYY-MM-DD
                    if need_year:
                        date_parts.append(f"{now.tm_year}")
                    if need_month:
                        date_parts.append(f"{now.tm_mon:02d}")
                    if need_day:
                        date_parts.append(f"{now.tm_mday:02d}")
                elif selected_format == 1:  # DD-MM-YYYY
                    if need_day:
                        date_parts.append(f"{now.tm_mday:02d}")
                    if need_month:
                        date_parts.append(f"{now.tm_mon:02d}")
                    if need_year:
                        date_parts.append(f"{now.tm_year}")
                else:  # MM-DD-YYYY
                    if need_month:
                        date_parts.append(f"{now.tm_mon:02d}")
                    if need_day:
                        date_parts.append(f"{now.tm_mday:02d}")
                    if need_year:
                        date_parts.append(f"{now.tm_year}")

                # Always use hyphen separator for filename compatibility
                date_str = "-".join(date_parts)

                # Add time as a separate component if enabled
                if time_row.get_active():
                    time_str = f"{now.tm_hour:02d}{now.tm_min:02d}"
                    if date_parts:
                        date_str = f"{date_str}-{time_str}"
                    else:
                        date_str = time_str
            else:
                date_str = ""

            # Create sample filename based on settings
            if use_original:
                # Use example name to show it's using original name
                sample_name = "original_document"
                preview_text = f"{sample_name}.pdf"
            else:
                sample_name = "document"
                if date_str:
                    preview_text = f"{sample_name}-{suffix}-{date_str}.pdf"
                else:
                    preview_text = f"{sample_name}-{suffix}.pdf"

            preview_value.set_text(preview_text)

            # Update UI sensitivity
            suffix_row.set_sensitive(not use_original)
            # Show/hide warning based on original filename setting
            warning_row.set_visible(use_original)

        # Function to enable/disable date options based on switch
        def update_date_options_sensitivity(*args):
            is_date_enabled = include_date_row.get_active()
            year_row.set_sensitive(is_date_enabled)
            month_row.set_sensitive(is_date_enabled)
            day_row.set_sensitive(is_date_enabled)
            time_row.set_sensitive(is_date_enabled)
            format_row.set_sensitive(is_date_enabled)
            update_preview()

        # Function to enable/disable text extraction options
        def update_text_options_sensitivity(*args):
            is_save_txt = save_txt_row.get_active()
            is_separate_folder = separate_folder_row.get_active()

            separate_folder_row.set_sensitive(is_save_txt)
            text_folder_row.set_sensitive(is_save_txt and is_separate_folder)
            folder_button.set_sensitive(is_save_txt and is_separate_folder)

        # Function to handle folder selection
        def on_folder_selected(dialog, result):
            try:
                file = dialog.select_folder_finish(result)
                if file:
                    path = file.get_path()
                    folder_label.set_label(path)
            except Exception as e:
                logger.error(f"Error selecting folder: {e}")

        # Setup folder selection
        def select_text_folder(*args):
            dialog = Gtk.FileDialog.new()
            dialog.set_title(_("Select Folder for Text Files"))
            txt_folder = getattr(self.parent_window.settings, "txt_folder", "")
            if txt_folder and os.path.exists(txt_folder):
                dialog.set_initial_folder(Gio.File.new_for_path(txt_folder))
            dialog.select_folder(
                parent=self.parent_window, cancellable=None, callback=on_folder_selected
            )

        # Handle save button
        def on_save_button_clicked(button):
            # Save settings to the application
            self.parent_window.settings.use_original_filename = (
                use_original_name_row.get_active()
            )
            self.parent_window.settings.pdf_suffix = suffix_row.get_text() or "ocr"
            self.parent_window.settings.overwrite_existing = overwrite_row.get_active()
            self.parent_window.settings.include_date = include_date_row.get_active()
            self.parent_window.settings.include_year = year_row.get_active()
            self.parent_window.settings.include_month = month_row.get_active()
            self.parent_window.settings.include_day = day_row.get_active()
            self.parent_window.settings.include_time = time_row.get_active()

            # Save text extraction settings
            self.parent_window.settings.save_txt = save_txt_row.get_active()
            self.parent_window.settings.separate_txt_folder = (
                separate_folder_row.get_active()
            )
            self.parent_window.settings.txt_folder = (
                folder_label.get_label()
                if folder_label.get_label() != _("Not set")
                else ""
            )

            # Save date format order preferences based on selected dropdown value
            selected_format = format_row.get_selected()

            if not hasattr(self.parent_window.settings, "date_format_order"):
                self.parent_window.settings.date_format_order = {}

            # Set order based on dropdown selection
            if selected_format == 1:  # DD/MM/YYYY (Europe)
                self.parent_window.settings.date_format_order = {
                    "day": 1,
                    "month": 2,
                    "year": 3,
                }
            elif selected_format == 2:  # MM/DD/YYYY (US)
                self.parent_window.settings.date_format_order = {
                    "month": 1,
                    "day": 2,
                    "year": 3,
                }
            else:  # YYYY-MM-DD (ISO)
                self.parent_window.settings.date_format_order = {
                    "year": 1,
                    "month": 2,
                    "day": 3,
                }

            # Close dialog and call callback
            self.dialog.destroy()
            if callback:
                callback(True)

        # Connect all event handlers
        suffix_row.connect("changed", update_preview)
        use_original_name_row.connect("notify::active", update_preview)
        include_date_row.connect("notify::active", update_date_options_sensitivity)
        year_row.connect("notify::active", update_preview)
        month_row.connect("notify::active", update_preview)
        day_row.connect("notify::active", update_preview)
        time_row.connect("notify::active", update_preview)
        format_row.connect("notify::selected", update_preview)

        save_txt_row.connect("notify::active", update_text_options_sensitivity)
        separate_folder_row.connect("notify::active", update_text_options_sensitivity)
        folder_button.connect("clicked", select_text_folder)

        save_button.connect("clicked", on_save_button_clicked)

        # Initial updates
        update_date_options_sensitivity()
        update_text_options_sensitivity()
        update_preview()

    def _apply_settings_and_close(self, callback) -> None:
        """Apply settings and close the dialog

        Args:
            callback: Callback function to call after applying settings
        """
        # Here you would typically save the settings
        # This is a placeholder implementation
        logger.info("PDF output settings applied")

        # Close dialog
        self.dialog.close()

        # Call callback if provided
        if callback:
            callback(None)


class AdvancedSettings:
    """Advanced settings for OCR processing"""

    def __init__(self, parent_window: "BigOcrPdfWindow"):
        """Initialize advanced settings

        Args:
            parent_window: The parent window
        """
        self.parent_window = parent_window

    def show_advanced_dialog(self) -> None:
        """Show advanced settings dialog"""
        # Create the dialog
        dialog = Adw.PreferencesWindow()
        dialog.set_title(_("Advanced Settings"))
        dialog.set_default_size(600, 500)
        dialog.set_modal(True)
        dialog.set_transient_for(self.parent_window)

        # Create a preferences page
        prefs_page = Adw.PreferencesPage()
        prefs_page.set_title(_("Advanced Settings"))

        # OCR Engine Settings
        ocr_group = Adw.PreferencesGroup()
        ocr_group.set_title(_("OCR Engine"))
        ocr_group.set_description(_("Configure OCR engine parameters"))

        # DPI Setting
        dpi_row = Adw.SpinRow()
        dpi_row.set_title(_("Image DPI"))
        dpi_row.set_subtitle(_("Resolution for image processing"))
        dpi_adjustment = Gtk.Adjustment(
            value=300, lower=150, upper=600, step_increment=50
        )
        dpi_row.set_adjustment(dpi_adjustment)
        ocr_group.add(dpi_row)

        # Add OCR group to page
        prefs_page.add(ocr_group)

        # Preprocessing options
        preprocessing_group = Adw.PreferencesGroup()
        preprocessing_group.set_title(_("Image Preprocessing"))

        # Enhance contrast
        contrast_row = Adw.SwitchRow()
        contrast_row.set_title(_("Enhance Contrast"))
        contrast_row.set_subtitle(_("Improve image contrast before OCR"))
        contrast_row.set_active(True)
        preprocessing_group.add(contrast_row)

        # Noise reduction
        noise_row = Adw.SwitchRow()
        noise_row.set_title(_("Noise Reduction"))
        noise_row.set_subtitle(_("Remove noise from images"))
        noise_row.set_active(True)
        preprocessing_group.add(noise_row)

        # Add preprocessing group to page
        prefs_page.add(preprocessing_group)

        # Performance Settings
        performance_group = Adw.PreferencesGroup()
        performance_group.set_title(_("Performance"))

        # Parallel processing
        parallel_row = Adw.SwitchRow()
        parallel_row.set_title(_("Parallel Processing"))
        parallel_row.set_subtitle(_("Process multiple pages simultaneously"))
        parallel_row.set_active(True)
        performance_group.add(parallel_row)

        # Thread count
        thread_row = Adw.SpinRow()
        thread_row.set_title(_("Thread Count"))
        thread_row.set_subtitle(_("Number of processing threads"))
        thread_adjustment = Gtk.Adjustment(value=4, lower=1, upper=16, step_increment=1)
        thread_row.set_adjustment(thread_adjustment)
        performance_group.add(thread_row)

        # Add performance group to page
        prefs_page.add(performance_group)

        # Add the preferences page to the window
        dialog.add(prefs_page)

        # Show the dialog
        dialog.present()

    def _reset_to_defaults(self) -> None:
        """Reset all settings to default values"""
        logger.info("Resetting advanced settings to defaults")
        # Implementation would reset all settings to their default values
