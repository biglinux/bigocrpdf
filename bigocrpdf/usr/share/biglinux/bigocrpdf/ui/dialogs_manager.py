"""
BigOcrPdf - Dialogs Manager Module

This module handles all dialog creation and management for the application.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, Gdk, Pango

import os
import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from utils.logger import logger
from utils.i18n import _


class DialogsManager:
    """Manages all dialogs and modal windows for the application"""
    
    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the dialogs manager
        
        Args:
            window: Reference to the main application window
        """
        self.window = window

    def show_pdf_options_dialog(self, callback: Callable) -> None:
        """Show dialog with PDF output options before starting OCR

        Args:
            callback: Function to call with options when dialog is confirmed
        """
        # Create the options dialog
        dialog = Adw.Window()
        dialog.set_title(_("PDF Output Options"))
        dialog.set_default_size(550, 590)
        dialog.set_modal(True)
        dialog.set_transient_for(self.window)

        # Set up the main structure
        dialog_content = self._create_pdf_options_content(dialog, callback)
        dialog.set_content(dialog_content)

        # Show the dialog
        dialog.present()

    def _create_pdf_options_content(self, dialog: Adw.Window, callback: Callable) -> Adw.ToolbarView:
        """Create the content for PDF options dialog
        
        Args:
            dialog: The dialog window
            callback: Callback function
            
        Returns:
            The main content widget
        """
        # Set up the Adwaita toolbar view structure (main container)
        toolbar_view = Adw.ToolbarView()

        # Create header bar
        header_bar = self._create_pdf_options_header(dialog, callback)
        toolbar_view.add_top_bar(header_bar)

        # Create scrolled content
        scrolled_content = self._create_pdf_options_scrolled_content()
        toolbar_view.set_content(scrolled_content)

        return toolbar_view

    def _create_pdf_options_header(self, dialog: Adw.Window, callback: Callable) -> Adw.HeaderBar:
        """Create header bar for PDF options dialog
        
        Args:
            dialog: The dialog window
            callback: Callback function
            
        Returns:
            The header bar widget
        """
        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(False)
        header_bar.set_title_widget(Gtk.Label(label=_("PDF Output Settings")))

        # Add cancel button to header
        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda b: dialog.destroy())
        header_bar.pack_start(cancel_button)

        # Add save button to header
        save_button = Gtk.Button(label=_("Save"))
        save_button.add_css_class("suggested-action")
        header_bar.pack_end(save_button)

        # Store save button reference for later connection
        header_bar.save_button = save_button

        return header_bar

    def _create_pdf_options_scrolled_content(self) -> Gtk.ScrolledWindow:
        """Create scrolled content for PDF options dialog
        
        Returns:
            Scrolled window with preferences content
        """
        # Create scrolled window for content
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)

        # Create preferences page with all options
        prefs_page = self._create_pdf_preferences_page()
        scrolled.set_child(prefs_page)

        return scrolled

    def _create_pdf_preferences_page(self) -> Adw.PreferencesPage:
        """Create the preferences page with all PDF options
        
        Returns:
            Preferences page with all option groups
        """
        prefs_page = Adw.PreferencesPage()
        
        # Add all preference groups
        file_group = self._create_file_settings_group()
        text_group = self._create_text_extraction_group()
        date_group = self._create_date_time_group()
        preview_group = self._create_preview_group()

        prefs_page.add(file_group)
        prefs_page.add(text_group)
        prefs_page.add(date_group)
        prefs_page.add(preview_group)

        # Store references for access in callbacks
        prefs_page.file_group = file_group
        prefs_page.text_group = text_group
        prefs_page.date_group = date_group
        prefs_page.preview_group = preview_group

        return prefs_page

    def _create_file_settings_group(self) -> Adw.PreferencesGroup:
        """Create file settings preference group
        
        Returns:
            Preferences group for file settings
        """
        main_group = Adw.PreferencesGroup()
        main_group.set_title(_("File Settings"))

        # Use original filename switch row
        use_original_name_row = Adw.SwitchRow()
        use_original_name_row.set_title(_("Use Original Filename"))
        use_original_name_row.set_subtitle(_("Use the same name as the original file"))
        use_orig_name = getattr(self.window.settings, "use_original_filename", False)
        use_original_name_row.set_active(use_orig_name)

        # Warning row for original filename
        warning_row = self._create_warning_row()
        warning_row.set_visible(use_orig_name)

        # Suffix entry row
        suffix_row = Adw.EntryRow()
        suffix_row.set_title(_("Filename Suffix"))
        suffix_row.set_tooltip_text(_("Text added to the filename (e.g. document-ocr.pdf)"))
        suffix_row.set_text(self.window.settings.pdf_suffix or "ocr")
        suffix_row.set_show_apply_button(False)
        suffix_row.set_sensitive(not use_orig_name)

        # Overwrite files switch row
        overwrite_row = Adw.SwitchRow()
        overwrite_row.set_title(_("Overwrite Existing Files"))
        overwrite_row.set_subtitle(_("Replace files with the same name"))
        overwrite_row.set_active(self.window.settings.overwrite_existing)

        # Add all rows to group
        main_group.add(use_original_name_row)
        main_group.add(warning_row)
        main_group.add(suffix_row)
        main_group.add(overwrite_row)

        # Store references
        main_group.use_original_name_row = use_original_name_row
        main_group.warning_row = warning_row
        main_group.suffix_row = suffix_row
        main_group.overwrite_row = overwrite_row

        return main_group

    def _create_warning_row(self) -> Adw.ActionRow:
        """Create warning row for original filename option
        
        Returns:
            Action row with warning message
        """
        warning_row = Adw.ActionRow()
        warning_row.set_title(_("Warning"))
        warning_row.set_subtitle(
            _("If saving to the same folder as original, this will replace the original files")
        )
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        warning_icon.set_pixel_size(16)
        warning_row.add_prefix(warning_icon)
        warning_row.add_css_class("warning-row")
        return warning_row

    def _create_text_extraction_group(self) -> Adw.PreferencesGroup:
        """Create text extraction preference group
        
        Returns:
            Preferences group for text extraction settings
        """
        text_group = Adw.PreferencesGroup()
        text_group.set_title(_("Text Extraction"))

        # Save TXT files switch row
        save_txt_row = Adw.SwitchRow()
        save_txt_row.set_title(_("Save Text Files"))
        save_txt_row.set_subtitle(_("Automatically save extracted text as .txt files"))
        save_txt = getattr(self.window.settings, "save_txt", False)
        save_txt_row.set_active(save_txt)

        # Separate folder switch row
        separate_folder_row = Adw.SwitchRow()
        separate_folder_row.set_title(_("Use Separate Folder for Text Files"))
        separate_folder_row.set_subtitle(_("Save text files to a different folder"))
        use_separate_folder = getattr(self.window.settings, "separate_txt_folder", False)
        separate_folder_row.set_active(use_separate_folder)
        separate_folder_row.set_sensitive(save_txt)

        # Text folder selection row
        text_folder_row = self._create_text_folder_row(save_txt and use_separate_folder)

        # Add all rows to group
        text_group.add(save_txt_row)
        text_group.add(separate_folder_row)
        text_group.add(text_folder_row)

        # Store references
        text_group.save_txt_row = save_txt_row
        text_group.separate_folder_row = separate_folder_row
        text_group.text_folder_row = text_folder_row

        return text_group

    def _create_text_folder_row(self, initial_sensitivity: bool) -> Adw.ActionRow:
        """Create text folder selection row
        
        Args:
            initial_sensitivity: Initial sensitivity state
            
        Returns:
            Action row for text folder selection
        """
        text_folder_row = Adw.ActionRow()
        text_folder_row.set_title(_("Text Files Folder"))
        text_folder_row.set_subtitle(_("Select where to save text files"))
        text_folder_row.set_sensitive(initial_sensitivity)

        # Get current text folder
        txt_folder = getattr(self.window.settings, "txt_folder", "")
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
        folder_button.set_sensitive(initial_sensitivity)

        text_folder_row.add_suffix(folder_label)
        text_folder_row.add_suffix(folder_button)

        # Store references
        text_folder_row.folder_label = folder_label
        text_folder_row.folder_button = folder_button

        return text_folder_row

    def _create_date_time_group(self) -> Adw.PreferencesGroup:
        """Create date and time preference group
        
        Returns:
            Preferences group for date/time settings
        """
        date_group = Adw.PreferencesGroup()
        date_group.set_title(_("Date and Time"))

        # Date switch row
        include_date_row = Adw.SwitchRow()
        include_date_row.set_title(_("Add Date to Filename"))
        include_date_row.set_subtitle(_("Include date elements in the filename"))
        include_date_row.set_active(self.window.settings.include_date)

        # Date format row
        format_row = self._create_date_format_row(include_date_row.get_active())

        # Date component switches
        year_row = self._create_date_component_row(
            _("Include Year"), _("Add YYYY to the date"),
            self.window.settings.include_year, include_date_row.get_active()
        )
        month_row = self._create_date_component_row(
            _("Include Month"), _("Add MM to the date"),
            self.window.settings.include_month, include_date_row.get_active()
        )
        day_row = self._create_date_component_row(
            _("Include Day"), _("Add DD to the date"),
            self.window.settings.include_day, include_date_row.get_active()
        )
        time_row = self._create_date_component_row(
            _("Include Time"), _("Add HHMM to the filename"),
            self.window.settings.include_time, include_date_row.get_active()
        )

        # Add all rows to group
        date_group.add(include_date_row)
        date_group.add(format_row)
        date_group.add(year_row)
        date_group.add(month_row)
        date_group.add(day_row)
        date_group.add(time_row)

        # Store references
        date_group.include_date_row = include_date_row
        date_group.format_row = format_row
        date_group.year_row = year_row
        date_group.month_row = month_row
        date_group.day_row = day_row
        date_group.time_row = time_row

        return date_group

    def _create_date_format_row(self, initial_sensitivity: bool) -> Adw.ComboRow:
        """Create date format selection row
        
        Args:
            initial_sensitivity: Initial sensitivity state
            
        Returns:
            Combo row for date format selection
        """
        format_row = Adw.ComboRow()
        format_row.set_title(_("Date Format"))
        format_model = Gtk.StringList()
        format_model.append(_("YYYY-MM-DD (ISO)"))
        format_model.append(_("DD-MM-YYYY (Europe)"))
        format_model.append(_("MM-DD-YYYY (US)"))
        format_row.set_model(format_model)
        format_row.set_sensitive(initial_sensitivity)

        # Set default format based on settings
        order = getattr(self.window.settings, "date_format_order", {"year": 1, "month": 2, "day": 3})
        if order["day"] < order["month"] and order["month"] < order["year"]:
            format_row.set_selected(1)  # DD-MM-YYYY
        elif order["month"] < order["day"] and order["day"] < order["year"]:
            format_row.set_selected(2)  # MM-DD-YYYY
        else:
            format_row.set_selected(0)  # YYYY-MM-DD

        return format_row

    def _create_date_component_row(self, title: str, subtitle: str, 
                                  initial_active: bool, initial_sensitivity: bool) -> Adw.SwitchRow:
        """Create a date component switch row
        
        Args:
            title: Row title
            subtitle: Row subtitle
            initial_active: Initial active state
            initial_sensitivity: Initial sensitivity state
            
        Returns:
            Switch row for date component
        """
        row = Adw.SwitchRow()
        row.set_title(title)
        row.set_subtitle(subtitle)
        row.set_active(initial_active)
        row.set_sensitive(initial_sensitivity)
        return row

    def _create_preview_group(self) -> Adw.PreferencesGroup:
        """Create preview preference group
        
        Returns:
            Preferences group for filename preview
        """
        preview_group = Adw.PreferencesGroup()
        preview_group.set_title(_("Preview"))

        # Create a preview row with custom widget
        preview_row = Adw.ActionRow()
        preview_row.set_title(_("Filename Example:"))

        # Sample file preview
        preview_value = Gtk.Label()
        preview_value.set_halign(Gtk.Align.END)
        preview_value.add_css_class("monospace")
        preview_value.add_css_class("caption")
        preview_value.add_css_class("dim-label")
        preview_row.add_suffix(preview_value)

        preview_group.add(preview_row)

        # Store reference
        preview_group.preview_value = preview_value

        return preview_group

    def _setup_pdf_options_callbacks(self, dialog: Adw.Window, prefs_page: Adw.PreferencesPage, 
                                   callback: Callable) -> None:
        """Set up all callbacks for PDF options dialog
        
        Args:
            dialog: The dialog window
            prefs_page: The preferences page
            callback: The completion callback
        """
        # Get all groups and their components
        file_group = prefs_page.file_group
        text_group = prefs_page.text_group
        date_group = prefs_page.date_group
        preview_group = prefs_page.preview_group

        # Set up preview update function
        def update_preview(*args):
            self._update_filename_preview(file_group, date_group, preview_group)

        def update_date_options_sensitivity(*args):
            is_date_enabled = date_group.include_date_row.get_active()
            date_group.year_row.set_sensitive(is_date_enabled)
            date_group.month_row.set_sensitive(is_date_enabled)
            date_group.day_row.set_sensitive(is_date_enabled)
            date_group.time_row.set_sensitive(is_date_enabled)
            date_group.format_row.set_sensitive(is_date_enabled)
            update_preview()

        def update_text_options_sensitivity(*args):
            is_save_txt = text_group.save_txt_row.get_active()
            is_separate_folder = text_group.separate_folder_row.get_active()
            
            text_group.separate_folder_row.set_sensitive(is_save_txt)
            text_group.text_folder_row.set_sensitive(is_save_txt and is_separate_folder)
            text_group.text_folder_row.folder_button.set_sensitive(is_save_txt and is_separate_folder)

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

        # Folder selection callback
        def select_text_folder(*args):
            self._show_folder_selection_dialog(text_group.text_folder_row.folder_label)

        text_group.text_folder_row.folder_button.connect("clicked", select_text_folder)

        # Save button callback
        def on_save_button_clicked(button):
            self._save_pdf_options(dialog, file_group, text_group, date_group, callback)

        # Find and connect save button
        header_bar = dialog.get_content().get_top_bar()
        header_bar.save_button.connect("clicked", on_save_button_clicked)

        # Initial updates
        update_preview()
        update_date_options_sensitivity()
        update_text_options_sensitivity()

    def _update_filename_preview(self, file_group: Adw.PreferencesGroup, 
                               date_group: Adw.PreferencesGroup, 
                               preview_group: Adw.PreferencesGroup) -> None:
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

    def _format_date_for_preview(self, date_group: Adw.PreferencesGroup, now: time.struct_time) -> str:
        """Format date string for preview
        
        Args:
            date_group: Date settings group
            now: Current time struct
            
        Returns:
            Formatted date string
        """
        date_parts = []
        selected_format = date_group.format_row.get_selected()

        # Get date components to include
        need_year = date_group.year_row.get_active()
        need_month = date_group.month_row.get_active()
        need_day = date_group.day_row.get_active()

        # Format based on selected format
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

        date_str = "-".join(date_parts)

        # Add time if enabled
        if date_group.time_row.get_active():
            time_str = f"{now.tm_hour:02d}{now.tm_min:02d}"
            if date_parts:
                date_str = f"{date_str}-{time_str}"
            else:
                date_str = time_str

        return date_str

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

    def _save_pdf_options(self, dialog: Adw.Window, file_group: Adw.PreferencesGroup,
                         text_group: Adw.PreferencesGroup, date_group: Adw.PreferencesGroup,
                         callback: Callable) -> None:
        """Save PDF options and close dialog
        
        Args:
            dialog: The dialog window
            file_group: File settings group
            text_group: Text extraction group
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

    def show_extracted_text(self, file_path: str) -> None:
        """Display the extracted text from a PDF file in a dialog

        Args:
            file_path: Path to the PDF file
        """
        # Get extracted text
        extracted_text = self._get_extracted_text_for_file(file_path)

        # Create main dialog window
        dialog = self._create_text_viewer_dialog(file_path)
        
        # Create dialog content
        dialog_content = self._create_text_viewer_content(extracted_text)
        dialog.set_child(dialog_content)

        # Set up keyboard shortcut after dialog is ready
        search_box = dialog_content.get_first_child()  # The search box
        search_entry = search_box.get_first_child()     # The search entry
        if hasattr(search_entry, 'keyboard_setup'):
            search_entry.keyboard_setup(dialog)

        # Show the dialog
        dialog.present()

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file with fallback options
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        # Initialize extracted text dictionary if needed
        if not hasattr(self.window.settings, "extracted_text"):
            self.window.settings.extracted_text = {}

        # Check if we have text stored in memory
        if file_path in self.window.settings.extracted_text:
            text = self.window.settings.extracted_text[file_path]
            if text and text.strip():
                return text

        # Try to read from sidecar file
        sidecar_text = self._read_from_sidecar_file(file_path)
        if sidecar_text:
            self.window.settings.extracted_text[file_path] = sidecar_text
            return sidecar_text

        # Try to read from temporary file
        temp_text = self._read_from_temp_file(file_path)
        if temp_text:
            self.window.settings.extracted_text[file_path] = temp_text
            return temp_text

        # Provide generic fallback message
        fallback_text = _("OCR processing was completed for this file, but the extracted text could not be found.")
        self.window.settings.extracted_text[file_path] = fallback_text
        return fallback_text

    def _read_from_sidecar_file(self, file_path: str) -> Optional[str]:
        """Try to read text from a sidecar file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Text content or None if not found
        """
        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, "r", encoding="utf-8") as f:
                    text = f.read()
                logger.info(f"Read {len(text)} characters from sidecar file")
                return text
            except Exception as e:
                logger.error(f"Error reading sidecar file: {e}")
        return None

    def _read_from_temp_file(self, file_path: str) -> Optional[str]:
        """Try to read text from a temporary file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Text content or None if not found
        """
        temp_dir = os.path.join(os.path.dirname(file_path), ".temp")
        if os.path.exists(temp_dir):
            temp_filename = f"temp_{os.path.basename(os.path.splitext(file_path)[0])}.txt"
            temp_sidecar = os.path.join(temp_dir, temp_filename)
            if os.path.exists(temp_sidecar):
                try:
                    with open(temp_sidecar, "r", encoding="utf-8") as f:
                        text = f.read()
                    logger.info(f"Found text in temporary file: {temp_sidecar}")
                    return text
                except Exception as e:
                    logger.error(f"Error reading temp sidecar file: {e}")
        return None

    def _create_text_viewer_dialog(self, file_path: str) -> Gtk.Window:
        """Create the main text viewer dialog window
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            The dialog window
        """
        dialog = Gtk.Window()
        dialog.set_title(f"Extracted Text - {os.path.basename(file_path)}")
        dialog.set_default_size(700, 500)
        dialog.set_modal(True)
        dialog.set_transient_for(self.window)
        return dialog

    def _create_text_viewer_content(self, extracted_text: str) -> Gtk.Box:
        """Create the content for text viewer dialog
        
        Args:
            extracted_text: The text to display
            
        Returns:
            Main content box
        """
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content_box.set_margin_top(16)
        content_box.set_margin_bottom(16)
        content_box.set_margin_start(16)
        content_box.set_margin_end(16)

        # Add search functionality
        search_box, search_entry, prev_button, next_button = self._create_search_box()
        content_box.append(search_box)

        # Add text view
        scrolled, text_view = self._create_text_view(extracted_text)
        content_box.append(scrolled)

        # Add status and action buttons
        status_box = self._create_text_viewer_status_box(extracted_text)
        content_box.append(status_box)

        # Set up search functionality
        self._setup_text_search(search_entry, prev_button, next_button, text_view, status_box)

        return content_box

    def _create_search_box(self) -> tuple[Gtk.Box, Gtk.SearchEntry, Gtk.Button, Gtk.Button]:
        """Create search box for text viewer
        
        Returns:
            Tuple of (search_box, search_entry, prev_button, next_button)
        """
        search_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_box.set_margin_bottom(8)

        search_entry = Gtk.SearchEntry()
        search_entry.set_placeholder_text(_("Search in text..."))
        search_entry.set_hexpand(True)
        search_box.append(search_entry)

        # Previous/next search buttons
        prev_button = Gtk.Button()
        prev_button.set_icon_name("go-up-symbolic")
        prev_button.set_tooltip_text(_("Find previous match"))
        prev_button.add_css_class("flat")
        prev_button.set_sensitive(False)
        search_box.append(prev_button)

        next_button = Gtk.Button()
        next_button.set_icon_name("go-down-symbolic")
        next_button.set_tooltip_text(_("Find next match"))
        next_button.add_css_class("flat")
        next_button.set_sensitive(False)
        search_box.append(next_button)

        return search_box, search_entry, prev_button, next_button

    def _create_text_view(self, extracted_text: str) -> tuple[Gtk.ScrolledWindow, Gtk.TextView]:
        """Create text view for displaying extracted text
        
        Args:
            extracted_text: Text to display
            
        Returns:
            Tuple of (scrolled_window, text_view)
        """
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)

        text_view = Gtk.TextView()
        text_view.set_editable(False)
        text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        text_view.set_monospace(True)
        text_view.set_left_margin(12)
        text_view.set_right_margin(12)
        text_view.set_top_margin(12)
        text_view.set_bottom_margin(12)

        # Set text content
        buffer = text_view.get_buffer()
        buffer.set_text(extracted_text)

        # Add search highlight tag
        tag_table = buffer.get_tag_table()
        highlight_tag = Gtk.TextTag.new("search_highlight")
        highlight_tag.set_property("background", "#ffff00")
        highlight_tag.set_property("foreground", "#000000")
        tag_table.add(highlight_tag)

        scrolled.set_child(text_view)
        return scrolled, text_view

    def _create_text_viewer_status_box(self, extracted_text: str) -> Gtk.Box:
        """Create status and action buttons box
        
        Args:
            extracted_text: The extracted text
            
        Returns:
            Box containing status and buttons
        """
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        status_box.set_margin_top(16)

        # Status label for search results
        status_label = Gtk.Label()
        status_label.set_halign(Gtk.Align.START)
        status_label.add_css_class("caption")
        status_label.add_css_class("dim-label")
        status_label.set_hexpand(True)
        status_box.append(status_label)

        # Action buttons
        button_box = self._create_text_viewer_buttons(extracted_text)
        status_box.append(button_box)

        # Store status label reference
        status_box.status_label = status_label

        return status_box

    def _create_text_viewer_buttons(self, extracted_text: str) -> Gtk.Box:
        """Create action buttons for text viewer
        
        Args:
            extracted_text: The extracted text
            
        Returns:
            Box containing action buttons
        """
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        button_box.set_halign(Gtk.Align.END)

        # Copy to clipboard button
        copy_button = Gtk.Button(label=_("Copy to Clipboard"))
        copy_button.set_tooltip_text(_("Copy the entire text to clipboard"))
        copy_button.connect("clicked", lambda b: self._copy_text_to_clipboard(extracted_text))
        button_box.append(copy_button)

        # Save to file button  
        save_button = Gtk.Button(label=_("Save as TXT"))
        save_button.set_tooltip_text(_("Save the extracted text to a .txt file"))
        save_button.connect("clicked", lambda b: self._save_text_to_file(extracted_text))
        button_box.append(save_button)

        # Close button
        close_button = Gtk.Button(label=_("Close"))
        close_button.add_css_class("suggested-action")
        close_button.connect("clicked", lambda b: button_box.get_root().destroy())
        button_box.append(close_button)

        return button_box

    def _setup_text_search(self, search_entry: Gtk.SearchEntry, prev_button: Gtk.Button,
                        next_button: Gtk.Button, text_view: Gtk.TextView, 
                        status_box: Gtk.Box) -> None:
        """Set up text search functionality
        
        Args:
            search_entry: Search entry widget
            prev_button: Previous button
            next_button: Next button
            text_view: Text view widget
            status_box: Status box containing status label
        """
        buffer = text_view.get_buffer()
        search_positions = []
        current_match = -1

        def on_search_changed(entry):
            nonlocal search_positions, current_match
            search_text = entry.get_text().lower()

            # Clear previous highlights
            start_iter = buffer.get_start_iter()
            end_iter = buffer.get_end_iter()
            buffer.remove_tag_by_name("search_highlight", start_iter, end_iter)

            # Reset search state
            search_positions = []
            current_match = -1

            if not search_text:
                prev_button.set_sensitive(False)
                next_button.set_sensitive(False)
                status_box.status_label.set_text("")
                return

            # Find all occurrences
            full_text = buffer.get_text(start_iter, end_iter, False).lower()
            pos = 0
            while True:
                pos = full_text.find(search_text, pos)
                if pos == -1:
                    break
                search_positions.append(pos)
                pos += 1

            # Highlight all matches
            for pos in search_positions:
                start_iter = buffer.get_iter_at_offset(pos)
                end_iter = buffer.get_iter_at_offset(pos + len(search_text))
                buffer.apply_tag_by_name("search_highlight", start_iter, end_iter)

            # Update UI
            match_count = len(search_positions)
            status_box.status_label.set_text(_("{0} matches found").format(match_count))
            prev_button.set_sensitive(match_count > 0)
            next_button.set_sensitive(match_count > 0)

            # Go to first match
            if match_count > 0:
                goto_match(0)

        def goto_match(index):
            nonlocal current_match
            if not search_positions or index < 0 or index >= len(search_positions):
                return

            current_match = index
            pos = search_positions[index]

            # Scroll to match
            match_iter = buffer.get_iter_at_offset(pos)
            text_view.scroll_to_iter(match_iter, 0.2, False, 0.0, 0.0)

            # Update status
            status_box.status_label.set_text(
                _("Match {0} of {1}").format(index + 1, len(search_positions))
            )

        def on_prev_clicked(_button):
            nonlocal current_match
            if current_match > 0:
                goto_match(current_match - 1)

        def on_next_clicked(_button):
            nonlocal current_match
            if current_match < len(search_positions) - 1:
                goto_match(current_match + 1)

        # Connect signals
        search_entry.connect("search-changed", on_search_changed)
        prev_button.connect("clicked", on_prev_clicked)
        next_button.connect("clicked", on_next_clicked)

        # Store the keyboard controller setup for later - will be added after dialog is shown
        def setup_keyboard_shortcut(dialog):
            key_controller = Gtk.EventControllerKey()
            
            def on_key_pressed(controller, keyval, keycode, state):
                if keyval == Gdk.KEY_f and state & Gdk.ModifierType.CONTROL_MASK:
                    search_entry.grab_focus()
                    return True
                return False

            key_controller.connect("key-pressed", on_key_pressed)
            dialog.add_controller(key_controller)
        
        # Store the setup function to be called later
        search_entry.keyboard_setup = setup_keyboard_shortcut

    def _copy_text_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard
        
        Args:
            text: Text to copy
        """
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        logger.info("Text copied to clipboard")

    def _save_text_to_file(self, text: str) -> None:
        """Save text to file with dialog
        
        Args:
            text: Text to save
        """
        # Create file save dialog
        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Save Extracted Text"))
        save_dialog.set_modal(True)
        save_dialog.set_initial_name("extracted_text.txt")

        # Show save dialog
        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_save_dialog_response(d, r, text)
        )

    def _on_save_dialog_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, text: str) -> None:
        """Handle save dialog response
        
        Args:
            dialog: File dialog
            result: Async result
            text: Text to save
        """
        try:
            file = dialog.save_finish(result)
            file_path = file.get_path()

            # Check if file exists
            if os.path.exists(file_path):
                self._show_file_exists_dialog(file_path, text)
                return

            # Save file
            self._write_text_to_file(file_path, text)

        except Exception as e:
            if "Dismissed" not in str(e):
                logger.error(f"Error saving text to file: {e}")
                self._show_error_dialog(_("Save Failed"), str(e))

    def _show_file_exists_dialog(self, file_path: str, text: str) -> None:
        """Show dialog for handling existing files
        
        Args:
            file_path: Path to existing file
            text: Text to save
        """
        dialog = Adw.MessageDialog(
            transient_for=self.window,
            heading=_("File Already Exists"),
            body=_("The file '{0}' already exists. What would you like to do?").format(
                os.path.basename(file_path)
            ),
        )

        dialog.add_response("overwrite", _("Overwrite"))
        dialog.add_response("rename", _("Auto-Rename"))
        dialog.add_response("cancel", _("Cancel"))

        dialog.set_response_appearance("overwrite", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_response_appearance("rename", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect("response", self._on_file_exists_response, file_path, text)
        dialog.present()

    def _on_file_exists_response(self, dialog: Adw.MessageDialog, response: str, 
                               file_path: str, text: str) -> None:
        """Handle file exists dialog response
        
        Args:
            dialog: Message dialog
            response: Response ID
            file_path: File path
            text: Text to save
        """
        if response == "overwrite":
            self._write_text_to_file(file_path, text)
        elif response == "rename":
            new_path = self._generate_unique_filename(file_path)
            self._write_text_to_file(new_path, text)

    def _generate_unique_filename(self, file_path: str) -> str:
        """Generate unique filename by appending number
        
        Args:
            file_path: Original file path
            
        Returns:
            Unique file path
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        counter = 1
        new_path = file_path

        while os.path.exists(new_path):
            new_path = os.path.join(dir_name, f"{name}-{counter}{ext}")
            counter += 1

        return new_path

    def _write_text_to_file(self, file_path: str, text: str) -> None:
        """Write text to file
        
        Args:
            file_path: Path to save file
            text: Text content
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info(f"Text saved to {file_path}")
            self._show_success_toast(f"Text saved to {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Error writing text to file: {e}")
            self._show_error_dialog(_("Save Failed"), str(e))

    def _show_success_toast(self, message: str) -> None:
        """Show success toast notification
        
        Args:
            message: Message to display
        """
        if hasattr(self.window, "toast_overlay") and self.window.toast_overlay:
            toast = Adw.Toast.new(message)
            toast.set_timeout(3)
            self.window.toast_overlay.add_toast(toast)

    def _show_error_dialog(self, title: str, message: str) -> None:
        """Show error dialog
        
        Args:
            title: Dialog title
            message: Error message
        """
        error_dialog = Adw.MessageDialog(transient_for=self.window)
        error_dialog.set_heading(title)
        error_dialog.set_body(message)
        error_dialog.add_response("ok", _("OK"))
        error_dialog.present()