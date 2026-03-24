"""PDF Options Dialog UI Creation Mixin."""

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk, Pango

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _


class PDFOptionsUICreationMixin:
    """Mixin providing PDF options dialog UI creation methods."""

    def show_pdf_options_dialog(self, callback: Callable) -> None:
        """Show dialog with PDF output options before starting OCR

        Args:
            callback: Function to call with options when dialog is confirmed
        """
        # Create the options dialog
        dialog = Adw.Dialog()
        dialog.set_title(_("PDF Output Options"))
        dialog.set_content_width(550)
        dialog.set_content_height(590)

        # Set up the main structure
        dialog_content, prefs_page, header_bar = self._create_pdf_options_content(dialog, callback)
        dialog.set_child(dialog_content)

        # Set up callbacks for all widgets
        self._setup_pdf_options_callbacks(dialog, prefs_page, header_bar, callback)

        # Show the dialog
        dialog.present(self.window)

    def _create_pdf_options_content(self, dialog: Adw.Dialog, callback: Callable) -> tuple:
        """Create the content for PDF options dialog

        Args:
            dialog: The dialog window
            callback: Callback function

        Returns:
            Tuple of (toolbar_view, prefs_page, header_bar)
        """
        # Set up the Adwaita toolbar view structure (main container)
        toolbar_view = Adw.ToolbarView()

        # Create header bar
        header_bar = self._create_pdf_options_header(dialog, callback)
        toolbar_view.add_top_bar(header_bar)

        # Create main vertical box to hold scrolled content + fixed preview
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Create scrolled content with preferences page (without preview)
        scrolled_content, prefs_page = self._create_pdf_options_scrolled_content()
        main_box.append(scrolled_content)

        # Create fixed preview group at bottom (outside scrolled area)
        preview_group = self._create_preview_group()
        preview_group.set_margin_start(12)
        preview_group.set_margin_end(12)
        preview_group.set_margin_bottom(12)
        main_box.append(preview_group)

        # Store preview group reference in prefs_page for callback access
        prefs_page.preview_group = preview_group

        toolbar_view.set_content(main_box)

        return toolbar_view, prefs_page, header_bar

    def _create_pdf_options_header(self, dialog: Adw.Dialog, _callback: Callable) -> Adw.HeaderBar:
        """Create header bar for PDF options dialog

        Args:
            dialog: The dialog window
            _callback: Callback function (unused, connected later via save_button)

        Returns:
            The header bar widget
        """
        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(False)
        header_bar.set_title_widget(Gtk.Label(label=_("PDF Output Settings")))

        # Add cancel button to header
        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda _: dialog.close())
        set_a11y_label(cancel_button, _("Cancel"))
        header_bar.pack_start(cancel_button)

        # Add save button to header
        save_button = Gtk.Button(label=_("Save"))
        save_button.add_css_class("suggested-action")
        set_a11y_label(save_button, _("Save"))
        header_bar.pack_end(save_button)

        # Store save button reference for later connection
        header_bar.save_button = save_button

        return header_bar

    def _create_pdf_options_scrolled_content(self) -> tuple:
        """Create scrolled content for PDF options dialog

        Returns:
            Tuple of (scrolled_window, prefs_page)
        """
        # Create scrolled window for content
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)

        # Create preferences page with all options
        prefs_page = self._create_pdf_preferences_page()
        scrolled.set_child(prefs_page)

        return scrolled, prefs_page

    def _create_pdf_preferences_page(self) -> Adw.PreferencesPage:
        """Create the preferences page with all PDF options

        Returns:
            Preferences page with all option groups
        """
        prefs_page = Adw.PreferencesPage()

        # Add all preference groups (preview is now added separately outside scroll)
        file_group = self._create_file_settings_group()
        text_group = self._create_text_extraction_group()
        odf_group = self._create_odf_extraction_group()
        date_group = self._create_date_time_group()

        prefs_page.add(file_group)
        prefs_page.add(text_group)
        prefs_page.add(odf_group)
        prefs_page.add(date_group)

        # Store references for access in callbacks
        prefs_page.file_group = file_group
        prefs_page.text_group = text_group
        prefs_page.odf_group = odf_group
        prefs_page.date_group = date_group
        # Note: preview_group is added in _create_pdf_options_content()

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
        suffix_row.set_tooltip_text(
            _("Text added to the end of the filename (e.g. document-ocr.pdf)")
        )
        suffix_row.set_text(self.window.settings.pdf_suffix or "ocr")
        suffix_row.set_show_apply_button(False)
        suffix_row.set_sensitive(not use_orig_name)

        # Overwrite files switch row
        overwrite_row = Adw.SwitchRow()
        overwrite_row.set_title(_("Overwrite Existing Files"))
        overwrite_row.set_subtitle(_("Replace files with the same name"))
        overwrite_row.set_active(self.window.settings.overwrite_existing)
        overwrite_row.connect("notify::active", self._on_overwrite_toggled)

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
            _(
                "To replace original files, also enable 'Overwrite Existing Files' below "
                "and save to the same folder"
            )
        )
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        warning_icon.set_pixel_size(16)
        warning_row.add_prefix(warning_icon)
        warning_row.add_css_class("warning-row")
        return warning_row

    def _on_overwrite_toggled(self, row: Adw.SwitchRow, _pspec) -> None:
        """Show a confirmation when overwrite is enabled."""
        if not row.get_active():
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Overwrite Existing Files?"))
        dialog.set_body(
            _(
                "If files with the same name already exist in the output "
                "folder, they will be permanently replaced. This cannot be undone."
            )
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("confirm", _("Enable"))
        dialog.set_response_appearance("confirm", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")

        def on_response(_d, response):
            if response != "confirm":
                row.set_active(False)

        dialog.connect("response", on_response)
        dialog.present(self.window)

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
        folder_button.set_tooltip_text(_("Choose where to save the text files"))
        folder_button.update_property(
            [Gtk.AccessibleProperty.LABEL], [_("Choose where to save the text files")]
        )
        folder_button.set_sensitive(initial_sensitivity)

        text_folder_row.add_suffix(folder_label)
        text_folder_row.add_suffix(folder_button)

        # Store references
        text_folder_row.folder_label = folder_label
        text_folder_row.folder_button = folder_button

        return text_folder_row

    def _create_odf_extraction_group(self) -> Adw.PreferencesGroup:
        """Create ODF export preference group

        Returns:
            Preferences group for ODF export settings
        """
        odf_group = Adw.PreferencesGroup()
        odf_group.set_title(_("ODF Export"))

        # Save ODF files switch row
        save_odf_row = Adw.SwitchRow()
        save_odf_row.set_title(_("Save ODF Files"))
        save_odf_row.set_subtitle(_("Automatically save extracted text as .odt files"))
        save_odf = getattr(self.window.settings, "save_odf", False)
        save_odf_row.set_active(save_odf)

        # Include images in ODF switch row
        include_images_row = Adw.SwitchRow()
        include_images_row.set_title(_("Include Page Images"))
        include_images_row.set_subtitle(_("Embed original page images in ODF"))
        include_images = getattr(self.window.settings, "odf_include_images", True)
        include_images_row.set_active(include_images)
        include_images_row.set_sensitive(save_odf)

        # Use formatting in ODF switch row
        use_formatting_row = Adw.SwitchRow()
        use_formatting_row.set_title(_("Use Structured Formatting"))
        use_formatting_row.set_subtitle(_("Apply detected layout and text styles"))
        use_formatting = getattr(self.window.settings, "odf_use_formatting", True)
        use_formatting_row.set_active(use_formatting)
        use_formatting_row.set_sensitive(save_odf)

        # Add all rows to group
        odf_group.add(save_odf_row)
        odf_group.add(include_images_row)
        odf_group.add(use_formatting_row)

        # Store references
        odf_group.save_odf_row = save_odf_row
        odf_group.include_images_row = include_images_row
        odf_group.use_formatting_row = use_formatting_row

        return odf_group

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
            _("Include Year"),
            _("Add YYYY to the date"),
            self.window.settings.include_year,
            include_date_row.get_active(),
        )
        month_row = self._create_date_component_row(
            _("Include Month"),
            _("Add MM to the date"),
            self.window.settings.include_month,
            include_date_row.get_active(),
        )
        day_row = self._create_date_component_row(
            _("Include Day"),
            _("Add DD to the date"),
            self.window.settings.include_day,
            include_date_row.get_active(),
        )
        time_row = self._create_date_component_row(
            _("Include Time"),
            _("Add HHMM to the filename"),
            self.window.settings.include_time,
            include_date_row.get_active(),
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
        format_model.append(_("Standard (2026-02-23)"))
        format_model.append(_("European (23-02-2026)"))
        format_model.append(_("American (02-23-2026)"))
        format_row.set_model(format_model)
        format_row.set_sensitive(initial_sensitivity)

        # Set default format based on settings
        order = getattr(
            self.window.settings, "date_format_order", {"year": 1, "month": 2, "day": 3}
        )
        if order["day"] < order["month"] and order["month"] < order["year"]:
            format_row.set_selected(1)  # DD-MM-YYYY
        elif order["month"] < order["day"] and order["day"] < order["year"]:
            format_row.set_selected(2)  # MM-DD-YYYY
        else:
            format_row.set_selected(0)  # YYYY-MM-DD

        return format_row

    def _create_date_component_row(
        self, title: str, subtitle: str, initial_active: bool, initial_sensitivity: bool
    ) -> Adw.SwitchRow:
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
