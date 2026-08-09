"""PDF output options dialog controller."""
# GTK containers carry related child references used by dialog callbacks.
# pyright: reportAttributeAccessIssue=false

import os
import re
import time
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk, Pango

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

_INVALID_SUFFIX_CHARS = re.compile(r'[/:*?"<>|\\]')
_DATE_ORDERS = (
    ("year", "month", "day"),
    ("day", "month", "year"),
    ("month", "day", "year"),
)
_DATE_POSITIONS = (
    {"year": 1, "month": 2, "day": 3},
    {"day": 1, "month": 2, "year": 3},
    {"month": 1, "day": 2, "year": 3},
)


def _sanitize_suffix(value: str) -> str:
    return _INVALID_SUFFIX_CHARS.sub("-", value.strip()) or "ocr"


class PDFOptionsController:
    """Build and present PDF output options."""

    def __init__(self, parent, settings) -> None:
        self._parent = parent
        self._settings = settings

    def show_pdf_options_dialog(self, callback: Callable[[bool], None]) -> None:
        """Present the PDF output settings dialog."""
        dialog = Adw.Dialog(
            title=_("PDF Output Options"),
            content_width=550,
            content_height=590,
        )
        toolbar_view = Adw.ToolbarView()
        header_bar = Adw.HeaderBar()
        header_bar.set_show_start_title_buttons(False)
        header_bar.set_show_end_title_buttons(False)

        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.connect("clicked", lambda _button: dialog.close())
        header_bar.pack_start(cancel_button)

        save_button = Gtk.Button(label=_("Save"))
        save_button.add_css_class("suggested-action")
        header_bar.pack_end(save_button)
        toolbar_view.add_top_bar(header_bar)

        prefs_page = self._create_pdf_preferences_page()
        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
            child=prefs_page,
        )
        preview_group = Adw.PreferencesGroup(title=_("Preview"))
        preview_group.set_margin_start(12)
        preview_group.set_margin_end(12)
        preview_group.set_margin_bottom(12)
        preview_row = Adw.ActionRow(title=_("Filename Example:"))
        preview_value = Gtk.Label(halign=Gtk.Align.END)
        preview_value.add_css_class("monospace")
        preview_value.add_css_class("caption")
        preview_value.add_css_class("dim-label")
        preview_row.add_suffix(preview_value)
        preview_group.add(preview_row)
        preview_group.preview_value = preview_value
        prefs_page.preview_group = preview_group

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.append(scrolled)
        content.append(preview_group)
        toolbar_view.set_content(content)
        dialog.set_child(toolbar_view)

        self._setup_callbacks(dialog, prefs_page, save_button, callback)
        dialog.present(self._parent)

    def _create_pdf_preferences_page(self) -> Adw.PreferencesPage:
        prefs_page = Adw.PreferencesPage()
        prefs_page.file_group = self._create_file_settings_group()
        prefs_page.text_group = self._create_text_extraction_group()
        prefs_page.odf_group = self._create_odf_extraction_group()
        prefs_page.date_group = self._create_date_time_group()
        prefs_page.add(prefs_page.file_group)
        prefs_page.add(prefs_page.text_group)
        prefs_page.add(prefs_page.odf_group)
        prefs_page.add(prefs_page.date_group)
        return prefs_page

    def _create_file_settings_group(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=_("File Settings"))

        use_original_name_row = Adw.SwitchRow(
            title=_("Use Original Filename"),
            subtitle=_("Use the same name as the original file"),
            active=self._settings.use_original_filename,
        )

        warning_row = Adw.ActionRow(
            title=_("Warning"),
            subtitle=_(
                "To replace original files, also enable 'Overwrite Existing Files' below "
                "and save to the same folder"
            ),
            visible=use_original_name_row.get_active(),
        )
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        warning_icon.set_pixel_size(16)
        warning_row.add_prefix(warning_icon)
        warning_row.add_css_class("warning-row")

        suffix_row = Adw.EntryRow(
            title=_("Filename Suffix"),
            text=self._settings.pdf_suffix or "ocr",
            show_apply_button=False,
            sensitive=not use_original_name_row.get_active(),
        )
        suffix_row.set_tooltip_text(
            _("Text added to the end of the filename (e.g. document-ocr.pdf)")
        )

        overwrite_row = Adw.SwitchRow(
            title=_("Overwrite Existing Files"),
            subtitle=_("Replace files with the same name"),
            active=self._settings.overwrite_existing,
        )
        overwrite_row.connect("notify::active", self._on_overwrite_toggled)

        for row in (use_original_name_row, warning_row, suffix_row, overwrite_row):
            group.add(row)

        group.use_original_name_row = use_original_name_row
        group.warning_row = warning_row
        group.suffix_row = suffix_row
        group.overwrite_row = overwrite_row
        return group

    def _on_overwrite_toggled(self, row: Adw.SwitchRow, _pspec) -> None:
        if not row.get_active():
            return

        dialog = Adw.AlertDialog(
            heading=_("Overwrite Existing Files?"),
            body=_(
                "If files with the same name already exist in the output "
                "folder, they will be permanently replaced. This cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("confirm", _("Enable"))
        dialog.set_response_appearance("confirm", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.connect(
            "response",
            lambda _dialog, response: row.set_active(False) if response != "confirm" else None,
        )
        dialog.present(self._parent)

    def _create_text_extraction_group(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=_("Text Extraction"))
        save_txt_row = Adw.SwitchRow(
            title=_("Save Text Files"),
            subtitle=_("Automatically save extracted text as .txt files"),
            active=self._settings.save_txt,
        )
        separate_folder_row = Adw.SwitchRow(
            title=_("Use Separate Folder for Text Files"),
            subtitle=_("Save text files to a different folder"),
            active=self._settings.separate_txt_folder,
            sensitive=save_txt_row.get_active(),
        )

        txt_folder = self._settings.txt_folder
        folder_label = Gtk.Label(
            label=txt_folder or _("Not set"),
            ellipsize=Pango.EllipsizeMode.START,
            halign=Gtk.Align.END,
            margin_end=8,
        )
        folder_button = Gtk.Button(
            icon_name="folder-symbolic",
            valign=Gtk.Align.CENTER,
            sensitive=save_txt_row.get_active() and separate_folder_row.get_active(),
        )
        folder_button.add_css_class("flat")
        folder_button.set_tooltip_text(_("Choose where to save the text files"))
        folder_button.update_property(
            [Gtk.AccessibleProperty.LABEL], [_("Choose where to save the text files")]
        )
        text_folder_row = Adw.ActionRow(
            title=_("Text Files Folder"),
            subtitle=_("Select where to save text files"),
            sensitive=folder_button.get_sensitive(),
        )
        text_folder_row.add_suffix(folder_label)
        text_folder_row.add_suffix(folder_button)
        text_folder_row.folder_label = folder_label
        text_folder_row.folder_button = folder_button

        for row in (save_txt_row, separate_folder_row, text_folder_row):
            group.add(row)

        group.save_txt_row = save_txt_row
        group.separate_folder_row = separate_folder_row
        group.text_folder_row = text_folder_row
        return group

    def _create_odf_extraction_group(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=_("ODF Export"))
        save_odf_row = Adw.SwitchRow(
            title=_("Save ODF Files"),
            subtitle=_("Automatically save extracted text as .odt files"),
            active=self._settings.save_odf,
        )
        include_images_row = Adw.SwitchRow(
            title=_("Preserve Text Layout"),
            subtitle=_("Turn off for easier paragraph, table, and column editing"),
            active=self._settings.odf_include_images,
            sensitive=save_odf_row.get_active(),
        )
        group.add(save_odf_row)
        group.add(include_images_row)
        group.save_odf_row = save_odf_row
        group.include_images_row = include_images_row
        return group

    def _create_date_time_group(self) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=_("Date and Time"))
        include_date_row = Adw.SwitchRow(
            title=_("Add Date to Filename"),
            subtitle=_("Include date elements in the filename"),
            active=self._settings.include_date,
        )
        enabled = include_date_row.get_active()

        format_row = Adw.ComboRow(title=_("Date Format"), sensitive=enabled)
        format_row.set_model(
            Gtk.StringList.new(
                [
                    _("Standard (2026-02-23)"),
                    _("European (23-02-2026)"),
                    _("American (02-23-2026)"),
                ]
            )
        )
        order = self._settings.date_format_order
        if order.get("day", 3) < order.get("month", 2) < order.get("year", 1):
            format_row.set_selected(1)
        elif order.get("month", 2) < order.get("day", 3) < order.get("year", 1):
            format_row.set_selected(2)
        else:
            format_row.set_selected(0)

        def component_row(title: str, subtitle: str, active: bool) -> Adw.SwitchRow:
            return Adw.SwitchRow(
                title=title,
                subtitle=subtitle,
                active=active,
                sensitive=enabled,
            )

        year_row = component_row(
            _("Include Year"), _("Add YYYY to the date"), self._settings.include_year
        )
        month_row = component_row(
            _("Include Month"), _("Add MM to the date"), self._settings.include_month
        )
        day_row = component_row(
            _("Include Day"), _("Add DD to the date"), self._settings.include_day
        )
        time_row = component_row(
            _("Include Time"), _("Add HHMM to the filename"), self._settings.include_time
        )

        for row in (include_date_row, format_row, year_row, month_row, day_row, time_row):
            group.add(row)

        group.include_date_row = include_date_row
        group.format_row = format_row
        group.year_row = year_row
        group.month_row = month_row
        group.day_row = day_row
        group.time_row = time_row
        return group

    def _setup_callbacks(
        self,
        dialog: Adw.Dialog,
        prefs_page: Adw.PreferencesPage,
        save_button: Gtk.Button,
        callback: Callable[[bool], None],
    ) -> None:
        file_group = prefs_page.file_group
        text_group = prefs_page.text_group
        odf_group = prefs_page.odf_group
        date_group = prefs_page.date_group
        preview_group = prefs_page.preview_group

        def update_preview(*_args) -> None:
            self._update_filename_preview(file_group, date_group, preview_group)

        def update_date_options(*_args) -> None:
            enabled = date_group.include_date_row.get_active()
            for row in (
                date_group.year_row,
                date_group.month_row,
                date_group.day_row,
                date_group.time_row,
                date_group.format_row,
            ):
                row.set_sensitive(enabled)
            update_preview()

        def update_text_options(*_args) -> None:
            save_txt = text_group.save_txt_row.get_active()
            separate = text_group.separate_folder_row.get_active()
            needs_folder = save_txt and separate
            folder = text_group.text_folder_row.folder_label.get_label()
            has_folder = folder != _("Not set") and os.path.isdir(folder)

            text_group.separate_folder_row.set_sensitive(save_txt)
            text_group.text_folder_row.set_sensitive(needs_folder)
            text_group.text_folder_row.folder_button.set_sensitive(needs_folder)
            save_button.set_sensitive(not needs_folder or has_folder)

        def update_odf_options(*_args) -> None:
            odf_group.include_images_row.set_sensitive(odf_group.save_odf_row.get_active())

        file_group.use_original_name_row.connect("notify::active", update_preview)
        file_group.suffix_row.connect("changed", update_preview)
        date_group.include_date_row.connect("notify::active", update_date_options)
        for row in (
            date_group.year_row,
            date_group.month_row,
            date_group.day_row,
            date_group.time_row,
        ):
            row.connect("notify::active", update_preview)
        date_group.format_row.connect("notify::selected", update_preview)
        text_group.save_txt_row.connect("notify::active", update_text_options)
        text_group.separate_folder_row.connect("notify::active", update_text_options)
        odf_group.save_odf_row.connect("notify::active", update_odf_options)
        text_group.text_folder_row.folder_button.connect(
            "clicked",
            lambda *_args: self._show_folder_selection_dialog(
                text_group.text_folder_row.folder_label,
                update_text_options,
            ),
        )
        save_button.connect(
            "clicked",
            lambda _button: self._save_pdf_options(
                dialog,
                file_group,
                text_group,
                odf_group,
                date_group,
                callback,
            ),
        )

        update_preview()
        update_date_options()
        update_text_options()
        update_odf_options()

    def _update_filename_preview(
        self,
        file_group: Adw.PreferencesGroup,
        date_group: Adw.PreferencesGroup,
        preview_group: Adw.PreferencesGroup,
    ) -> None:
        use_original = file_group.use_original_name_row.get_active()
        file_group.suffix_row.set_sensitive(not use_original)
        file_group.warning_row.set_visible(use_original)

        if use_original:
            preview_group.preview_value.set_text("original_document.pdf")
            return

        suffix = _sanitize_suffix(file_group.suffix_row.get_text())
        date = self._format_date_for_preview(date_group, time.localtime())
        preview_group.preview_value.set_text(f"document-{suffix}{f'-{date}' if date else ''}.pdf")

    @staticmethod
    def _format_date_for_preview(
        date_group: Adw.PreferencesGroup,
        now: time.struct_time,
    ) -> str:
        if not date_group.include_date_row.get_active():
            return ""

        values = {
            "year": f"{now.tm_year}" if date_group.year_row.get_active() else "",
            "month": f"{now.tm_mon:02d}" if date_group.month_row.get_active() else "",
            "day": f"{now.tm_mday:02d}" if date_group.day_row.get_active() else "",
        }
        selected = date_group.format_row.get_selected()
        order = _DATE_ORDERS[selected] if selected < len(_DATE_ORDERS) else _DATE_ORDERS[0]
        parts = [values[component] for component in order if values[component]]
        if date_group.time_row.get_active():
            parts.append(f"{now.tm_hour:02d}{now.tm_min:02d}")
        return "-".join(parts)

    def _show_folder_selection_dialog(
        self,
        folder_label: Gtk.Label,
        on_selected: Callable[[], None],
    ) -> None:
        dialog = Gtk.FileDialog(
            title=_("Select Folder for Text Files"),
            modal=True,
        )
        current_folder = folder_label.get_label()
        if current_folder != _("Not set") and os.path.isdir(current_folder):
            dialog.set_initial_folder(Gio.File.new_for_path(current_folder))
        dialog.select_folder(
            self._parent,
            None,
            lambda file_dialog, result: self._on_folder_selected(
                file_dialog, result, folder_label, on_selected
            ),
        )

    @staticmethod
    def _on_folder_selected(
        dialog: Gtk.FileDialog,
        result,
        folder_label: Gtk.Label,
        on_selected: Callable[[], None],
    ) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as error:
            cancelled = error.matches(
                Gtk.DialogError.quark(), Gtk.DialogError.CANCELLED
            ) or error.matches(Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED)
            if not cancelled:
                logger.error("Error selecting folder: %s", error)
            return

        path = folder.get_path()
        if path is None:
            logger.warning("Remote locations are not supported")
            return
        folder_label.set_label(path)
        on_selected()

    def _save_pdf_options(
        self,
        dialog: Adw.Dialog,
        file_group: Adw.PreferencesGroup,
        text_group: Adw.PreferencesGroup,
        odf_group: Adw.PreferencesGroup,
        date_group: Adw.PreferencesGroup,
        callback: Callable[[bool], None],
    ) -> None:
        save_txt = text_group.save_txt_row.get_active()
        separate_txt_folder = text_group.separate_folder_row.get_active()
        folder = text_group.text_folder_row.folder_label.get_label()
        txt_folder = "" if folder == _("Not set") else folder
        if save_txt and separate_txt_folder and not os.path.isdir(txt_folder):
            return

        self._settings.use_original_filename = file_group.use_original_name_row.get_active()
        self._settings.pdf_suffix = _sanitize_suffix(file_group.suffix_row.get_text())
        self._settings.overwrite_existing = file_group.overwrite_row.get_active()
        self._settings.save_txt = save_txt
        self._settings.separate_txt_folder = separate_txt_folder
        self._settings.txt_folder = txt_folder
        self._settings.save_odf = odf_group.save_odf_row.get_active()
        self._settings.odf_include_images = odf_group.include_images_row.get_active()
        self._settings.include_date = date_group.include_date_row.get_active()
        self._settings.include_year = date_group.year_row.get_active()
        self._settings.include_month = date_group.month_row.get_active()
        self._settings.include_day = date_group.day_row.get_active()
        self._settings.include_time = date_group.time_row.get_active()

        selected = date_group.format_row.get_selected()
        self._settings.date_format_order = (
            _DATE_POSITIONS[selected] if selected < len(_DATE_POSITIONS) else _DATE_POSITIONS[0]
        ).copy()
        if not self._settings._save_all_settings():
            return
        logger.info(_("PDF output settings saved"))
        dialog.close()
        callback(True)
