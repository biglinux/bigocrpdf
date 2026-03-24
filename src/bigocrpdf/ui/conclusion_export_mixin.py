"""Conclusion Page ODF Export Mixin."""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class ConclusionExportMixin:
    """Mixin providing ODF export functionality for the conclusion page."""

    _MODE_FILENAME_SUFFIX = {
        "formatted_images": "_fmt_img",
        "formatted": "_fmt",
    }

    def _show_odf_export_options_dialog(self, file_path: str) -> None:
        """Show export options dialog with a single Export button and option switches.

        Args:
            file_path: Path to the source PDF file
        """
        dialog = Adw.Dialog()
        dialog.set_title(_("Export to OpenDocument"))
        dialog.set_content_width(380)

        toolbar_view = Adw.ToolbarView()
        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)

        # --- Options group ---
        options_group = Adw.PreferencesGroup()

        settings = self.window.settings
        init_images = getattr(settings, "odf_include_images", True)
        init_open = getattr(settings, "odf_open_after_export", False)
        switch_state = {"images": init_images, "open_after": init_open}

        images_row = Adw.SwitchRow()
        images_row.set_title(_("Include images"))
        images_row.set_subtitle(_("Embed page images alongside text"))
        images_row.set_active(init_images)
        images_row.connect(
            "notify::active",
            lambda row, _p: self._update_odf_setting(
                "odf_include_images", row.get_active(), switch_state, "images"
            ),
        )
        options_group.add(images_row)

        open_row = Adw.SwitchRow()
        open_row.set_title(_("Open after export"))
        open_row.set_subtitle(_("Open file in the default application"))
        open_row.set_active(init_open)
        open_row.connect(
            "notify::active",
            lambda row, _p: self._update_odf_setting(
                "odf_open_after_export", row.get_active(), switch_state, "open_after"
            ),
        )
        options_group.add(open_row)

        content_box.append(options_group)

        # --- Export button ---
        btn_content = Adw.ButtonContent()
        btn_content.set_icon_name("document-save-symbolic")
        btn_content.set_label(_("Export"))

        export_btn = Gtk.Button()
        export_btn.set_child(btn_content)
        export_btn.add_css_class("suggested-action")
        export_btn.add_css_class("pill")
        export_btn.set_halign(Gtk.Align.CENTER)
        set_a11y_label(export_btn, _("Export"))
        export_btn.connect(
            "clicked",
            lambda _b: self._on_export_clicked(
                switch_state["images"],
                switch_state["open_after"],
                file_path,
                dialog,
            ),
        )
        content_box.append(export_btn)

        toolbar_view.set_content(content_box)
        dialog.set_child(toolbar_view)

        dialog.present(self.window)

    def _update_odf_setting(self, attr: str, value: bool, switch_state: dict, key: str) -> None:
        """Persist an ODF export setting and update local switch state."""
        switch_state[key] = value
        settings = self.window.settings
        setattr(settings, attr, value)
        settings._save_odf_settings()
        settings._config.save()

    def _on_export_clicked(
        self,
        include_images: bool,
        open_after: bool,
        file_path: str,
        options_dialog: Adw.Dialog,
    ) -> None:
        """Handle click on the Export button.

        Args:
            include_images: Whether the Include images switch is active
            open_after: Whether to open the file after export
            file_path: Source PDF file path
            options_dialog: The options dialog to close
        """
        export_mode = "formatted_images" if include_images else "formatted"
        logger.info(f"ODF Export: mode={export_mode}, open_after={open_after}")

        self._odf_export_mode = export_mode
        self._odf_open_after = open_after

        options_dialog.force_close()
        self._show_odf_file_dialog(file_path)

    def _show_odf_file_dialog(self, file_path: str) -> None:
        """Show file save dialog for ODF export.

        Args:
            file_path: Source PDF file path
        """
        from gi.repository import Gio

        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Export to OpenDocument"))
        save_dialog.set_modal(True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        suffix = self._MODE_FILENAME_SUFFIX.get(
            getattr(self, "_odf_export_mode", "formatted"), "_fmt"
        )
        save_dialog.set_initial_name(f"{base_name}{suffix}.odt")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        odf_filter = Gtk.FileFilter()
        odf_filter.set_name(_("OpenDocument Text (*.odt)"))
        odf_filter.add_pattern("*.odt")
        odf_filter.add_mime_type("application/vnd.oasis.opendocument.text")
        filters.append(odf_filter)
        save_dialog.set_filters(filters)
        save_dialog.set_default_filter(odf_filter)

        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_odf_save_response(d, r, file_path),
        )

    def _on_odf_save_response(self, dialog: Gtk.FileDialog, result, file_path: str) -> None:
        """Handle ODF save dialog response.

        Args:
            dialog: File dialog
            result: Async result
            file_path: Source PDF file path
        """
        try:
            file = dialog.save_finish(result)
            output_path = file.get_path()
            if not output_path.lower().endswith(".odt"):
                output_path += ".odt"
            self._export_odf_file(output_path, file_path)
        except Exception as e:
            if "Dismissed" not in str(e):
                logger.error(f"Error exporting to ODF: {e}")
                self.window.show_toast(_("Export failed"))

    def _export_odf_file(self, output_path: str, file_path: str) -> None:
        """Export content to ODF using the TSV-based formatted converter.

        Runs the conversion in a background thread while showing a loading dialog.

        Args:
            output_path: Destination ODF file path
            file_path: Source PDF file path (with OCR text layer)
        """
        import threading

        from gi.repository import GLib

        cancel_event = threading.Event()

        # Build and show the loading dialog
        loading_dialog = Adw.Dialog()
        loading_dialog.set_title(_("Exporting…"))
        loading_dialog.set_content_width(300)
        loading_dialog.set_can_close(False)

        toolbar_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_show_start_title_buttons(False)
        header.set_show_end_title_buttons(False)
        toolbar_view.add_top_bar(header)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(32)
        box.set_margin_end(32)
        box.set_margin_top(24)
        box.set_margin_bottom(32)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)

        spinner = Gtk.Spinner()
        spinner.set_size_request(48, 48)
        spinner.start()
        spinner.set_halign(Gtk.Align.CENTER)
        box.append(spinner)

        label = Gtk.Label(label=_("Exporting to OpenDocument…"))
        label.add_css_class("title-4")
        box.append(label)

        cancel_btn = Gtk.Button(label=_("Cancel"))
        cancel_btn.add_css_class("destructive-action")
        cancel_btn.add_css_class("pill")
        cancel_btn.set_halign(Gtk.Align.CENTER)
        cancel_btn.set_margin_top(8)
        set_a11y_label(cancel_btn, _("Cancel"))
        cancel_btn.connect("clicked", lambda _b: cancel_event.set())
        box.append(cancel_btn)

        toolbar_view.set_content(box)
        loading_dialog.set_child(toolbar_view)
        loading_dialog.present(self.window)

        include_images = getattr(self, "_odf_export_mode", "formatted") == "formatted_images"

        def _do_export() -> None:
            from bigocrpdf.utils.odf_builder import ExportCancelled

            success = False
            cancelled = False
            try:
                from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_odf

                convert_pdf_to_odf(
                    file_path,
                    output_path,
                    include_images=include_images,
                    cancel_event=cancel_event,
                )
                success = True
            except ExportCancelled:
                cancelled = True
                logger.info("ODF export cancelled by user")
            except Exception as e:
                logger.error(f"ODF conversion failed: {e}")

            # Clean up partial file on cancel or failure
            if not success and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

            GLib.idle_add(self._on_export_finished, loading_dialog, success, cancelled, output_path)

        threading.Thread(target=_do_export, daemon=True).start()

    def _on_export_finished(
        self, dialog: Adw.Dialog, success: bool, cancelled: bool, output_path: str
    ) -> None:
        """Handle export thread completion on the main thread."""
        dialog.force_close()
        if cancelled:
            self.window.show_toast(_("Export cancelled"))
        else:
            self._report_export_result(success, output_path)
        return False

    def _report_export_result(self, success: bool, output_path: str) -> None:
        """Report the export result to user and optionally open the file.

        Args:
            success: Whether export succeeded
            output_path: Destination file path
        """
        if success:
            self.window.show_toast(_("Exported to {}").format(os.path.basename(output_path)))
            if getattr(self, "_odf_open_after", False):
                from bigocrpdf.utils.pdf_utils import open_file_with_default_app

                open_file_with_default_app(output_path)
        else:
            self.window.show_toast(_("Export failed"))

    def _open_file(self, file_path: str) -> None:
        """Open a file using the default application.

        Args:
            file_path: Path to the file to open
        """
        from bigocrpdf.utils.pdf_utils import open_file_with_default_app

        open_file_with_default_app(file_path)

    def _reveal_in_file_manager(self, file_path: str) -> None:
        """Open the system file manager with the given file selected.

        Uses the freedesktop.org FileManager1 D-Bus interface (ShowItems),
        which is supported by Dolphin, Nautilus, Thunar, Nemo, Caja, etc.
        Falls back to opening the parent directory if D-Bus is unavailable.
        """
        import subprocess

        file_uri = f"file://{os.path.abspath(file_path)}"
        try:
            subprocess.Popen(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:{file_uri}",
                    "string:",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # dbus-send not available; open parent folder instead
            from bigocrpdf.utils.pdf_utils import open_file_with_default_app

            open_file_with_default_app(os.path.dirname(file_path))

    def _show_extracted_text(self, file_path: str) -> None:
        """Show extracted text dialog.

        Args:
            file_path: Path to the PDF file
        """
        if hasattr(self.window, "ui") and hasattr(self.window.ui, "show_extracted_text"):
            self.window.ui.show_extracted_text(file_path)
        else:
            logger.warning("Text viewer dialog not available")
            self._show_simple_text_dialog(file_path)

    def _show_simple_text_dialog(self, file_path: str) -> None:
        """Show a simple text dialog as fallback.

        Args:
            file_path: Path to the PDF file
        """
        extracted_text = self._get_extracted_text_for_file(file_path)
        dialog = Adw.AlertDialog(
            heading=_("Extracted Text"),
            body=extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present(self.window)

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text or placeholder message
        """
        if (
            hasattr(self.window.settings, "extracted_text")
            and file_path in self.window.settings.extracted_text
        ):
            return self.window.settings.extracted_text[file_path]

        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading sidecar file: {e}")

        return _("No text content was detected in this file during OCR processing.")

    def reset_page(self) -> None:
        """Reset the conclusion page to initial state."""
        if self.result_file_count:
            self.result_file_count.set_text("0")
        if self.result_page_count:
            self.result_page_count.set_text("0")
        if self.result_time:
            self.result_time.set_text("00:00")
        if self.result_file_size:
            self.result_file_size.set_text("0 KB")
        if self.result_size_change:
            self.result_size_change.set_text("--")
            self.result_size_change.remove_css_class("success")
            self.result_size_change.remove_css_class("warning")

        if self.output_list_box:
            self._clear_output_list()
