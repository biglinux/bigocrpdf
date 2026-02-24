"""Sidebar tool dialogs for PDFEditorWindow: compress, split, save-copy, reverse."""

from __future__ import annotations

import os
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class EditorToolsMixin:
    """Mixin providing document-level tool dialogs for the PDF editor."""

    def _on_save_copy(self, _action, _param) -> None:
        """Save included pages as a new PDF file."""
        if not self._document:
            return

        dialog = Gtk.FileDialog()
        dialog.set_title(_("Save a copy"))
        dialog.set_initial_name(os.path.basename(self._pdf_path))

        pdf_filter = Gtk.FileFilter()
        pdf_filter.set_name(_("PDF Files"))
        pdf_filter.add_pattern("*.pdf")
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(pdf_filter)
        dialog.set_filters(store)

        dialog.save(self, None, self._on_save_copy_response)

    def _on_save_copy_response(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle Save Copy dialog response."""
        try:
            file = dialog.save_finish(result)
            if file:
                path = file.get_path()
                if path:
                    from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

                    if apply_changes_to_pdf(self._document, path):
                        logger.info("Saved PDF copy to %s", path)
                        self._show_info(_("Saved to {}").format(os.path.basename(path)))
                    else:
                        self._show_error(_("Failed to save PDF."))
        except GLib.Error as e:
            if "dismissed" not in str(e).lower():
                logger.error("Save copy error: %s", e)

    def _on_tools_compress(self, _action, _param) -> None:
        """Show compress dialog and compress the document."""
        if not self._document:
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Compress PDF"))
        dialog.set_body(
            _(
                "Reduce the file size by compressing the images inside the PDF. "
                "Lower values produce smaller files but with less image detail."
            )
        )

        quality_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        quality_box.set_halign(Gtk.Align.CENTER)

        quality_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        quality_row.set_halign(Gtk.Align.CENTER)
        quality_label = Gtk.Label(label=_("Image Quality:"))
        quality_spin = Gtk.SpinButton.new_with_range(10, 95, 5)
        quality_spin.set_value(60)
        quality_row.append(quality_label)
        quality_row.append(quality_spin)
        quality_box.append(quality_row)

        quality_hint = Gtk.Label(
            label=_("10 = smallest file, 95 = best quality. 60 is a good default.")
        )
        quality_hint.add_css_class("dim-label")
        quality_hint.add_css_class("caption")
        quality_box.append(quality_hint)

        dpi_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        dpi_box.set_halign(Gtk.Align.CENTER)

        dpi_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dpi_row.set_halign(Gtk.Align.CENTER)
        dpi_label = Gtk.Label(label=_("Image Resolution (DPI):"))
        dpi_spin = Gtk.SpinButton.new_with_range(72, 600, 10)
        dpi_spin.set_value(150)
        dpi_row.append(dpi_label)
        dpi_row.append(dpi_spin)
        dpi_box.append(dpi_row)

        dpi_hint = Gtk.Label(
            label=_("72 = screen only, 150 = digital reading, 300 = print quality.")
        )
        dpi_hint.add_css_class("dim-label")
        dpi_hint.add_css_class("caption")
        dpi_box.append(dpi_hint)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.append(quality_box)
        content_box.append(dpi_box)
        dialog.set_extra_child(content_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("compress", _("Compress"))
        dialog.set_response_appearance("compress", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect(
            "response",
            lambda d, r: (
                self._do_compress(int(quality_spin.get_value()), int(dpi_spin.get_value()))
                if r == "compress"
                else None
            ),
        )
        dialog.present(self)

    def _do_compress(self, quality: int, dpi: int) -> None:
        """Execute PDF compression."""
        if not self._document:
            return

        from bigocrpdf.services.pdf_operations import compress_pdf
        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf
        from bigocrpdf.utils.temp_manager import mkstemp as _mkstemp
        from bigocrpdf.utils.temp_manager import remove_file as _rmfile

        fd, tmp_edited = _mkstemp(suffix=".pdf", prefix="bigocr_edit_")
        os.close(fd)

        if not apply_changes_to_pdf(self._document, tmp_edited):
            self._show_error(_("Failed to prepare document for compression."))
            _rmfile(tmp_edited)
            return

        fd2, tmp_compressed = _mkstemp(suffix=".pdf", prefix="bigocr_cmp_")
        os.close(fd2)

        result = compress_pdf(tmp_edited, tmp_compressed, image_quality=quality, image_dpi=dpi)
        _rmfile(tmp_edited)

        if result.success:
            dialog = Gtk.FileDialog()
            dialog.set_title(_("Save Compressed PDF"))
            dialog.set_initial_name("compressed_" + os.path.basename(self._pdf_path))

            pdf_filter = Gtk.FileFilter()
            pdf_filter.set_name(_("PDF Files"))
            pdf_filter.add_pattern("*.pdf")
            store = Gio.ListStore.new(Gtk.FileFilter)
            store.append(pdf_filter)
            dialog.set_filters(store)

            dialog.save(
                self,
                None,
                lambda d, r: self._finish_compress_save(d, r, tmp_compressed, result.message),
            )
        else:
            os.unlink(tmp_compressed)
            self._show_error(_("Compression failed: {}").format(result.message))

    def _finish_compress_save(self, dialog, result, tmp_path, message) -> None:
        """Finish saving the compressed file."""
        import shutil

        try:
            file = dialog.save_finish(result)
            if file:
                path = file.get_path()
                if path:
                    shutil.move(tmp_path, path)
                    self._show_info(message)
                    return
        except GLib.Error:
            pass

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def _on_tools_split_pages(self, _action, _param) -> None:
        """Show split-by-pages dialog."""
        if not self._document:
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Split by Page Count"))
        dialog.set_body(_("Split the document into parts with a fixed number of pages each."))

        spin_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        spin_box.set_halign(Gtk.Align.CENTER)
        spin_label = Gtk.Label(label=_("Pages per file:"))
        spin = Gtk.SpinButton.new_with_range(1, 9999, 1)
        spin.set_value(5)
        spin_box.append(spin_label)
        spin_box.append(spin)
        dialog.set_extra_child(spin_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("split", _("Split"))
        dialog.set_response_appearance("split", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect(
            "response",
            lambda d, r: self._do_split_by_pages(int(spin.get_value())) if r == "split" else None,
        )
        dialog.present(self)

    def _do_split_by_pages(self, pages_per_file: int) -> None:
        """Execute split by page count."""
        self._pick_output_dir_and_split("pages", pages_per_file)

    def _on_tools_split_size(self, _action, _param) -> None:
        """Show split-by-size dialog."""
        if not self._document:
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Split by File Size"))
        dialog.set_body(_("Split the document so each part is at most the specified size."))

        spin_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        spin_box.set_halign(Gtk.Align.CENTER)
        spin_label = Gtk.Label(label=_("Max size (MB):"))
        spin = Gtk.SpinButton.new_with_range(1, 500, 1)
        spin.set_value(10)
        spin_box.append(spin_label)
        spin_box.append(spin)
        dialog.set_extra_child(spin_box)

        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("split", _("Split"))
        dialog.set_response_appearance("split", Adw.ResponseAppearance.SUGGESTED)

        dialog.connect(
            "response",
            lambda d, r: self._do_split_by_size(float(spin.get_value())) if r == "split" else None,
        )
        dialog.present(self)

    def _do_split_by_size(self, max_mb: float) -> None:
        """Execute split by file size."""
        self._pick_output_dir_and_split("size", max_mb)

    def _pick_output_dir_and_split(self, mode: str, value: float) -> None:
        """Let user pick an output directory, then run the split."""
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Select Output Directory"))

        dialog.select_folder(
            self,
            None,
            lambda d, r: self._finish_split(d, r, mode, value),
        )

    def _finish_split(self, dialog, result, mode: str, value: float) -> None:
        """Finish the split operation after directory selection."""
        from bigocrpdf.utils.temp_manager import mkstemp as _mkstemp
        from bigocrpdf.utils.temp_manager import remove_file as _rmfile

        try:
            folder = dialog.select_folder_finish(result)
            if not folder:
                return
            output_dir = folder.get_path()
            if not output_dir:
                return
        except GLib.Error:
            return

        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

        fd, tmp_path = _mkstemp(suffix=".pdf", prefix="bigocr_split_")
        os.close(fd)

        if not apply_changes_to_pdf(self._document, tmp_path):
            self._show_error(_("Failed to prepare document for splitting."))
            _rmfile(tmp_path)
            return

        from bigocrpdf.services.pdf_operations import split_by_pages, split_by_size

        original_stem = Path(self._document.path).stem
        try:
            if mode == "pages":
                result_split = split_by_pages(
                    tmp_path, output_dir, int(value), prefix=original_stem
                )
            else:
                result_split = split_by_size(tmp_path, output_dir, value, prefix=original_stem)

            self._show_info(
                _("Split into {} parts ({} pages)").format(
                    result_split.parts, result_split.total_pages
                )
            )
        except Exception as e:
            self._show_error(_("Split failed: {}").format(str(e)))
        finally:
            os.unlink(tmp_path)

    def _on_tools_reverse(self, _action, _param) -> None:
        """Reverse the page order."""
        if not self._document:
            return

        self._push_undo()
        active = self._document.get_active_pages()
        total = len(active)
        for i, page in enumerate(active):
            page.position = total - 1 - i

        self._document.mark_modified()
        self._page_grid.refresh()
        self._update_status_bar()
        logger.info("Reversed page order")
