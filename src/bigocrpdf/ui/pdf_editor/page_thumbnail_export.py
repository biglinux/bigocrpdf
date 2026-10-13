"""Page export actions for PDF page thumbnails."""

import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow
    from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail

_PAGE_IMAGE_DPI = 300


class PageThumbnailExporter:
    _COMMON_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, thumbnail: "PageThumbnail") -> None:
        self._thumbnail = thumbnail

    def _save_page_as_image(self) -> None:
        """Save the current page as an image.

        Single-image pages: extracts original if common format, else renders.
        Multi-content pages: renders the full page at 300 DPI, or less when the
        page would exceed the OCR render megapixel budget.
        """
        from bigocrpdf.utils.temp_manager import mkdtemp

        source = self._thumbnail.page_state.source_file or self._thumbnail.pdf_path
        page_num = self._thumbnail.page_state.page_number
        tmpdir = mkdtemp(prefix="bigocrpdf_save_")

        def on_ready(image: tuple[str | None, str]) -> None:
            extracted, ext = image
            if not extracted:
                self._remove_temp_image_dir(tmpdir)
                self._report_error(_("Could not render page {page}").format(page=page_num))
                return
            base = os.path.splitext(os.path.basename(source))[0]
            self._open_save_image_dialog(f"{base}_page{page_num}{ext}", extracted, tmpdir)

        run_in_worker(
            lambda: self._extract_or_render_page_image(source, page_num, tmpdir), on_ready
        )

    def _extract_or_render_page_image(
        self,
        source: str,
        page_num: int,
        tmpdir: str,
    ) -> tuple[str | None, str]:
        if self._count_page_images(source, page_num) == 1:
            extracted, ext = self._extract_single_page_image(source, page_num, tmpdir)
            if extracted:
                return extracted, ext
        return self._render_page_image(source, page_num, tmpdir), ".png"

    @staticmethod
    def _count_page_images(source: str, page_num: int) -> int:
        import subprocess

        try:
            result = subprocess.run(
                ["pdfimages", "-list", "-f", str(page_num), "-l", str(page_num), source],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=10,
            )
        except Exception:
            return 0
        return sum(1 for line in result.stdout.splitlines()[2:] if line.split())

    def _extract_single_page_image(
        self,
        source: str,
        page_num: int,
        tmpdir: str,
    ) -> tuple[str | None, str]:
        import glob
        import subprocess

        prefix = os.path.join(tmpdir, "img")
        try:
            subprocess.run(
                [
                    "pdfimages",
                    "-all",
                    "-f",
                    str(page_num),
                    "-l",
                    str(page_num),
                    source,
                    prefix,
                ],
                check=True,
                timeout=30,
                capture_output=True,
            )
        except Exception as e:
            logger.error(f"pdfimages failed: {e}")
            return None, ".png"

        files = sorted(glob.glob(f"{prefix}-*"))
        if not files:
            return None, ".png"

        ext = os.path.splitext(files[0])[1].lower()
        if ext not in self._COMMON_IMAGE_EXTS:
            return None, ".png"
        return files[0], ext

    @staticmethod
    def _render_page_image(source: str, page_num: int, tmpdir: str) -> str | None:
        import glob
        import subprocess

        from bigocrpdf.services.ocr.config import DEFAULT_MAX_RENDER_MEGAPIXELS
        from bigocrpdf.services.ocr.resource_manager import (
            select_pdf_page_render_dpi,
        )

        prefix = os.path.join(tmpdir, "page")
        try:
            dpi = select_pdf_page_render_dpi(
                source, page_num, _PAGE_IMAGE_DPI, DEFAULT_MAX_RENDER_MEGAPIXELS
            )
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    str(dpi),
                    "-f",
                    str(page_num),
                    "-l",
                    str(page_num),
                    source,
                    prefix,
                ],
                check=True,
                timeout=30,
                capture_output=True,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            logger.error(f"pdftoppm failed: {e}")
            return None

        files = sorted(glob.glob(f"{prefix}-*"))
        return files[0] if files else None

    def _open_save_image_dialog(self, default_name: str, extracted: str, tmpdir: str) -> None:
        file_dialog = Gtk.FileDialog()
        file_dialog.set_initial_name(default_name)
        window = cast("Gtk.Window | None", self._thumbnail.get_root())

        def _on_save(_dialog, result):
            save_path = self._chosen_local_path(_dialog, result)
            if save_path is None:
                self._remove_temp_image_dir(tmpdir)
                return

            def publish() -> str | None:
                from bigocrpdf.utils.durable_writes import copy_file_atomically

                try:
                    copy_file_atomically(extracted, save_path, overwrite=True)
                except (OSError, ValueError) as error:
                    logger.error(f"Could not save page image {save_path}: {error}")
                    return str(error)
                finally:
                    self._remove_temp_image_dir(tmpdir)
                return None

            run_in_worker(publish, lambda error: self._report_saved(save_path, error))

        file_dialog.save(window, None, _on_save)

    def _chosen_local_path(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> str | None:
        """Return the picked destination, or None after a cancel or a reported error."""
        try:
            gfile = dialog.save_finish(result)
        except GLib.Error as error:
            if not error.matches(Gtk.dialog_error_quark(), Gtk.DialogError.DISMISSED):
                logger.error(f"Could not choose a destination: {error.message}")
                self._report_error(error.message)
            return None
        path = gfile.get_path() if gfile is not None else None
        if gfile is not None and path is None:
            self._report_error(_("Choose a folder on this computer to save the page."))
        return path

    def _report_saved(self, save_path: str, error: str | None) -> None:
        name = os.path.basename(save_path)
        if error is not None:
            self._report_error(_("Could not save {name}: {error}").format(name=name, error=error))
            return
        logger.info(f"Saved page export: {save_path}")
        window = self._editor_window()
        if window is not None:
            window._show_info(_("Saved: {}").format(name))

    def _report_error(self, message: str) -> None:
        window = self._editor_window()
        if window is not None:
            window._show_error(message)

    def _editor_window(self) -> "PDFEditorWindow | None":
        # Imported here: editor_window imports this module through the page grid.
        from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

        root = self._thumbnail.get_root()
        return root if isinstance(root, PDFEditorWindow) else None

    @staticmethod
    def _remove_temp_image_dir(tmpdir: str) -> None:
        from bigocrpdf.utils.temp_manager import remove_dir

        remove_dir(tmpdir)

    def _save_page_as_pdf(self) -> None:
        """Save the current page as a single-page PDF (original, unmodified)."""
        source = self._thumbnail.page_state.source_file or self._thumbnail.pdf_path
        page_num = self._thumbnail.page_state.page_number
        base = os.path.splitext(os.path.basename(source))[0]
        default_name = f"{base}_page{page_num}.pdf"

        file_dialog = Gtk.FileDialog()
        file_dialog.set_initial_name(default_name)
        window = cast("Gtk.Window | None", self._thumbnail.get_root())

        def _on_save(_dialog, result):
            save_path = self._chosen_local_path(_dialog, result)
            if save_path is None:
                return
            run_in_worker(
                lambda: self._extract_page_pdf(source, page_num, save_path),
                lambda error: self._report_saved(save_path, error),
            )

        file_dialog.save(window, None, _on_save)

    @staticmethod
    def _extract_page_pdf(source: str, page_num: int, save_path: str) -> str | None:
        """Extract a single page into a new PDF; return the failure message, if any."""
        from bigocrpdf.services.pdf_operations import extract_pages

        result = extract_pages(source, save_path, [page_num])
        if result.success:
            return None
        logger.error("Failed to extract page as PDF: %s", result.message)
        return result.message


def run_in_worker[T](work: Callable[[], T], on_done: Callable[[T], None]) -> None:
    """Run blocking work off the GTK main loop and deliver its result on it."""

    def run() -> None:
        value = work()

        def deliver() -> bool:
            on_done(value)
            return False

        GLib.idle_add(deliver)

    threading.Thread(target=run, daemon=True).start()
