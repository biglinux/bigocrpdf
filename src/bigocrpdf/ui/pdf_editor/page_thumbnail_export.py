"""Page export actions for PDF page thumbnails."""

from typing import TYPE_CHECKING, cast

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

if TYPE_CHECKING:
    from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail


class PageThumbnailExporter:
    _COMMON_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, thumbnail: "PageThumbnail") -> None:
        self._thumbnail = thumbnail

    def _save_page_as_image(self) -> None:
        """Save the current page as an image.

        Single-image pages: extracts original if common format, else renders.
        Multi-content pages: renders the full page at 300 DPI.
        """
        import os

        from bigocrpdf.utils.temp_manager import mkdtemp

        source = self._thumbnail.page_state.source_file or self._thumbnail.pdf_path
        page_num = self._thumbnail.page_state.page_number
        tmpdir = mkdtemp(prefix="bigocrpdf_save_")
        extracted, ext = self._extract_or_render_page_image(source, page_num, tmpdir)

        if not extracted:
            self._remove_temp_image_dir(tmpdir)
            return

        base = os.path.splitext(os.path.basename(source))[0]
        default_name = f"{base}_page{page_num}{ext}"
        self._open_save_image_dialog(default_name, extracted, tmpdir)

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
        import os
        import subprocess

        from bigocrpdf.utils.logger import logger

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
        import os
        import subprocess

        from bigocrpdf.utils.logger import logger

        prefix = os.path.join(tmpdir, "page")
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    "300",
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
            logger.error(f"pdftoppm failed: {e}")
            return None

        files = sorted(glob.glob(f"{prefix}-*"))
        return files[0] if files else None

    def _open_save_image_dialog(self, default_name: str, extracted: str, tmpdir: str) -> None:
        file_dialog = Gtk.FileDialog()
        file_dialog.set_initial_name(default_name)
        window = cast("Gtk.Window | None", self._thumbnail.get_root())

        def _on_save(_dialog, result):
            try:
                gfile = _dialog.save_finish(result)
            except GLib.Error:
                self._remove_temp_image_dir(tmpdir)
                return
            if gfile is None:
                self._remove_temp_image_dir(tmpdir)
                return
            try:
                import shutil

                from bigocrpdf.utils.logger import logger

                shutil.copy2(extracted, gfile.get_path())
                logger.info(f"Saved page image: {gfile.get_path()}")
            finally:
                self._remove_temp_image_dir(tmpdir)

        file_dialog.save(window, None, _on_save)

    @staticmethod
    def _remove_temp_image_dir(tmpdir: str) -> None:
        from bigocrpdf.utils.temp_manager import remove_dir

        remove_dir(tmpdir)

    def _save_page_as_pdf(self) -> None:
        """Save the current page as a single-page PDF (original, unmodified)."""
        import os

        source = self._thumbnail.page_state.source_file or self._thumbnail.pdf_path
        page_num = self._thumbnail.page_state.page_number
        base = os.path.splitext(os.path.basename(source))[0]
        default_name = f"{base}_page{page_num}.pdf"

        file_dialog = Gtk.FileDialog()
        file_dialog.set_initial_name(default_name)
        window = cast("Gtk.Window | None", self._thumbnail.get_root())

        def _on_save(_dialog, result):
            try:
                gfile = _dialog.save_finish(result)
            except GLib.Error:
                return
            if gfile is None:
                return
            self._extract_page_pdf(source, page_num, gfile.get_path())

        file_dialog.save(window, None, _on_save)

    @staticmethod
    def _extract_page_pdf(source: str, page_num: int, save_path: str) -> None:
        """Extract a single page from a PDF and save as a new PDF."""
        from bigocrpdf.services.pdf_operations import extract_pages
        from bigocrpdf.utils.logger import logger

        result = extract_pages(source, save_path, [page_num])
        if result.success:
            logger.info(f"Saved page {page_num} as PDF: {save_path}")
        else:
            logger.error("Failed to extract page as PDF: %s", result.message)
