"""Nautilus extension for BigOCR PDF and image context-menu actions."""

import gettext
import subprocess

from gi.repository import GObject, Nautilus

_ = gettext.translation("bigocrpdf", fallback=True).gettext

_OCR_IMAGE_MIMES = frozenset(
    (
        "image/apng",
        "image/avif",
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
        "image/x-portable-anymap",
        "image/x-portable-bitmap",
        "image/x-portable-graymap",
        "image/x-portable-pixmap",
    )
)
_PDF_IMAGE_MIMES = frozenset(
    (
        "image/avif",
        "image/bmp",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
    )
)
_PDF_MIME = "application/pdf"


def _paths(files):
    """Return local paths from selected Nautilus files."""
    paths = []
    for file in files:
        location = file.get_location()
        if location is not None:
            path = location.get_path()
            if path is not None:
                paths.append(path)
    return paths


def _run(command, paths):
    """Launch a command with selected local paths."""
    subprocess.Popen([*command, *paths])


class BigOcrPdfExtension(GObject.GObject, Nautilus.MenuProvider):
    """Context-menu entries for BigOCR PDF and image workflows."""

    def get_file_items(self, *args):
        # Nautilus 43+ passes (files,); older versions pass (window, files).
        files = args[-1] if args else []
        if not files:
            return []

        paths = _paths(files)
        if len(paths) != len(files):
            return []

        items = []
        mimes = {file.get_mime_type() for file in files}

        if mimes == {_PDF_MIME}:
            item_ocr = Nautilus.MenuItem(
                name="BigOcrPdf::ocr_pdf",
                label=_("Text recognition (OCR)"),
                icon="bigocrpdf",
            )
            item_ocr.connect("activate", self._on_ocr_pdf, paths)
            items.append(item_ocr)

            item_edit = Nautilus.MenuItem(
                name="BigOcrPdf::edit_pdf",
                label=_("Edit pages"),
                icon="bigocrpdf",
            )
            item_edit.connect("activate", self._on_edit_pdf, paths)
            items.append(item_edit)

        elif mimes <= _OCR_IMAGE_MIMES:
            if len(files) == 1:
                item_ocr = Nautilus.MenuItem(
                    name="BigOcrPdf::ocr_image",
                    label=_("Text recognition (OCR)"),
                    icon="bigocrimage",
                )
                item_ocr.connect("activate", self._on_ocr_image, paths)
                items.append(item_ocr)

            if mimes <= _PDF_IMAGE_MIMES:
                item_create = Nautilus.MenuItem(
                    name="BigOcrPdf::create_pdf",
                    label=_("Create PDF"),
                    icon="bigocrpdf",
                )
                item_create.connect("activate", self._on_create_pdf, paths)
                items.append(item_create)

        return items

    def _on_ocr_pdf(self, _menu, paths):
        _run(["bigocrpdf"], paths)

    def _on_edit_pdf(self, _menu, paths):
        _run(["bigocrpdf", "--edit"], paths)

    def _on_ocr_image(self, _menu, paths):
        _run(["bigocrimage"], paths)

    def _on_create_pdf(self, _menu, paths):
        _run(["bigocrpdf", "--edit"], paths)
