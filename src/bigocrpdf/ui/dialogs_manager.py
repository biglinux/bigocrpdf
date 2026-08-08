"""
BigOcrPdf - Dialogs Manager Module

This module handles all dialog creation and management for the application.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Adw", "1")
from gi.repository import Adw

from bigocrpdf.ui.file_save_controller import FileSaveController
from bigocrpdf.ui.pdf_options_controller import PDFOptionsController
from bigocrpdf.ui.text_viewer_controller import TextViewerController
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import images_to_pdf

if TYPE_CHECKING:
    from bigocrpdf.services.settings import OcrSettings


class DialogsManager:
    """Manages all dialogs and modal windows for the application"""

    def __init__(
        self,
        parent,
        settings: "OcrSettings",
        show_toast: Callable[[str], None],
    ) -> None:
        """Initialize the dialogs manager

        Args:
            parent: Parent window for modal dialogs.
        """
        self._parent = parent
        self._settings = settings
        self._show_toast = show_toast
        self._file_save = FileSaveController(parent, show_toast)
        self._text_viewer = TextViewerController(
            parent,
            settings,
            show_toast,
            self._file_save,
        )
        self._pdf_options = PDFOptionsController(parent, settings)

    def show_pdf_options_dialog(self, callback: Callable) -> None:
        """Present PDF output settings."""
        self._pdf_options.show_pdf_options_dialog(callback)

    def show_extracted_text(self, file_path: str) -> None:
        """Present extracted text for a processed file."""
        self._text_viewer.show_extracted_text(file_path)

    # ── Image merge dialog ──────────────────────────────────────────────

    def show_image_merge_dialog(
        self,
        image_files: list[str],
        *,
        heading: str,
        body: str,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Show dialog asking whether to merge images into one PDF or keep separate.

        Args:
            image_files: Paths to the image files.
            heading: Dialog heading text.
            body: Dialog body text.
            on_complete: Optional callback invoked after files are added.
        """
        if not image_files:
            logger.warning("Image merge dialog opened without images")
            if on_complete is not None:
                on_complete()
            return

        dialog = Adw.AlertDialog()
        dialog.set_heading(heading)
        dialog.set_body(body)
        dialog.add_response("separate", _("Separate PDFs"))
        dialog.add_response("merge", _("Merge into One PDF"))
        dialog.set_response_appearance("merge", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("merge")

        def on_response(_dialog: Adw.AlertDialog, response: str) -> None:
            if response == "merge":
                if self._convert_and_queue_images(image_files, image_files[0]):
                    self._show_toast(
                        ngettext(
                            "Merged {count} image into one PDF",
                            "Merged {count} images into one PDF",
                            len(image_files),
                        ).format(count=len(image_files))
                    )
                else:
                    self._show_toast(_("Error merging images"))
            elif response == "separate":
                for image_path in image_files:
                    self._convert_and_queue_images([image_path], image_path)

            if on_complete is not None:
                on_complete()

        dialog.connect("response", on_response)
        dialog.present(self._parent)

    def _convert_and_queue_images(self, image_files: list[str], original_path: str) -> bool:
        try:
            pdf_path = images_to_pdf(image_files)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Failed to convert images to PDF: %s", error)
            return False
        return self._settings._add_generated_file(pdf_path, original_path)
