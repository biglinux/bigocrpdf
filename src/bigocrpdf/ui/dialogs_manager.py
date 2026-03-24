"""
BigOcrPdf - Dialogs Manager Module

This module handles all dialog creation and management for the application.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Adw", "1")
from gi.repository import Adw

from bigocrpdf.ui.file_save_mixin import FileSaveDialogMixin
from bigocrpdf.ui.pdf_options_callbacks_mixin import PDFOptionsCallbacksMixin
from bigocrpdf.ui.pdf_options_ui_mixin import PDFOptionsUICreationMixin
from bigocrpdf.ui.text_viewer_mixin import TextViewerDialogMixin
from bigocrpdf.utils.i18n import _

if TYPE_CHECKING:
    from bigocrpdf.services.settings import OcrSettings
    from bigocrpdf.window import BigOcrPdfWindow

logger = logging.getLogger(__name__)


class DialogsManager(
    PDFOptionsUICreationMixin,
    PDFOptionsCallbacksMixin,
    TextViewerDialogMixin,
    FileSaveDialogMixin,
):
    """Manages all dialogs and modal windows for the application"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the dialogs manager

        Args:
            window: Reference to the main application window
        """
        self.window = window

    # ── Image merge dialog ──────────────────────────────────────────────

    def show_image_merge_dialog(
        self,
        image_files: list[str],
        settings: "OcrSettings",
        *,
        heading: str,
        body: str,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Show dialog asking whether to merge images into one PDF or keep separate.

        Args:
            image_files: Paths to the image files.
            settings: OcrSettings instance for tracking file origins.
            heading: Dialog heading text.
            body: Dialog body text.
            on_complete: Optional callback invoked after files are added.
        """
        from bigocrpdf.utils.pdf_utils import images_to_pdf

        dialog = Adw.AlertDialog()
        dialog.set_heading(heading)
        dialog.set_body(body)
        dialog.add_response("separate", _("Separate PDFs"))
        dialog.add_response("merge", _("Merge into One PDF"))
        dialog.set_response_appearance("merge", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("merge")

        def on_response(_dialog: Adw.AlertDialog, response: str) -> None:
            if response == "merge":
                try:
                    pdf_path = images_to_pdf(image_files)
                    settings.original_file_paths[pdf_path] = image_files[0]
                    added = settings.add_files([pdf_path])
                    if added > 0:
                        self.window.show_toast(
                            _("Merged {} images into one PDF").format(len(image_files))
                        )
                except (OSError, ValueError) as e:
                    logger.error("Failed to merge images: %s", e)
                    self.window.show_toast(_("Error merging images"))
            elif response == "separate":
                converted: list[str] = []
                for img_path in image_files:
                    try:
                        pdf_path = images_to_pdf([img_path])
                        converted.append(pdf_path)
                        settings.original_file_paths[pdf_path] = img_path
                    except (OSError, ValueError) as e:
                        logger.error("Failed to convert image to PDF: %s", e)
                if converted:
                    settings.add_files(converted)

            if on_complete:
                on_complete()

        dialog.connect("response", on_response)
        dialog.present(self.window)
