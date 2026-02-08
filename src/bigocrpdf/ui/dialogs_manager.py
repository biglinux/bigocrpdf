"""
BigOcrPdf - Dialogs Manager Module

This module handles all dialog creation and management for the application.
"""

from typing import TYPE_CHECKING

from bigocrpdf.ui.file_save_mixin import FileSaveDialogMixin
from bigocrpdf.ui.pdf_options_callbacks_mixin import PDFOptionsCallbacksMixin
from bigocrpdf.ui.pdf_options_ui_mixin import PDFOptionsUICreationMixin
from bigocrpdf.ui.text_viewer_mixin import TextViewerDialogMixin

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow


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
