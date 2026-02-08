"""
BigOcrPdf - PDF Editor Module

This module provides a PDF editor interface for manipulating PDF pages
before OCR processing. Features include page rotation, deletion, reordering,
and selection for OCR.

Main Components:
- PDFEditorWindow: Main editor window (Adw.Window)
- PageGrid: FlowBox-based page grid display
- PageThumbnail: Individual page thumbnail widget
- PDFDocument: Document model with page states
"""

from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow
from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.page_operations import (
    delete_pages,
    reorder_pages,
    rotate_pages,
    set_ocr_selection,
)
from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import ThumbnailRenderer

__all__ = [
    "PDFEditorWindow",
    "PageGrid",
    "PageThumbnail",
    "PageState",
    "PDFDocument",
    "ThumbnailRenderer",
    "rotate_pages",
    "delete_pages",
    "reorder_pages",
    "set_ocr_selection",
]
