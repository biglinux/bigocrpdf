"""Public, lazily loaded PDF editor API."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow
    from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
    from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
    from bigocrpdf.ui.pdf_editor.page_operations import (
        delete_pages,
        rotate_pages,
        set_ocr_selection,
    )
    from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail
    from bigocrpdf.ui.pdf_editor.thumbnail_renderer import ThumbnailRenderer

_EXPORT_MODULES = {
    "PDFEditorWindow": "editor_window",
    "PageGrid": "page_grid",
    "PageState": "page_model",
    "PDFDocument": "page_model",
    "delete_pages": "page_operations",
    "rotate_pages": "page_operations",
    "set_ocr_selection": "page_operations",
    "PageThumbnail": "page_thumbnail",
    "ThumbnailRenderer": "thumbnail_renderer",
}

__all__ = [
    "PDFEditorWindow",
    "PageGrid",
    "PageThumbnail",
    "PageState",
    "PDFDocument",
    "ThumbnailRenderer",
    "rotate_pages",
    "delete_pages",
    "set_ocr_selection",
]


def __getattr__(name: str) -> Any:
    """Load editor components only when their public export is requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
