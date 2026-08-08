"""Tests for crash-safe page-thumbnail exports."""

from unittest.mock import patch

from bigocrpdf.services.pdf_operations import OperationResult
from bigocrpdf.ui.pdf_editor.page_thumbnail_export import PageThumbnailExporter


def test_pdf_page_export_delegates_to_atomic_pdf_operation() -> None:
    result = OperationResult(
        success=True,
        output_path="/output/page.pdf",
        pages_affected=1,
    )

    with patch(
        "bigocrpdf.services.pdf_operations.extract_pages",
        return_value=result,
    ) as extract_pages:
        PageThumbnailExporter._extract_page_pdf(
            "/input/source.pdf",
            3,
            "/output/page.pdf",
        )

    extract_pages.assert_called_once_with(
        "/input/source.pdf",
        "/output/page.pdf",
        [3],
    )
