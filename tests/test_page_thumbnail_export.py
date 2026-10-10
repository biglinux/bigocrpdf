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


def test_page_image_render_stays_within_megapixel_budget(tmp_path) -> None:
    import pikepdf

    # 40 x 40 inches: 144 megapixels at 300 DPI.
    source = tmp_path / "poster.pdf"
    with pikepdf.new() as pdf:
        pdf.add_blank_page(page_size=(2880, 2880))
        pdf.save(source)

    with patch("subprocess.run") as run:
        PageThumbnailExporter._render_page_image(str(source), 1, str(tmp_path))

    argv = run.call_args.args[0]
    dpi = int(argv[argv.index("-r") + 1])
    assert (40 * dpi) ** 2 <= 45_000_000
