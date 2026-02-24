"""Error handling and edge case tests for PDF operations."""

import os
import tempfile

import pikepdf
import pytest


def _create_test_pdf(path: str, num_pages: int = 3) -> str:
    """Create a minimal valid PDF for testing."""
    pdf = pikepdf.Pdf.new()
    for _ in range(num_pages):
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 612, 792],
            )
        )
        pdf.pages.append(page)
    pdf.save(path)
    return path


class TestPdfOperationsErrorHandling:
    """Error handling tests for pdf_operations module."""

    def test_split_nonexistent_file(self):
        from bigocrpdf.services.pdf_operations import split_by_pages

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(Exception):
                split_by_pages("/nonexistent/file.pdf", tmpdir, pages_per_split=1)

    def test_extract_pages_arg_order(self):
        from bigocrpdf.services.pdf_operations import extract_pages

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test.pdf")
            _create_test_pdf(pdf_path, num_pages=5)
            out_path = os.path.join(tmpdir, "out.pdf")
            result = extract_pages(pdf_path, out_path, [1, 3])
            assert os.path.exists(out_path)

    def test_rotate_preserves_page_count(self):
        from bigocrpdf.services.pdf_operations import rotate_pages

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test.pdf")
            _create_test_pdf(pdf_path, num_pages=3)
            out_path = os.path.join(tmpdir, "rotated.pdf")
            rotate_pages(pdf_path, out_path, [1, 2, 3], angle=90)
            with pikepdf.Pdf.open(out_path) as pdf:
                assert len(pdf.pages) == 3


class TestPdfExtractorErrorHandling:
    """Error handling tests for pdf_extractor module."""

    def test_has_native_text_empty_pdf(self):
        from bigocrpdf.services.rapidocr_service.pdf_extractor import has_native_text

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "empty.pdf")
            _create_test_pdf(pdf_path, num_pages=1)
            result = has_native_text(pdf_path)
            assert result is False  # Blank pages have no text

    def test_has_native_text_nonexistent_returns_false(self):
        from bigocrpdf.services.rapidocr_service.pdf_extractor import has_native_text

        # The function catches exceptions and returns False
        result = has_native_text("/definitely/not/here.pdf")
        assert result is False

    def test_extract_image_positions_empty_pdf(self):
        from bigocrpdf.services.rapidocr_service.pdf_extractor import (
            extract_image_positions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "empty.pdf")
            _create_test_pdf(pdf_path, num_pages=1)
            positions = extract_image_positions(pdf_path)
            assert isinstance(positions, dict)


class TestCorruptPdfHandling:
    """Tests for handling corrupt/invalid PDFs."""

    def test_open_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"This is not a PDF file at all")
            corrupt_path = f.name
        try:
            with pytest.raises(Exception):
                with pikepdf.Pdf.open(corrupt_path):
                    pass
        finally:
            os.unlink(corrupt_path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            empty_path = f.name
        try:
            with pytest.raises(Exception):
                with pikepdf.Pdf.open(empty_path):
                    pass
        finally:
            os.unlink(empty_path)
