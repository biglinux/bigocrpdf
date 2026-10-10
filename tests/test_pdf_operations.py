"""Tests for pdf_operations module."""

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from bigocrpdf.services import pdf_operations
from bigocrpdf.services.pdf_operations import (
    PDFInfo,
    get_pdf_info,
    insert_pages,
    rotate_pages,
)


def _create_test_pdf(path: str, num_pages: int = 3) -> str:
    """Create a simple test PDF with the given number of pages."""
    pdf = pikepdf.Pdf.new()
    for i in range(num_pages):
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 612, 792],
                Resources=pikepdf.Dictionary(),
                Contents=pdf.make_stream(f"BT /F1 12 Tf 100 700 Td (Page {i + 1}) Tj ET".encode()),
            )
        )
        pdf.pages.append(page)
    pdf.save(path)
    return path


class TestGetPdfInfo:
    def test_basic_info(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            _create_test_pdf(path, 5)
            info = get_pdf_info(path)
            assert isinstance(info, PDFInfo)
            assert info.page_count == 5
            assert info.file_size_bytes > 0
            assert info.path == path
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            get_pdf_info("/nonexistent/file.pdf")


def test_atomic_pdf_save_preserves_existing_destination_on_failure(tmp_path):
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing")
    pdf = MagicMock()

    def fail_after_partial_write(path, **_options):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated save failure")

    pdf.save.side_effect = fail_after_partial_write

    with pytest.raises(OSError, match="simulated save failure"):
        pdf_operations._save_pdf_atomically(pdf, destination)

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_pdf_save_publishes_the_pdf_and_nothing_else(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output.pdf"
    _create_test_pdf(str(destination), 1)
    replacement = pikepdf.Pdf.new()
    replacement.add_blank_page()

    try:
        pdf_operations._save_pdf_atomically(replacement, destination)
    finally:
        replacement.close()

    assert [entry.name for entry in tmp_path.iterdir()] == ["output.pdf"]


def test_atomic_new_pdf_save_closes_document_on_failure(tmp_path):
    pdf = MagicMock()

    with (
        patch.object(pdf_operations, "_save_pdf_atomically", side_effect=OSError("failure")),
        pytest.raises(OSError, match="failure"),
    ):
        pdf_operations._save_and_close_pdf_atomically(pdf, tmp_path / "output.pdf")

    pdf.close.assert_called_once_with()


def test_atomic_pdf_save_preserves_existing_destination_mode(tmp_path: Path) -> None:
    destination = tmp_path / "existing.pdf"
    _create_test_pdf(str(destination), 1)
    destination.chmod(0o640)
    replacement = pikepdf.Pdf.new()
    replacement.add_blank_page()

    try:
        pdf_operations._save_pdf_atomically(replacement, destination)
    finally:
        replacement.close()

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_rotate_duplicate_page_only_once(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "rotated.pdf"
    _create_test_pdf(str(source), 1)

    result = rotate_pages(source, output, [1, 1], 90)

    assert result.success is True
    assert result.pages_affected == 1
    with pikepdf.open(output) as pdf:
        assert int(pdf.pages[0].get("/Rotate", 0)) == 90


def test_insert_empty_source_page_list_inserts_nothing(tmp_path: Path) -> None:
    target = tmp_path / "target.pdf"
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _create_test_pdf(str(target), 1)
    _create_test_pdf(str(source), 2)

    result = insert_pages(target, source, output, source_pages=[])

    assert result.success is False
    assert result.pages_affected == 0
    assert not output.exists()


def test_atomic_pdf_save_cleans_staged_file_when_mode_copy_fails(tmp_path: Path) -> None:
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing")
    destination.chmod(0o640)
    pdf = MagicMock()

    with (
        patch.object(pdf_operations.os, "fchmod", side_effect=OSError("chmod failed")),
        pytest.raises(OSError, match="chmod failed"),
    ):
        pdf_operations._save_pdf_atomically(pdf, destination)

    pdf.save.assert_not_called()
    assert list(tmp_path.iterdir()) == [destination]
