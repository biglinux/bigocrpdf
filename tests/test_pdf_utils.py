"""Tests for pdf_utils module."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from bigocrpdf.utils.pdf_utils import get_pdf_info, images_to_pdf, is_image_file


def test_pdf_info_uses_one_snapshot_for_pages_and_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"pdf")
    result = SimpleNamespace(stdout="Pages: 3\nTitle: Report\nAuthor: Ada\n")

    with patch("bigocrpdf.utils.pdf_utils.subprocess.run", return_value=result) as run:
        info = get_pdf_info(str(pdf_path))

    run.assert_called_once()
    assert info["pages"] == 3
    assert info["title"] == "Report"
    assert info["author"] == "Ada"


def test_pdf_info_keeps_metadata_when_page_count_is_invalid(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"pdf")
    result = SimpleNamespace(stdout="Pages: unknown\nTitle: Report\nAuthor: Ada\n")

    with patch("bigocrpdf.utils.pdf_utils.subprocess.run", return_value=result):
        info = get_pdf_info(str(pdf_path))

    assert info["pages"] == 0
    assert info["title"] == "Report"
    assert info["author"] == "Ada"


def test_images_to_pdf_removes_owned_output_when_save_fails(tmp_path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    generated_path = ""

    def create_temp(*, prefix: str, suffix: str) -> tuple[int, str]:
        nonlocal generated_path
        fd, generated_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=tmp_path)
        return fd, generated_path

    with (
        patch("bigocrpdf.utils.temp_manager.mkstemp", side_effect=create_temp),
        patch.object(Image.Image, "save", side_effect=OSError("disk write failed")),
        pytest.raises(RuntimeError, match="Failed to create PDF"),
    ):
        images_to_pdf([str(image_path)])

    assert generated_path
    assert not os.path.exists(generated_path)


class TestIsImageFile:
    def test_png_is_image(self):
        assert is_image_file("photo.png") is True

    def test_jpg_is_image(self):
        assert is_image_file("photo.jpg") is True

    def test_jpeg_is_image(self):
        assert is_image_file("photo.jpeg") is True

    def test_tiff_is_image(self):
        assert is_image_file("scan.tiff") is True

    def test_bmp_is_image(self):
        assert is_image_file("image.bmp") is True

    def test_webp_is_image(self):
        assert is_image_file("image.webp") is True

    def test_pdf_is_not_image(self):
        assert is_image_file("document.pdf") is False

    def test_txt_is_not_image(self):
        assert is_image_file("notes.txt") is False

    def test_case_insensitive(self):
        assert is_image_file("PHOTO.PNG") is True
        assert is_image_file("photo.JPG") is True

    def test_empty_string(self):
        assert is_image_file("") is False

    def test_no_extension(self):
        assert is_image_file("myfile") is False

    def test_path_with_directories(self):
        assert is_image_file("/home/user/photo.png") is True
