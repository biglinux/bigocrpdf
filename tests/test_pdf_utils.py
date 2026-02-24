"""Tests for pdf_utils module (pure logic functions)."""

from bigocrpdf.utils.pdf_utils import is_image_file


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
