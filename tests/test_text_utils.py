"""Tests for text_utils module."""

import os
import tempfile
from types import SimpleNamespace

from bigocrpdf.utils.text_utils import group_ocr_text_by_page, read_text_from_sidecar


class TestReadTextFromSidecar:
    def test_empty_path_returns_none(self):
        assert read_text_from_sidecar("") is None

    def test_nonexistent_file_returns_none(self):
        assert read_text_from_sidecar("/nonexistent/file.txt") is None

    def test_reads_utf8_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world")
            path = f.name
        try:
            assert read_text_from_sidecar(path) == "Hello world"
        finally:
            os.unlink(path)

    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            assert read_text_from_sidecar(path) is None
        finally:
            os.unlink(path)

    def test_reads_latin1_fallback(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"\xe9\xe0\xfc")  # Latin-1 chars that are not valid UTF-8
            path = f.name
        try:
            result = read_text_from_sidecar(path)
            assert result is not None
            assert "é" in result
        finally:
            os.unlink(path)


class TestGroupOcrTextByPage:
    def test_empty_boxes(self):
        assert group_ocr_text_by_page([], 3) == ["", "", ""]

    def test_single_page(self):
        boxes = [SimpleNamespace(text="hello", page_num=1)]
        assert group_ocr_text_by_page(boxes, 1) == ["hello"]

    def test_multiple_pages(self):
        boxes = [
            SimpleNamespace(text="page1-a", page_num=1),
            SimpleNamespace(text="page1-b", page_num=1),
            SimpleNamespace(text="page2", page_num=2),
        ]
        result = group_ocr_text_by_page(boxes, 2)
        assert result[0] == "page1-a\npage1-b"
        assert result[1] == "page2"

    def test_box_without_page_num_defaults_to_1(self):
        boxes = [SimpleNamespace(text="text")]
        result = group_ocr_text_by_page(boxes, 1)
        assert result[0] == "text"

    def test_box_with_invalid_page_num_ignored(self):
        boxes = [SimpleNamespace(text="orphan", page_num=99)]
        result = group_ocr_text_by_page(boxes, 2)
        assert result == ["", ""]
