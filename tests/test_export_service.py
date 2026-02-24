"""Tests for export_service module."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bigocrpdf.services.export_service import save_text_file


class TestSaveTextFile:
    def test_basic_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "doc.pdf")
            result = save_text_file(output, "Hello world")
            expected = os.path.join(tmp, "doc.txt")
            assert result == expected
            with open(expected, encoding="utf-8") as f:
                assert f.read() == "Hello world"

    def test_saves_to_separate_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "doc.pdf")
            sep = os.path.join(tmp, "texts")
            result = save_text_file(output, "content", separate_folder=sep)
            assert result == os.path.join(sep, "doc.txt")
            assert os.path.isdir(sep)
            with open(result, encoding="utf-8") as f:
                assert f.read() == "content"

    def test_derives_name_from_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "my_scan_ocr.pdf")
            result = save_text_file(output, "text")
            assert result.endswith("my_scan_ocr.txt")

    def test_returns_none_on_write_error(self):
        result = save_text_file("/nonexistent/dir/file.pdf", "text")
        assert result is None

    def test_empty_text_creates_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "doc.pdf")
            result = save_text_file(output, "")
            assert result is not None
            with open(result, encoding="utf-8") as f:
                assert f.read() == ""

    def test_unicode_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "doc.pdf")
            text = "日本語テスト — αβγ — «português»"
            result = save_text_file(output, text)
            with open(result, encoding="utf-8") as f:
                assert f.read() == text

    def test_with_ocr_boxes_triggers_spread_detection(self):
        boxes = [SimpleNamespace(text="left", x=10, y=10, width=40, height=20, page_num=1)]
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "doc.pdf")
            with patch(
                "bigocrpdf.utils.spread_detector.detect_and_split_spreads",
                return_value=(boxes, {}),
            ) as mock_detect:
                result = save_text_file(output, "left", ocr_boxes=boxes)
                mock_detect.assert_called_once_with(boxes)
                assert result is not None

    def test_with_ocr_boxes_applies_split(self):
        boxes = [SimpleNamespace(text="left", x=10, y=10, width=40, height=20, page_num=1)]
        split_map = {1: (1, 2)}
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "doc.pdf")
            with (
                patch(
                    "bigocrpdf.utils.spread_detector.detect_and_split_spreads",
                    return_value=(boxes, split_map),
                ),
                patch(
                    "bigocrpdf.utils.spread_detector.split_text_by_spreads",
                    return_value="split text",
                ) as mock_split,
            ):
                result = save_text_file(output, "original", ocr_boxes=boxes)
                mock_split.assert_called_once()
                with open(result, encoding="utf-8") as f:
                    assert f.read() == "split text"
