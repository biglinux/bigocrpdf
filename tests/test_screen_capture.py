"""Tests for ScreenCaptureService static parsing methods."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock GTK and heavy deps before importing the module
_MOCK_MODULES = [
    "gi",
    "gi.repository",
    "gi.repository.Gtk",
    "gi.repository.Gio",
    "gi.repository.GLib",
    "gi.repository.Gdk",
    "bigocrpdf.services.rapidocr_service.preprocessor",
]
_saved = {}
for mod in _MOCK_MODULES:
    _saved[mod] = sys.modules.get(mod)
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Must mock gi.require_version before importing
mock_gi = sys.modules["gi"]
mock_gi.require_version = MagicMock()

from bigocrpdf.services.rapidocr_service.config import OCRResult  # noqa: E402
from bigocrpdf.services.screen_capture import ScreenCaptureService  # noqa: E402


class TestParseOcrResults:
    """Tests for ScreenCaptureService._parse_ocr_results."""

    def test_valid_json(self):
        data = {
            "boxes": [[[0, 0], [100, 0], [100, 20], [0, 20]]],
            "txts": ["Hello"],
            "scores": [0.95],
        }
        results = ScreenCaptureService._parse_ocr_results(json.dumps(data))
        assert len(results) == 1
        assert results[0].text == "Hello"
        assert results[0].confidence == 0.95

    def test_invalid_json(self):
        results = ScreenCaptureService._parse_ocr_results("not json{")
        assert results == []

    def test_error_in_response(self):
        data = {"error": "model not found"}
        results = ScreenCaptureService._parse_ocr_results(json.dumps(data))
        assert results == []

    def test_empty_boxes(self):
        data = {"boxes": [], "txts": [], "scores": []}
        results = ScreenCaptureService._parse_ocr_results(json.dumps(data))
        assert results == []

    def test_no_boxes_key(self):
        data = {"result": "none"}
        results = ScreenCaptureService._parse_ocr_results(json.dumps(data))
        assert results == []

    def test_multiple_results(self):
        data = {
            "boxes": [
                [[0, 0], [100, 0], [100, 20], [0, 20]],
                [[0, 30], [100, 30], [100, 50], [0, 50]],
            ],
            "txts": ["Line 1", "Line 2"],
            "scores": [0.9, 0.85],
        }
        results = ScreenCaptureService._parse_ocr_results(json.dumps(data))
        assert len(results) == 2
        assert results[1].text == "Line 2"


class TestFormatText:
    """Tests for ScreenCaptureService._format_text."""

    def test_empty_results(self):
        assert ScreenCaptureService._format_text([], 800) == ""

    def test_single_line(self):
        results = [OCRResult(text="Hello World", box=[[0, 10], [200, 10], [200, 30], [0, 30]])]
        text = ScreenCaptureService._format_text(results, 800)
        assert "Hello World" in text

    def test_reading_order(self):
        # Second box is above first box — should appear first in output
        results = [
            OCRResult(text="Line 2", box=[[0, 50], [200, 50], [200, 70], [0, 70]]),
            OCRResult(text="Line 1", box=[[0, 10], [200, 10], [200, 30], [0, 30]]),
        ]
        text = ScreenCaptureService._format_text(results, 800)
        pos1 = text.find("Line 1")
        pos2 = text.find("Line 2")
        assert pos1 < pos2

    def test_paragraph_break(self):
        results = [
            OCRResult(text="Para 1", box=[[0, 10], [200, 10], [200, 30], [0, 30]]),
            OCRResult(text="Para 2", box=[[0, 60], [200, 60], [200, 80], [0, 80]]),
        ]
        text = ScreenCaptureService._format_text(results, 800)
        assert "\n\n" in text
