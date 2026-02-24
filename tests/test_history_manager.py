"""Tests for history_manager module."""

import json
import os
import tempfile
from unittest.mock import patch

from bigocrpdf.utils.history_manager import HistoryEntry, HistoryManager


class TestHistoryEntry:
    def test_to_dict_roundtrip(self):
        entry = HistoryEntry(
            input_path="/tmp/in.pdf",
            output_path="/tmp/out.pdf",
            pages_processed=5,
            processing_time_seconds=12.3,
            language="deu",
        )
        d = entry.to_dict()
        restored = HistoryEntry.from_dict(d)
        assert restored.input_path == entry.input_path
        assert restored.output_path == entry.output_path
        assert restored.pages_processed == entry.pages_processed
        assert restored.language == entry.language
        assert restored.success is True

    def test_input_filename(self):
        entry = HistoryEntry(input_path="/path/to/document.pdf", output_path="/out.pdf")
        assert entry.input_filename == "document.pdf"

    def test_size_mb_properties(self):
        entry = HistoryEntry(
            input_path="/in.pdf",
            output_path="/out.pdf",
            input_size_bytes=1024 * 1024,
            output_size_bytes=2 * 1024 * 1024,
        )
        assert abs(entry.input_size_mb - 1.0) < 0.01
        assert abs(entry.output_size_mb - 2.0) < 0.01

    def test_from_dict_with_missing_fields(self):
        d = {"input_path": "/in.pdf", "output_path": "/out.pdf"}
        entry = HistoryEntry.from_dict(d)
        assert entry.pages_processed == 0
        assert entry.success is True


class TestHistoryManager:
    def test_add_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", path):
                hm = HistoryManager()
                entry = hm.add_entry("/in.pdf", "/out.pdf", pages_processed=3)
                assert isinstance(entry, HistoryEntry)
                assert hm.count == 1
        finally:
            os.unlink(path)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", path):
                hm1 = HistoryManager()
                hm1.add_entry("/in.pdf", "/out.pdf")
            with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", path):
                hm2 = HistoryManager()
                assert hm2.count == 1
        finally:
            os.unlink(path)

    def test_empty_history(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", path):
                hm = HistoryManager()
                assert hm.count == 0
        finally:
            os.unlink(path)
