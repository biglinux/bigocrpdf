"""Tests for history_manager module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bigocrpdf.utils.history_manager import HistoryEntry, HistoryManager


class TestHistoryEntry:
    def test_to_dict_roundtrip(self):
        entry = HistoryEntry(
            input_path="/tmp/in.pdf",  # nosec B108
            output_path="/tmp/out.pdf",  # nosec B108
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

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            {"input_path": 42, "output_path": "/out.pdf"},
            {"input_path": "/in.pdf", "output_path": "/out.pdf", "success": "yes"},
            {"input_path": "/in.pdf", "output_path": "/out.pdf", "pages_processed": -1},
        ],
    )
    def test_from_dict_rejects_invalid_entries(self, payload):
        with pytest.raises((TypeError, ValueError)):
            HistoryEntry.from_dict(payload)

    def test_from_dict_ignores_unknown_future_fields(self):
        entry = HistoryEntry.from_dict(
            {"input_path": "/in.pdf", "output_path": "/out.pdf", "future_field": "ignored"}
        )
        assert entry.input_path == "/in.pdf"


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

    def test_save_replaces_symlink_without_overwriting_its_target(self, tmp_path: Path):
        victim = tmp_path / "victim.json"
        original = '{"entries": []}\n'
        victim.write_text(original, encoding="utf-8")
        history_path = tmp_path / "history.json"
        history_path.symlink_to(victim)

        with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", str(history_path)):
            manager = HistoryManager()
            manager.add_entry("/input.pdf", "/output.pdf")

        assert victim.read_text(encoding="utf-8") == original
        assert not history_path.is_symlink()

    def test_load_does_not_follow_history_symlink(self, tmp_path: Path):
        victim = tmp_path / "victim.json"
        victim.write_text(
            json.dumps({"entries": [{"input_path": "/secret.pdf", "output_path": "/out.pdf"}]}),
            encoding="utf-8",
        )
        history_path = tmp_path / "history.json"
        history_path.symlink_to(victim)

        with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", str(history_path)):
            manager = HistoryManager()

        assert manager.count == 0
        assert history_path.is_symlink()

    def test_load_keeps_valid_entries_and_skips_invalid_ones(self, tmp_path: Path):
        history_path = tmp_path / "history.json"
        history_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {"input_path": "/valid-one.pdf", "output_path": "/one.pdf"},
                        None,
                        {"input_path": 42, "output_path": "/invalid.pdf"},
                        {
                            "input_path": "/valid-two.pdf",
                            "output_path": "/two.pdf",
                            "future_field": "ignored",
                        },
                        {"input_path": "/missing-output.pdf"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", str(history_path)):
            manager = HistoryManager()

        assert manager.count == 2
        assert [entry.input_path for entry in manager._entries] == [
            "/valid-one.pdf",
            "/valid-two.pdf",
        ]

    @pytest.mark.parametrize("payload", [[], None, 42, {"entries": {}}, {"entries": "bad"}])
    def test_malformed_history_document_does_not_break_startup(self, tmp_path: Path, payload):
        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(payload), encoding="utf-8")

        with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", str(history_path)):
            manager = HistoryManager()

        assert manager.count == 0

    def test_invalid_json_history_does_not_break_startup(self, tmp_path: Path):
        history_path = tmp_path / "history.json"
        history_path.write_text('{"entries":', encoding="utf-8")

        with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", str(history_path)):
            manager = HistoryManager()

        assert manager.count == 0


def test_output_size_is_recorded_when_input_size_lookup_fails(tmp_path: Path):
    history_path = tmp_path / "history.json"

    with patch("bigocrpdf.utils.history_manager.HISTORY_FILE", str(history_path)):
        manager = HistoryManager()
        with patch(
            "bigocrpdf.utils.history_manager.os.path.getsize",
            side_effect=[OSError("input disappeared"), 2048],
        ):
            entry = manager.add_entry("/input.pdf", "/output.pdf")

    assert entry.input_size_bytes == 0
    assert entry.output_size_bytes == 2048
