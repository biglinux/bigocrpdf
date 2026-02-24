"""Tests for checkpoint_manager module."""

import tempfile
from pathlib import Path

from bigocrpdf.utils.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    def _make_manager(self, tmp_dir):
        return CheckpointManager(checkpoint_dir=Path(tmp_dir))

    def test_no_incomplete_session_initially(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            assert cm.has_incomplete_session() is False

    def test_start_session(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            sid = cm.start_session(["file1.pdf", "file2.pdf"])
            assert isinstance(sid, str)
            assert len(sid) > 0

    def test_incomplete_session_after_start(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.start_session(["file1.pdf"])
            assert cm.has_incomplete_session() is True

    def test_complete_session_clears_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.start_session(["file1.pdf"])
            cm.mark_file_completed("file1.pdf", "output1.pdf")
            cm.complete_session()
            assert cm.has_incomplete_session() is False

    def test_mark_file_completed(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.start_session(["a.pdf", "b.pdf"])
            cm.mark_file_completed("a.pdf", "a_out.pdf")
            info = cm.get_incomplete_session_info()
            assert info is not None
            assert info.get("completed_files", 0) == 1

    def test_mark_file_failed(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.start_session(["a.pdf", "b.pdf"])
            cm.mark_file_failed("a.pdf", "disk full")
            info = cm.get_incomplete_session_info()
            assert info is not None
            assert info.get("failed_files", 0) == 1

    def test_resume_session(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.start_session(["a.pdf", "b.pdf"])
            cm.mark_file_completed("a.pdf", "a_out.pdf")
            result = cm.resume_session()
            assert result is not None
            remaining, _ = result
            assert "b.pdf" in remaining

    def test_discard_session(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            cm.start_session(["a.pdf"])
            assert cm.discard_session() is True
            assert cm.has_incomplete_session() is False

    def test_discard_no_session(self):
        with tempfile.TemporaryDirectory() as d:
            cm = self._make_manager(d)
            assert cm.discard_session() is False

    def test_session_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            cm1 = self._make_manager(d)
            cm1.start_session(["file.pdf"])
            cm2 = self._make_manager(d)
            assert cm2.has_incomplete_session() is True
