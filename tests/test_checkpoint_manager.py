"""Tests for checkpoint_manager module."""

import json
import tempfile
from pathlib import Path

import pytest

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

    def test_discard_active_session_without_checkpoint_file(self, tmp_path: Path):
        manager = self._make_manager(tmp_path)
        manager.start_session(["file.pdf"])
        (tmp_path / "checkpoint.json").unlink()

        assert manager.discard_session() is True
        assert manager._current_checkpoint is None

    def test_session_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            cm1 = self._make_manager(d)
            cm1.start_session(["file.pdf"])
            cm2 = self._make_manager(d)
            assert cm2.has_incomplete_session() is True

    def test_file_modifications_persist_across_instances(self, tmp_path: Path):
        first = self._make_manager(tmp_path)
        modifications = {"pages": [{"page_number": 1, "rotation": 90}]}
        first.start_session(["file.pdf"], file_modifications={"file.pdf": modifications})

        second = self._make_manager(tmp_path)
        assert second.resume_session() is not None
        restored = second.get_file_modifications()
        assert restored == {"file.pdf": modifications}
        restored["file.pdf"]["pages"][0]["rotation"] = 180
        assert second.get_file_modifications() == {"file.pdf": modifications}

    def test_resumed_settings_are_a_defensive_copy(self, tmp_path: Path):
        first = self._make_manager(tmp_path)
        settings = {"ocr": {"language": "eng"}}
        first.start_session(["file.pdf"], settings=settings)

        second = self._make_manager(tmp_path)
        result = second.resume_session()
        assert result is not None
        _pending, restored_settings = result
        restored_settings["ocr"]["language"] = "por"

        assert second._current_checkpoint is not None
        assert second._current_checkpoint.settings_snapshot == settings

    def test_checkpoint_temp_symlink_cannot_overwrite_target(self, tmp_path: Path):
        victim = tmp_path / "victim.txt"
        victim.write_text("KEEP", encoding="utf-8")
        predictable_temp = tmp_path / "checkpoint.tmp"
        predictable_temp.symlink_to(victim)

        manager = self._make_manager(tmp_path)
        manager.start_session(["file.pdf"])

        assert victim.read_text(encoding="utf-8") == "KEEP"
        assert predictable_temp.is_symlink()

    def test_completed_and_failed_transitions_are_coherent_and_idempotent(self, tmp_path: Path):
        manager = self._make_manager(tmp_path)
        manager.start_session(["file.pdf", "other.pdf"])
        manager.mark_file_failed("file.pdf", "first failure")
        manager.mark_file_completed("file.pdf", "output.pdf")

        checkpoint = manager._current_checkpoint
        assert checkpoint is not None
        assert checkpoint.files_completed == ["file.pdf"]
        assert checkpoint.files_failed == []
        assert checkpoint.output_files == {"file.pdf": "output.pdf"}
        assert checkpoint.file_errors == {}

        checkpoint_path = tmp_path / "checkpoint.json"
        completed_state = checkpoint_path.read_text(encoding="utf-8")
        manager.mark_file_completed("file.pdf", "output.pdf")
        assert checkpoint_path.read_text(encoding="utf-8") == completed_state

        manager.mark_file_failed("file.pdf", "retry failed")
        assert checkpoint.files_completed == []
        assert checkpoint.files_failed == ["file.pdf"]
        assert checkpoint.output_files == {}
        assert checkpoint.file_errors == {"file.pdf": "retry failed"}

        failed_state = checkpoint_path.read_text(encoding="utf-8")
        manager.mark_file_failed("file.pdf", "retry failed")
        assert checkpoint_path.read_text(encoding="utf-8") == failed_state

    def test_file_error_persists_and_old_checkpoint_without_errors_migrates(self, tmp_path: Path):
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "session_id": "old-session",
                    "files_to_process": ["failed.pdf", "pending.pdf"],
                    "files_completed": [],
                    "files_failed": ["failed.pdf"],
                    "output_files": {},
                    "settings_snapshot": {},
                    "start_time": 1,
                    "last_update": 2,
                    "is_complete": False,
                }
            ),
            encoding="utf-8",
        )
        manager = self._make_manager(tmp_path)

        assert manager.resume_session() == (["pending.pdf"], {})
        assert manager._current_checkpoint is not None
        assert manager._current_checkpoint.file_errors == {}

        manager.mark_file_failed("pending.pdf", "disk full")
        reloaded = self._make_manager(tmp_path)
        info = reloaded.get_incomplete_session_info()
        assert info is None
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert data["file_errors"] == {"pending.pdf": "disk full"}

    def test_complete_session_refuses_to_hide_pending_files(self, tmp_path: Path):
        manager = self._make_manager(tmp_path)
        manager.start_session(["done.pdf", "pending.pdf"])
        manager.mark_file_completed("done.pdf", "done-output.pdf")

        manager.complete_session()

        assert manager._current_checkpoint is not None
        assert manager._current_checkpoint.is_complete is False
        assert manager.has_incomplete_session() is True

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"session_id": 42, "files_to_process": ["file.pdf"]},
            {"session_id": "session", "files_to_process": "file.pdf"},
            {
                "session_id": "session",
                "files_to_process": ["file.pdf"],
                "settings_snapshot": [],
            },
        ],
    )
    def test_invalid_checkpoint_schema_is_never_restored(self, tmp_path: Path, payload):
        (tmp_path / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
        manager = self._make_manager(tmp_path)

        assert manager.resume_session() is None
        assert manager._current_checkpoint is None

    def test_invalid_checkpoint_json_is_never_restored(self, tmp_path: Path):
        (tmp_path / "checkpoint.json").write_text('{"session_id":', encoding="utf-8")
        manager = self._make_manager(tmp_path)

        assert manager.resume_session() is None
        assert manager._current_checkpoint is None

    def test_legacy_conflicting_state_is_normalized_using_output_evidence(self, tmp_path: Path):
        (tmp_path / "checkpoint.json").write_text(
            json.dumps(
                {
                    "session_id": "session",
                    "files_to_process": ["file.pdf", "pending.pdf"],
                    "files_completed": ["file.pdf"],
                    "files_failed": ["file.pdf"],
                    "output_files": {"file.pdf": "output.pdf"},
                    "settings_snapshot": {},
                    "is_complete": False,
                }
            ),
            encoding="utf-8",
        )
        manager = self._make_manager(tmp_path)

        assert manager.resume_session() == (["pending.pdf"], {})
        assert manager._current_checkpoint is not None
        assert manager._current_checkpoint.files_completed == ["file.pdf"]
        assert manager._current_checkpoint.files_failed == []

    def test_checkpoint_symlink_is_not_restored(self, tmp_path: Path):
        victim = tmp_path / "victim.json"
        victim.write_text(
            json.dumps(
                {
                    "session_id": "external-session",
                    "files_to_process": ["file.pdf"],
                    "settings_snapshot": {},
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "checkpoint.json").symlink_to(victim)

        manager = self._make_manager(tmp_path)

        assert manager.resume_session() is None
        assert manager._current_checkpoint is None

    def test_checkpoint_target_symlink_is_replaced_without_touching_target(self, tmp_path: Path):
        victim = tmp_path / "victim.json"
        original = '{"owner": "victim"}\n'
        victim.write_text(original, encoding="utf-8")
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.symlink_to(victim)

        manager = self._make_manager(tmp_path)
        manager.start_session(["file.pdf"])

        assert victim.read_text(encoding="utf-8") == original
        assert not checkpoint_path.is_symlink()
        assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["files_to_process"] == [
            "file.pdf"
        ]
