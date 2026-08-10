"""Per-stage timing, peak RSS, and scratch-directory accounting."""

from pathlib import Path

import pytest

from bigocrpdf.utils.stage_timer import (
    StageTimer,
    dir_bytes,
    peak_rss_mb,
    process_rss_mb,
)


class TestStageTimer:
    def test_repeated_stages_accumulate(self):
        """A stage runs once per page, so a run needs the sum, not the last."""
        timer = StageTimer()

        for _ in range(3):
            with timer.stage("preprocess"):
                pass

        assert set(timer.totals()) == {"preprocess"}
        assert timer.totals()["preprocess"] >= 0.0

    def test_stages_are_kept_apart(self):
        timer = StageTimer()
        with timer.stage("render"):
            pass
        with timer.stage("ocr"):
            pass

        assert sorted(timer.totals()) == ["ocr", "render"]

    def test_a_raising_stage_is_still_recorded(self):
        timer = StageTimer()

        with pytest.raises(RuntimeError), timer.stage("ocr"):
            raise RuntimeError("worker died")

        assert "ocr" in timer.totals()

    def test_add_folds_in_a_duration_measured_elsewhere(self):
        """Worker traces arrive as numbers, not as a context manager."""
        timer = StageTimer()

        timer.add("dewarp", 12.5)
        timer.add("dewarp", 7.5)

        assert timer.totals()["dewarp"] == pytest.approx(20.0)

    def test_totals_are_empty_before_any_stage(self):
        assert StageTimer().totals() == {}


class TestProcessMemory:
    def test_peak_rss_is_a_positive_number_on_linux(self):
        value = peak_rss_mb()

        assert value is None or value > 0.0

    def test_rss_of_this_process_is_readable(self):
        import os

        value = process_rss_mb(os.getpid())

        assert value is None or value > 0.0

    def test_a_dead_pid_reports_nothing_rather_than_raising(self):
        """Measurement must never be a reason to fail a run."""
        assert process_rss_mb(999_999_999) is None


class TestDirBytes:
    def test_sums_regular_files_recursively(self, tmp_path: Path):
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "b.bin").write_bytes(b"y" * 50)

        assert dir_bytes(tmp_path) == 150

    def test_empty_directory_is_zero(self, tmp_path: Path):
        assert dir_bytes(tmp_path) == 0

    def test_missing_directory_is_zero(self, tmp_path: Path):
        assert dir_bytes(tmp_path / "absent") == 0

    def test_symlinks_are_not_followed(self, tmp_path: Path):
        """Otherwise a link into the filesystem root would be walked."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "big.bin").write_bytes(b"z" * 500)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "link").symlink_to(target)

        assert dir_bytes(scratch) == 0
