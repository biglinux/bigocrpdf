"""Tests for FileQueueManager."""

from bigocrpdf.services.file_queue import FileQueueManager


class TestFileQueueManager:
    """Tests for FileQueueManager initialization and state."""

    def test_initial_state(self):
        fq = FileQueueManager()
        assert fq.selected_files == []
        assert fq.page_ranges == {}
        assert fq.processed_files == []
        assert fq.original_file_paths == {}
        assert fq.file_modifications == {}
        assert fq.pages_count == 0

    def test_add_files(self):
        fq = FileQueueManager()
        fq.selected_files.append("/path/to/file.pdf")
        assert len(fq.selected_files) == 1

    def test_add_page_range(self):
        fq = FileQueueManager()
        fq.page_ranges["file.pdf"] = (1, 5)
        assert fq.page_ranges["file.pdf"] == (1, 5)

    def test_page_range_none(self):
        fq = FileQueueManager()
        fq.page_ranges["file.pdf"] = None
        assert fq.page_ranges["file.pdf"] is None

    def test_track_original_path(self):
        fq = FileQueueManager()
        fq.original_file_paths["temp.pdf"] = "/original/path.pdf"
        assert fq.original_file_paths["temp.pdf"] == "/original/path.pdf"

    def test_file_modifications(self):
        fq = FileQueueManager()
        fq.file_modifications["file.pdf"] = [{"action": "rotate", "page": 1, "angle": 90}]
        assert len(fq.file_modifications["file.pdf"]) == 1

    def test_processed_files_tracking(self):
        fq = FileQueueManager()
        fq.processed_files.append("done.pdf")
        fq.processed_files.append("done2.pdf")
        assert len(fq.processed_files) == 2

    def test_pages_count_mutable(self):
        fq = FileQueueManager()
        fq.pages_count = 42
        assert fq.pages_count == 42
