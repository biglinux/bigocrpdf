"""Tests for thread safety in processor and engine modules."""

import threading
from unittest.mock import MagicMock, patch


class TestProcessorThreadSafety:
    """Thread safety tests for OcrProcessor state management."""

    def _make_processor(self):
        """Create a processor with mocked settings."""
        with patch("bigocrpdf.services.processor.get_checkpoint_manager"), \
             patch("bigocrpdf.services.processor.get_history_manager"):
            from bigocrpdf.services.processor import OcrProcessor

            settings = MagicMock()
            settings.selected_files = []
            settings.processed_files = []
            settings.display_name.return_value = "test.pdf"
            processor = OcrProcessor(settings)
            return processor

    def test_state_lock_exists(self):
        proc = self._make_processor()
        assert hasattr(proc, "_state_lock")
        assert isinstance(proc._state_lock, type(threading.Lock()))

    def test_concurrent_progress_updates(self):
        """Simulate concurrent progress updates from multiple threads."""
        proc = self._make_processor()
        proc._is_processing = True
        proc._total_files_at_start = 10
        errors = []

        def update_progress(thread_id: int) -> None:
            try:
                for i in range(100):
                    with proc._state_lock:
                        proc._file_progress = (i + 1) / 100.0
                        proc._current_status = f"Thread {thread_id} step {i}"
                        proc._current_filename = f"file_{thread_id}.pdf"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_progress, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Final state should be consistent (from whichever thread finished last)
        assert 0.0 <= proc._file_progress <= 1.0

    def test_get_current_file_info_during_processing(self):
        """get_current_file_info should return consistent snapshot."""
        proc = self._make_processor()
        proc._is_processing = True
        proc._total_files_at_start = 5
        proc._current_filename = "test.pdf"
        proc._current_status = "OCR in progress"
        proc._file_progress = 0.5

        info = proc.get_current_file_info()
        assert info["filename"] == "test.pdf"
        assert info["status_message"] == "OCR in progress"
        assert info["file_progress"] == 0.5

    def test_get_current_file_info_not_processing(self):
        proc = self._make_processor()
        proc._is_processing = False
        assert proc.get_current_file_info() == {}

    def test_setup_processing_resets_state(self):
        """_setup_processing should reset all progress fields."""
        proc = self._make_processor()
        proc._file_progress = 0.75
        proc._current_status = "old status"
        proc._current_filename = "old.pdf"

        proc.settings.selected_files = ["a.pdf", "b.pdf"]
        proc.settings.ocr_language = "latin"
        proc.settings.dpi = 300
        proc.settings.destination_folder = "/tmp"
        proc.settings.save_in_same_folder = False

        with patch("bigocrpdf.services.processor.get_checkpoint_manager"):
            proc._setup_processing()

        assert proc._file_progress == 0.0
        assert proc._current_status == ""
        assert proc._current_filename == ""


class TestEngineCacheThreadSafety:
    """Thread safety tests for _OCRModelCache."""

    def test_lock_exists(self):
        from bigocrpdf.services.rapidocr_service.engine import _OCRModelCache

        assert hasattr(_OCRModelCache, "_lock")
        assert isinstance(_OCRModelCache._lock, type(threading.Lock()))

    def test_concurrent_cache_access(self):
        """Multiple threads requesting engine should not crash."""
        from bigocrpdf.services.rapidocr_service.config import OCRConfig
        from bigocrpdf.services.rapidocr_service.engine import _OCRModelCache

        # Reset cache state
        _OCRModelCache._instance = None
        _OCRModelCache._config_hash = 0

        config = OCRConfig()
        errors = []

        with patch(
            "bigocrpdf.services.rapidocr_service.engine.ProfessionalPDFOCR"
        ) as mock_cls:
            mock_engine = MagicMock()
            mock_engine.config = config
            mock_cls.return_value = mock_engine

            def get_engine() -> None:
                try:
                    _OCRModelCache.get_cached_engine(config)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_engine) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        # Should have created engine exactly once (all threads see same config)
        assert mock_cls.call_count == 1

        # Cleanup
        _OCRModelCache._instance = None
        _OCRModelCache._config_hash = 0
