"""
RapidOCR Engine — cached model lifecycle management.

Keeps the expensive OCR engine loaded between processing runs so that
consecutive files reuse the same models. Configuration changes that do
NOT affect model loading (preprocessing flags, output options, …) are
propagated automatically via dataclass introspection.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

from .backend import ProfessionalPDFOCR
from .config import OCRConfig, ProcessingStats

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

# Fields that require engine re-creation when changed.
# Everything else is safe to update on the cached config.
_MODEL_AFFECTING_FIELDS = frozenset(
    {
        "language",
        "dpi",
        "use_server_models",
        "engine_type",
        "model_base_path",
        "font_base_path",
    }
)


class _OCRModelCache:
    """Process-level cache for the OCR backend (singleton).

    The cache key is computed from the small set of fields that require
    model re-loading. All other config fields are synced automatically
    by iterating over the dataclass fields, so new fields added to
    OCRConfig are picked up without touching this class.
    """

    _instance: ProfessionalPDFOCR | None = None
    _config_hash: int = 0

    @classmethod
    def get_cached_engine(cls, config: OCRConfig) -> ProfessionalPDFOCR:
        """Return a (possibly cached) OCR engine with *config* applied."""
        config_hash = cls._compute_config_hash(config)

        if cls._instance is not None and cls._config_hash == config_hash:
            logger.info("Using cached OCR engine (models already loaded)")
            cls._sync_non_model_fields(cls._instance.config, config)
            return cls._instance

        logger.info("Creating new OCR engine (first run or config changed)")
        cls._instance = ProfessionalPDFOCR(config)
        cls._config_hash = config_hash
        return cls._instance

    @classmethod
    def _compute_config_hash(cls, config: OCRConfig) -> int:
        """Hash only the fields that require model re-loading."""
        return hash(
            tuple(
                str(getattr(config, f.name))
                for f in fields(config)
                if f.name in _MODEL_AFFECTING_FIELDS
            )
        )

    @staticmethod
    def _sync_non_model_fields(cached: OCRConfig, fresh: OCRConfig) -> None:
        """Copy every non-model field from *fresh* into *cached*."""
        for f in fields(fresh):
            if f.name not in _MODEL_AFFECTING_FIELDS:
                setattr(cached, f.name, getattr(fresh, f.name))


# Public API for cache control
class RapidOCREngine:
    """OCR Engine wrapper using ProfessionalPDFOCR backend."""

    def __init__(self, config: OCRConfig):
        """Initialize with OCR configuration."""
        self.config = config
        self.cancel_event = threading.Event()  # Cooperative cancellation
        self._stats = ProcessingStats()

    def process(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        """Run OCR processing using the backend implementation."""
        logger.info(f"Starting OCR for {input_path}")

        try:
            # Auto-calculate workers if set to 0 (auto)
            if self.config.workers <= 0:
                import multiprocessing

                self.config.workers = max(1, multiprocessing.cpu_count())

            logger.info(
                f"Initialized ProfessionalPDFOCR with config: "
                f"engine={self.config.engine_type}, workers={self.config.workers}"
            )

            # Get cached or new backend instance
            # This reuses the OCR models if config hasn't changed significantly
            ocr = _OCRModelCache.get_cached_engine(self.config)

            # Propagate cancel event so backend can check for cancellation
            ocr.cancel_event = self.cancel_event

            # Run processing
            # note: backend.process returns ProcessingStats
            stats = ocr.process(input_path, output_path, progress_callback)

            # Populate computed fields on the stats object
            full_text = stats.full_text or ""
            stats.total_words = len(full_text.split())
            stats.total_chars = len(full_text)
            stats.total_time = stats.processing_time_seconds

            # Update internal stats for GUI compatibility
            self._stats = stats

            logger.info("OCR Processing completed successfully via backend")
            return stats

        except Exception as e:
            logger.error(f"Engine execution failed: {e}")
            raise
