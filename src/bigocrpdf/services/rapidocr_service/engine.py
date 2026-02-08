"""
RapidOCR Engine Implementation using ProfessionalPDFOCR backend.

This module provides the integration between the GUI and the reference
ProfessionalPDFOCR implementation (backend.py).

Features:
- Model caching for faster consecutive OCR runs
- Automatic cache invalidation on configuration change
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from .backend import ProfessionalPDFOCR
from .config import OCRConfig

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


# === Model Cache System ===
# Keeps the OCR engine loaded between processing runs for faster consecutive OCR
class _OCRModelCache:
    """Singleton cache for OCR model instances."""

    _instance: "ProfessionalPDFOCR | None" = None
    _config_hash: int = 0

    @classmethod
    def get_cached_engine(cls, config: OCRConfig) -> "ProfessionalPDFOCR":
        """Get or create a cached OCR engine instance.

        The engine is reused if the configuration hash matches.
        Otherwise, a new engine is created and cached.

        IMPORTANT: Per-file settings (page_modifications, page_range) are
        updated on the cached engine's config before each run since they
        change between files and shouldn't trigger engine recreation.

        Args:
            config: OCR configuration

        Returns:
            ProfessionalPDFOCR instance (cached or new)
        """
        config_hash = cls._compute_config_hash(config)

        if cls._instance is not None and cls._config_hash == config_hash:
            logger.info("Using cached OCR engine (models already loaded)")
            # Update ALL non-model settings on cached engine.
            # The cache only avoids reloading OCR models (language, dpi, engine_type,
            # model paths). ALL other settings (preprocessing flags, output options,
            # per-file settings) must be propagated from the new config.
            cached_config = cls._instance.config
            cached_config.page_modifications = config.page_modifications
            cached_config.page_range = config.page_range
            # Preprocessing flags
            cached_config.enable_perspective_correction = config.enable_perspective_correction
            cached_config.enable_deskew = config.enable_deskew
            cached_config.enable_orientation_detection = config.enable_orientation_detection
            cached_config.enable_preprocessing = config.enable_preprocessing
            cached_config.enable_auto_contrast = config.enable_auto_contrast
            cached_config.enable_auto_brightness = config.enable_auto_brightness
            cached_config.enable_denoise = config.enable_denoise
            cached_config.enable_border_clean = config.enable_border_clean
            cached_config.enable_scanner_effect = config.enable_scanner_effect
            cached_config.scanner_effect_strength = config.scanner_effect_strength
            cached_config.enable_vintage_look = config.enable_vintage_look
            cached_config.vintage_bw = config.vintage_bw
            # Output options
            cached_config.convert_to_pdfa = config.convert_to_pdfa
            cached_config.image_export_format = config.image_export_format
            cached_config.image_export_quality = config.image_export_quality
            cached_config.auto_detect_quality = config.auto_detect_quality
            # OCR thresholds
            cached_config.text_score_threshold = config.text_score_threshold
            cached_config.box_thresh = config.box_thresh
            cached_config.force_full_ocr = config.force_full_ocr
            cached_config.replace_existing_ocr = config.replace_existing_ocr
            cached_config.workers = config.workers
            return cls._instance

        logger.info("Creating new OCR engine (first run or config changed)")
        cls._instance = ProfessionalPDFOCR(config)
        cls._config_hash = config_hash
        return cls._instance

    @classmethod
    def _compute_config_hash(cls, config: OCRConfig) -> int:
        """Compute a hash of configuration values that affect model loading.

        Only includes settings that require engine reinitialization when changed.
        """
        # Key settings that affect model loading
        key_values = (
            str(config.language),
            config.dpi,
            config.use_server_models,
            str(config.engine_type),
            str(config.model_base_path),
        )
        return hash(key_values)


# Public API for cache control
class RapidOCREngine:
    """OCR Engine wrapper using ProfessionalPDFOCR backend."""

    def __init__(self, config: OCRConfig):
        """Initialize with OCR configuration."""
        self.config = config
        self.cancel_event = threading.Event()  # Cooperative cancellation
        self._stats = type(
            "Stats",
            (),
            {
                "processing_time_seconds": 0.0,
                "pages_processed": 0,
                "pages_total": 0,
                "total_text_regions": 0,
                "total_words": 0,
                "total_chars": 0,
                "average_confidence": 0.0,
                "total_time": 0.0,
            },
        )()

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

            # Update internal stats for GUI compatibility
            self._stats.pages_processed = stats.pages_processed
            self._stats.pages_total = stats.pages_total
            self._stats.total_text_regions = stats.total_text_regions
            self._stats.average_confidence = stats.average_confidence
            self._stats.processing_time_seconds = stats.processing_time_seconds
            self._stats.total_time = stats.processing_time_seconds

            logger.info("OCR Processing completed successfully via backend")

            # Return stats object compatible with processor expectations
            # Processor expects: pages_processed, pages_total, total_words, total_chars, total_time
            full_text = stats.full_text or ""
            return type(
                "Stats",
                (),
                {
                    "pages_processed": stats.pages_processed,
                    "pages_total": stats.pages_total,
                    "total_words": len(full_text.split()),
                    "total_chars": len(full_text),
                    "total_time": stats.processing_time_seconds,
                    "extracted_text": full_text,
                    "ocr_boxes": stats.ocr_boxes,  # Structured OCR data for high-fidelity export
                    "split_output_files": stats.split_output_files,  # Split parts if size limit applied
                },
            )()

        except Exception as e:
            logger.error(f"Engine execution failed: {e}")
            raise
