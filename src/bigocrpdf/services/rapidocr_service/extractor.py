"""
PDF Image Extraction for RapidOCR.

This module provides the PDFImageExtractor context manager for
temporary directory management during PDF processing.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an extracted page image.

    Attributes:
        page_num: 1-based page number
        image: Image as numpy array (BGR format for OpenCV)
        width: Image width in pixels
        height: Image height in pixels
    """

    page_num: int
    image: np.ndarray
    width: int
    height: int


class PDFImageExtractor:
    """Context manager for temporary directory during PDF processing.

    Provides a managed temporary directory for intermediate files
    generated during PDF image extraction and OCR.
    """

    def __init__(self, config: "OCRConfig") -> None:
        """Initialize the extractor.

        Args:
            config: OCR configuration object
        """
        self.config = config
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "PDFImageExtractor":
        """Context manager entry."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="bigocrpdf_")
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Context manager exit - cleanup temp files."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    @property
    def temp_path(self) -> Path:
        """Get the temporary directory path."""
        if not self._temp_dir:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="bigocrpdf_")
        return Path(self._temp_dir.name)
