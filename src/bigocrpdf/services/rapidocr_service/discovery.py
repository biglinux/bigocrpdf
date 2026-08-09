"""
Model Discovery for RapidOCR.

This module provides automatic detection of available OCR models
and fonts installed on the system.
"""

import logging
from pathlib import Path

from bigocrpdf.services.rapidocr_service.resource_paths import find_font_dir, find_model_dir
from bigocrpdf.utils.i18n import _

logger = logging.getLogger(__name__)

# Model and font locations come from resource_paths, resolved per instance so a
# relocatable build finds the copy it ships with instead of the host's.


class ModelDiscovery:
    """Discovers available RapidOCR models installed on the system.

    This class scans the model directory to find available language models
    and provides methods to query which languages and model variants are
    available for OCR processing.

    Attributes:
        model_path: Path to the models directory
        font_path: Path to the fonts directory
    """

    V6_MODEL_FILES = ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx")

    def __init__(
        self,
        model_path: Path | None = None,
        font_path: Path | None = None,
    ) -> None:
        """Initialize the model discovery.

        Args:
            model_path: Models directory. Defaults to the first location
                resource_paths finds, which prefers a bundled copy over the
                system one.
            font_path: Fonts directory, resolved the same way.
        """
        self.model_path = model_path or find_model_dir()
        self.font_path = font_path or find_font_dir()
        self._cached_languages: list[tuple[str, str]] | None = None

    def get_available_languages(self) -> list[tuple[str, str]]:
        """Report the unified PP-OCRv6 model when its required pair is installed.

        Returns:
            A single compatibility tuple when PP-OCRv6 is available, otherwise empty.
        """
        if self._cached_languages is not None:
            return self._cached_languages

        if not self.model_path.exists():
            logger.warning(f"Model path not found: {self.model_path}")
            return []

        if not all((self.model_path / name).exists() for name in self.V6_MODEL_FILES):
            logger.error("Required PP-OCRv6 small models are not installed")
            return []

        self._cached_languages = [("latin", _("Automatic multilingual recognition (PP-OCRv6)"))]
        return self._cached_languages
