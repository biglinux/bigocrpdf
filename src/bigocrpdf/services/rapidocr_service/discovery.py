"""
Model Discovery for RapidOCR.

This module provides automatic detection of available OCR models
and fonts installed on the system.
"""

import logging
from pathlib import Path

from bigocrpdf.utils.i18n import _

logger = logging.getLogger(__name__)

# Default paths for BigLinux
DEFAULT_MODEL_PATH = Path("/usr/share/rapidocr/models")
DEFAULT_FONT_PATH = Path("/usr/share/rapidocr/fonts")


class ModelDiscovery:
    """Discovers available RapidOCR models installed on the system.

    This class scans the model directory to find available language models
    and provides methods to query which languages and model variants are
    available for OCR processing.

    Attributes:
        model_path: Path to the models directory
        font_path: Path to the fonts directory
    """

    # Language display names — concise labels for the UI dropdown
    LANGUAGE_NAMES: dict[str, str] = {
        "latin": _("Latin Script (PT, EN, FR, DE, ES...)"),
        "en": _("English (Optimized)"),
        "ch": _("Chinese Simplified"),
        "korean": _("Korean"),
        "arabic": _("Arabic Script (AR, FA, UR...)"),
        "cyrillic": _("Cyrillic Script (RU, UK, BG...)"),
        "eslav": _("East Slavic (RU, BY, UK)"),
        "el": _("Greek"),
        "devanagari": _("Devanagari (Hindi, Marathi, Nepali)"),
        "ta": _("Tamil"),
        "te": _("Telugu"),
        "th": _("Thai"),
    }

    # Full language descriptions for the help dialog
    LANGUAGE_DETAILS: dict[str, str] = {
        "latin": (
            "French, German, Afrikaans, Italian, Spanish, Bosnian, "
            "Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, "
            "Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, "
            "Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, "
            "Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, "
            "Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, "
            "Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, "
            "Galician, Luxembourgish, Romansh, Catalan, Quechua"
        ),
        "en": "English",
        "ch": "Chinese (Simplified), English",
        "korean": "Korean, English",
        "arabic": ("Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English"),
        "cyrillic": (
            "Russian, Belarusian, Ukrainian, Serbian (Cyrillic), "
            "Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, "
            "Avar, Dargin, Ingush, Chechen, Lak, Lezgin, "
            "Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, "
            "Tatar, Chuvash, Bashkir, Malian, Moldovan, "
            "Udmurt, Komi, Ossetian, Buryat, Kalmyk, "
            "Tuvan, Sakha, Karakalpak, English"
        ),
        "eslav": "Russian, Belarusian, Ukrainian, English",
        "el": "Greek, English",
        "devanagari": (
            "Hindi, Marathi, Nepali, Bihari, Maithili, "
            "Angika, Bhojpuri, Magahi, Santali, Newari, "
            "Konkani, Sanskrit, Haryanvi, English"
        ),
        "ta": "Tamil, English",
        "te": "Telugu, English",
        "th": "Thai, English",
    }

    # Model file patterns for each language (matching actual installed files)
    MODEL_PATTERNS: dict[str, list[str]] = {
        "latin": [
            "latin_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "en": [
            "en_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "ch": [
            "ch_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "korean": [
            "korean_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "arabic": [
            "arabic_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "cyrillic": [
            "cyrillic_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "eslav": [
            "eslav_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "el": [
            "el_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "devanagari": [
            "devanagari_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "ta": [
            "ta_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "te": [
            "te_PP-OCRv5_rec_mobile_infer.onnx",
        ],
        "th": [
            "th_PP-OCRv5_rec_mobile_infer.onnx",
        ],
    }

    def __init__(
        self,
        model_path: Path | None = None,
        font_path: Path | None = None,
    ) -> None:
        """Initialize the model discovery.

        Args:
            model_path: Path to models directory. Defaults to /usr/share/rapidocr/models
            font_path: Path to fonts directory. Defaults to /usr/share/rapidocr/fonts
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.font_path = font_path or DEFAULT_FONT_PATH
        self._cached_languages: list[tuple[str, str]] | None = None

    def get_available_languages(self) -> list[tuple[str, str]]:
        """Get list of available languages based on installed models.

        Returns:
            List of (language_code, display_name) tuples, sorted alphabetically
            by display name. Always includes at least 'latin' as fallback.
        """
        if self._cached_languages is not None:
            return self._cached_languages

        if not self.model_path.exists():
            logger.warning(f"Model path not found: {self.model_path}")
            return [("latin", self.LANGUAGE_NAMES["latin"])]

        available: list[tuple[str, str]] = []

        for lang_code, patterns in self.MODEL_PATTERNS.items():
            for pattern in patterns:
                if (self.model_path / pattern).exists():
                    display_name = self.LANGUAGE_NAMES.get(lang_code, lang_code.title())
                    available.append((lang_code, display_name))
                    break

        if not available:
            logger.warning("No models found, using fallback 'latin'")
            available = [("latin", self.LANGUAGE_NAMES["latin"])]

        # Sort by display name, but put "Latim" first
        available.sort(key=lambda x: (x[0] != "latin", x[1]))

        self._cached_languages = available
        return available
