"""
BigOcrPdf - Settings Service

This module handles configuration settings and file management for the application.
The three sub-objects (FileQueueManager, OutputConfig, PreprocessingConfig) live
in their own modules; OcrSettings acts as a transparent facade.
"""

from __future__ import annotations

import os
import time
from typing import Any

from bigocrpdf.config import (
    CONFIG_DIR,
    SELECTED_FILE_PATH,
)
from bigocrpdf.services.file_queue import FileQueueManager
from bigocrpdf.services.output_config import DEFAULT_DATE_FORMAT, DEFAULT_SUFFIX, OutputConfig
from bigocrpdf.services.preprocessing_config import (
    PreprocessingConfig,
)
from bigocrpdf.services.rapidocr_service.config import (
    DEFAULT_AUTO_DETECT_QUALITY,
    DEFAULT_BOX_THRESH,
    DEFAULT_DETECTION_FULL_RESOLUTION,
    DEFAULT_DPI,
    DEFAULT_ENABLE_AUTO_BRIGHTNESS,
    DEFAULT_ENABLE_AUTO_CONTRAST,
    DEFAULT_ENABLE_BASELINE_DEWARP,
    DEFAULT_ENABLE_BILEVEL_COMPRESSION,
    DEFAULT_ENABLE_BORDER_CLEAN,
    DEFAULT_ENABLE_DENOISE,
    DEFAULT_ENABLE_DESKEW,
    DEFAULT_ENABLE_ORIENTATION_DETECTION,
    DEFAULT_ENABLE_PERSPECTIVE_CORRECTION,
    DEFAULT_ENABLE_PREPROCESSING,
    DEFAULT_ENABLE_SCANNER_EFFECT,
    DEFAULT_ENABLE_VINTAGE_LOOK,
    DEFAULT_FORCE_BILEVEL_COMPRESSION,
    DEFAULT_IMAGE_EXPORT_FORMAT,
    DEFAULT_IMAGE_EXPORT_QUALITY,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_SCANNER_EFFECT_STRENGTH,
    DEFAULT_TEXT_SCORE_THRESHOLD,
    DEFAULT_UNCLIP_RATIO,
    DEFAULT_VINTAGE_BW,
)
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

# Sub-object attribute names used by OcrSettings delegation
_SUB_OBJECTS = ("file_queue", "output", "preprocessing")


def _build_delegation_map(*sub_instances: object) -> dict[str, str]:
    """Build a static {attr_name: sub_object_field} map once at init time."""
    mapping: dict[str, str] = {}
    for sub, name in zip(sub_instances, _SUB_OBJECTS, strict=True):
        for attr in vars(sub):
            if not attr.startswith("_"):
                mapping[attr] = name
        # Also include public methods
        for attr in dir(type(sub)):
            if not attr.startswith("_") and callable(getattr(sub, attr, None)):
                mapping.setdefault(attr, name)
    return mapping


class OcrSettings:
    """Facade over FileQueueManager, OutputConfig, and PreprocessingConfig.

    All attribute reads/writes are transparently delegated to the
    appropriate sub-object, so existing code that accesses
    ``settings.enable_deskew`` or ``settings.selected_files`` continues
    to work without any changes.
    """

    def __init__(self) -> None:
        # Use object.__setattr__ during init to bypass delegation
        sup = object.__setattr__.__get__(self)
        sup("_config", get_config_manager())
        fq = FileQueueManager()
        out = OutputConfig()
        pre = PreprocessingConfig()
        sup("file_queue", fq)
        sup("output", out)
        sup("preprocessing", pre)

        # Static delegation map: {attr_name: sub_object_field} — O(1) lookup
        sup("_delegation_map", _build_delegation_map(fq, out, pre))

        # Attributes that stay directly on OcrSettings (cross-cutting / transient)
        sup("lang", DEFAULT_LANGUAGE)
        sup("extracted_text", {})
        sup("comparison_results", [])
        sup("ocr_boxes", {})

        # Enable delegation for all subsequent attribute access
        sup("_initialized", True)

        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)

        # Load settings from JSON config manager
        self.load_settings()

    # -- Delegation machinery --------------------------------------------------

    def __getattr__(self, name: str) -> object:
        """Delegate attribute reads to sub-objects via static map (O(1) lookup)."""
        try:
            dmap = object.__getattribute__(self, "_delegation_map")
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from None
        sub_name = dmap.get(name)
        if sub_name is not None:
            sub = object.__getattribute__(self, sub_name)
            return getattr(sub, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: object) -> None:
        """Delegate attribute writes to the sub-object that owns *name* (O(1) lookup)."""
        if not self.__dict__.get("_initialized"):
            object.__setattr__(self, name, value)
            return
        dmap = object.__getattribute__(self, "_delegation_map")
        sub_name = dmap.get(name)
        if sub_name is not None:
            sub = object.__getattribute__(self, sub_name)
            setattr(sub, name, value)
            return
        object.__setattr__(self, name, value)

    def load_settings(self) -> None:
        """Load all settings from JSON configuration"""
        self._load_language_settings()
        self._load_output_settings()
        self._load_date_settings()
        self._load_text_extraction_settings()
        self._load_odf_settings()
        self._load_preprocessing_settings()
        self._load_image_export_settings()
        self._load_pdf_output_settings()
        self._load_bilevel_settings()
        self.quick_start_mode = self._config.get("ui.quick_start_mode", True)

        # Load selected files from legacy file (file list not stored in JSON)
        self._load_selected_files()

        # Only initialize destination folder if it isn't already set
        if not self.destination_folder:
            self._initialize_destination_folder()

        logger.info("Settings loaded from JSON configuration")

    def _load_language_settings(self) -> None:
        config_lang = self._config.get("ocr.language")
        if config_lang:
            self.lang = config_lang
        else:
            detected_lang = self._detect_default_language()
            self.lang = detected_lang
            self._config.set("ocr.language", detected_lang, save_immediately=True)
            logger.info(f"Saved detected language to config: {detected_lang}")
        self.ocr_language = self.lang
        self.replace_existing_ocr = self._config.get("ocr.replace_existing_ocr", False)
        self.enhance_embedded_images = self._config.get("ocr.enhance_embedded_images", False)

    def _load_output_settings(self) -> None:
        self.pdf_suffix = self._config.get("output.suffix", DEFAULT_SUFFIX)
        self.use_original_filename = self._config.get("output.use_original_filename", False)
        self.overwrite_existing = self._config.get("output.overwrite_existing", False)
        self.save_in_same_folder = self._config.get("output.save_in_same_folder", False)
        self.destination_folder = self._config.get("output.destination_folder", "")

    def _load_date_settings(self) -> None:
        self.include_date = self._config.get("date.include_date", False)
        self.include_year = self._config.get("date.include_year", False)
        self.include_month = self._config.get("date.include_month", False)
        self.include_day = self._config.get("date.include_day", False)
        self.include_time = self._config.get("date.include_time", False)
        self.date_format_order = self._config.get("date.format_order", DEFAULT_DATE_FORMAT.copy())

    def _load_text_extraction_settings(self) -> None:
        self.save_txt = self._config.get("text_extraction.save_txt", False)
        self.separate_txt_folder = self._config.get("text_extraction.separate_folder", False)
        self.txt_folder = self._config.get("text_extraction.txt_folder", "")
        self.file_modifications: dict[str, dict[str, Any]] = {}

    def _load_odf_settings(self) -> None:
        self.save_odf = self._config.get("odf_export.save_odf", False)
        self.odf_include_images = self._config.get("odf_export.include_images", True)
        self.odf_use_formatting = self._config.get("odf_export.use_formatting", True)
        self.odf_open_after_export = self._config.get("odf_export.open_after_export", False)

    def _load_preprocessing_settings(self) -> None:
        self.dpi = self._config.get("rapidocr.dpi", DEFAULT_DPI)
        self.enable_preprocessing = self._config.get(
            "rapidocr.enable_preprocessing", DEFAULT_ENABLE_PREPROCESSING
        )
        self.enable_deskew = self._config.get("rapidocr.enable_deskew", DEFAULT_ENABLE_DESKEW)
        self.enable_baseline_dewarp = self._config.get(
            "rapidocr.enable_baseline_dewarp", DEFAULT_ENABLE_BASELINE_DEWARP
        )
        self.enable_perspective_correction = self._config.get(
            "rapidocr.enable_perspective_correction", DEFAULT_ENABLE_PERSPECTIVE_CORRECTION
        )
        self.enable_orientation_detection = self._config.get(
            "rapidocr.enable_orientation_detection", DEFAULT_ENABLE_ORIENTATION_DETECTION
        )
        self.enable_auto_contrast = self._config.get(
            "rapidocr.enable_auto_contrast", DEFAULT_ENABLE_AUTO_CONTRAST
        )
        self.enable_auto_brightness = self._config.get(
            "rapidocr.enable_auto_brightness", DEFAULT_ENABLE_AUTO_BRIGHTNESS
        )
        self.enable_denoise = self._config.get("rapidocr.enable_denoise", DEFAULT_ENABLE_DENOISE)
        self.enable_scanner_effect = self._config.get(
            "rapidocr.enable_scanner_effect", DEFAULT_ENABLE_SCANNER_EFFECT
        )
        self.scanner_effect_strength = self._config.get(
            "rapidocr.scanner_effect_strength", DEFAULT_SCANNER_EFFECT_STRENGTH
        )
        self.enable_border_clean = self._config.get(
            "rapidocr.enable_border_clean", DEFAULT_ENABLE_BORDER_CLEAN
        )
        self.enable_vintage_look = self._config.get(
            "rapidocr.enable_vintage_look", DEFAULT_ENABLE_VINTAGE_LOOK
        )
        self.vintage_bw = self._config.get("rapidocr.vintage_bw", DEFAULT_VINTAGE_BW)
        self.text_score_threshold = self._config.get(
            "rapidocr.text_score_threshold", DEFAULT_TEXT_SCORE_THRESHOLD
        )
        self.box_thresh = self._config.get("rapidocr.box_thresh", DEFAULT_BOX_THRESH)
        self.unclip_ratio = self._config.get("rapidocr.unclip_ratio", DEFAULT_UNCLIP_RATIO)
        self.ocr_profile = self._config.get("rapidocr.ocr_profile", "balanced")
        self.detection_full_resolution = self._config.get(
            "rapidocr.detection_full_resolution", DEFAULT_DETECTION_FULL_RESOLUTION
        )

    def _load_image_export_settings(self) -> None:
        self.image_export_format = self._config.get(
            "image_export.format", DEFAULT_IMAGE_EXPORT_FORMAT
        )
        self.image_export_quality = self._config.get(
            "image_export.quality", DEFAULT_IMAGE_EXPORT_QUALITY
        )
        self.image_export_preserve_original = self._config.get(
            "image_export.preserve_original", True
        )
        self.auto_detect_quality = self._config.get(
            "image_export.auto_detect_quality", DEFAULT_AUTO_DETECT_QUALITY
        )

    def _load_pdf_output_settings(self) -> None:
        self.convert_to_pdfa = self._config.get("output.convert_to_pdfa", True)
        self.max_file_size_mb = self._config.get(
            "output.max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB
        )

    def _load_bilevel_settings(self) -> None:
        self.enable_bilevel_compression = self._config.get(
            "output.enable_bilevel_compression", DEFAULT_ENABLE_BILEVEL_COMPRESSION
        )
        self.force_bilevel_compression = self._config.get(
            "output.force_bilevel_compression", DEFAULT_FORCE_BILEVEL_COMPRESSION
        )

    def _detect_default_language(self) -> str:
        """Detect system language and find best match in RapidOCR models.

        Maps Linux 2-letter locale codes to RapidOCR script names.

        Returns:
            Detected language code for RapidOCR, or 'latin' as fallback
        """
        try:
            # Get system language from LANG environment variable
            lang = os.environ.get("LANG", "")
            short_lang = lang[:2].lower() if lang else ""

            # Map common 2-letter ISO 639-1 codes to RapidOCR script names
            mapping = {
                # Latin script languages
                "pt": "latin",  # Portuguese
                "es": "latin",  # Spanish
                "en": "english",  # English (has dedicated model)
                "fr": "latin",  # French
                "de": "latin",  # German
                "it": "latin",  # Italian
                "nl": "latin",  # Dutch
                "pl": "slavic",  # Polish (Slavic)
                "sv": "latin",  # Swedish
                "no": "latin",  # Norwegian
                "da": "latin",  # Danish
                "fi": "latin",  # Finnish
                "cs": "slavic",  # Czech (Slavic)
                "sk": "slavic",  # Slovak (Slavic)
                "hr": "slavic",  # Croatian (Slavic)
                "tr": "latin",  # Turkish
                "ro": "latin",  # Romanian
                "hu": "latin",  # Hungarian
                # Cyrillic script languages
                "ru": "cyrillic",  # Russian
                "uk": "cyrillic",  # Ukrainian
                "bg": "cyrillic",  # Bulgarian
                # Greek
                "el": "greek",  # Greek
                # Arabic script
                "ar": "arabic",  # Arabic
                "he": "arabic",  # Hebrew (uses Arabic model)
                # Korean
                "ko": "korean",  # Korean
                # Thai
                "th": "thai",  # Thai
                # Devanagari script
                "hi": "devanagari",  # Hindi
                # Tamil
                "ta": "tamil",  # Tamil
                # Telugu
                "te": "telugu",  # Telugu
            }

            target_script = mapping.get(short_lang, "latin")

            # Check if target language model is installed
            try:
                from bigocrpdf.services.rapidocr_service import ModelDiscovery

                discovery = ModelDiscovery()
                available = discovery.get_available_languages()
                # Extract just the language codes from tuples (code, name)
                available_codes = [code for code, _name in available]

                if target_script in available_codes:
                    logger.info(f"Detected system language script: {target_script}")
                    return target_script
                elif available_codes:
                    # Use first available language
                    fallback = available_codes[0]
                    logger.info(f"Detected script {target_script} not installed, using {fallback}")
                    return fallback

            except ImportError:
                logger.warning("ModelDiscovery not available, using default language")

            return DEFAULT_LANGUAGE

        except Exception as e:
            logger.warning(f"Failed to detect system language: {e}")
            return DEFAULT_LANGUAGE

    def add_files(self, file_paths: list[str]) -> int:
        """Add files to the selected files list

        Args:
            file_paths: List of file paths to add

        Returns:
            Number of files successfully added
        """
        if not file_paths:
            return 0

        logger.info(_("Attempting to add {0} files").format(len(file_paths)))

        # Filter and collect valid files (PDFs and Images)
        valid_files = self._filter_valid_files(file_paths)

        # Add valid files and update count
        if valid_files:
            self.selected_files.extend(valid_files)

            self._save_selected_files()

            # Only initialize destination if it's not already set by the user
            if not self.destination_folder:
                self._initialize_destination_folder()

            logger.info(_("Successfully added {0} files").format(len(valid_files)))
        else:
            logger.warning(_("No valid files were found to add"))

        return len(valid_files)

    def _filter_valid_files(self, file_paths: list[str]) -> list[str]:
        """Filter a list of paths to only include valid files (PDF and Images)

        Args:
            file_paths: List of file paths to filter

        Returns:
            List of valid file paths
        """
        valid_files: list[str] = []

        for file_path in file_paths:
            # Skip empty paths
            if not file_path:
                logger.warning(_("Empty file path provided"))
                continue

            # Skip non-existent files
            if not os.path.exists(file_path):
                logger.warning(_("File does not exist: {0}").format(file_path))
                continue

            # Skip non-supported files
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in [
                ".pdf",
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
                ".avif",
            ]:
                logger.warning(_("Unsupported file type: {0}").format(file_path))
                continue

            # Skip duplicates
            if file_path in self.selected_files:
                logger.info(_("File already in list: {0}").format(file_path))
                continue

            # File is valid, add it
            logger.info(_("Adding valid file: {0}").format(file_path))
            valid_files.append(file_path)

        return valid_files

    def save_settings(
        self,
        lang: str,
        destination_folder: str,
        save_in_same_folder: bool = False,
    ) -> None:
        """Save current settings to configuration files

        Args:
            lang: OCR language code
            destination_folder: Path to save output files
            save_in_same_folder: Whether to save in same folder as original
        """
        try:
            # Update values
            self.lang = lang or DEFAULT_LANGUAGE
            self.destination_folder = destination_folder
            self.save_in_same_folder = save_in_same_folder

            # Save all settings to JSON
            self._save_all_settings()

            logger.info(_("Settings saved successfully"))

        except Exception as e:
            logger.error(_("Error saving settings: {0}").format(e))
            raise

    def _save_all_settings(self) -> None:
        """Save all settings to JSON configuration"""
        self.ocr_language = self.lang
        self._save_language_settings()
        self._save_output_settings()
        self._save_date_settings()
        self._save_text_extraction_settings()
        self._save_editor_settings()
        self._save_odf_settings()
        self._save_preprocessing_settings()
        self._save_image_export_settings()
        self._save_pdf_output_settings()
        self._save_bilevel_settings()
        self._config.set("ui.quick_start_mode", self.quick_start_mode, save_immediately=False)
        self._config.save()
        logger.debug("All settings saved to JSON configuration")

    def _save_language_settings(self) -> None:
        self._config.set("ocr.language", self.lang, save_immediately=False)
        self._config.set(
            "ocr.replace_existing_ocr", self.replace_existing_ocr, save_immediately=False
        )
        self._config.set(
            "ocr.enhance_embedded_images", self.enhance_embedded_images, save_immediately=False
        )

    def _save_output_settings(self) -> None:
        self._config.set("output.suffix", self.pdf_suffix, save_immediately=False)
        self._config.set(
            "output.use_original_filename", self.use_original_filename, save_immediately=False
        )
        self._config.set(
            "output.overwrite_existing", self.overwrite_existing, save_immediately=False
        )
        self._config.set(
            "output.save_in_same_folder", self.save_in_same_folder, save_immediately=False
        )
        self._config.set(
            "output.destination_folder", self.destination_folder, save_immediately=False
        )

    def _save_date_settings(self) -> None:
        self._config.set("date.include_date", self.include_date, save_immediately=False)
        self._config.set("date.include_year", self.include_year, save_immediately=False)
        self._config.set("date.include_month", self.include_month, save_immediately=False)
        self._config.set("date.include_day", self.include_day, save_immediately=False)
        self._config.set("date.include_time", self.include_time, save_immediately=False)
        self._config.set("date.format_order", self.date_format_order, save_immediately=False)

    def _save_text_extraction_settings(self) -> None:
        self._config.set("text_extraction.save_txt", self.save_txt, save_immediately=False)
        self._config.set(
            "text_extraction.separate_folder", self.separate_txt_folder, save_immediately=False
        )
        self._config.set("text_extraction.txt_folder", self.txt_folder, save_immediately=False)

    def _save_editor_settings(self) -> None:
        if self.page_ranges:
            self._config.set("editor.page_ranges", self.page_ranges, save_immediately=False)

    def _save_odf_settings(self) -> None:
        self._config.set("odf_export.save_odf", self.save_odf, save_immediately=False)
        self._config.set(
            "odf_export.include_images", self.odf_include_images, save_immediately=False
        )
        self._config.set(
            "odf_export.use_formatting", self.odf_use_formatting, save_immediately=False
        )
        self._config.set(
            "odf_export.open_after_export", self.odf_open_after_export, save_immediately=False
        )

    def _save_preprocessing_settings(self) -> None:
        self._config.set("rapidocr.dpi", self.dpi, save_immediately=False)
        self._config.set("rapidocr.language", self.ocr_language, save_immediately=False)
        self._config.set(
            "rapidocr.enable_preprocessing", self.enable_preprocessing, save_immediately=False
        )
        self._config.set("rapidocr.enable_deskew", self.enable_deskew, save_immediately=False)
        self._config.set(
            "rapidocr.enable_baseline_dewarp",
            self.enable_baseline_dewarp,
            save_immediately=False,
        )
        self._config.set(
            "rapidocr.enable_perspective_correction",
            self.enable_perspective_correction,
            save_immediately=False,
        )
        self._config.set(
            "rapidocr.enable_orientation_detection",
            self.enable_orientation_detection,
            save_immediately=False,
        )
        self._config.set(
            "rapidocr.enable_auto_contrast", self.enable_auto_contrast, save_immediately=False
        )
        self._config.set(
            "rapidocr.enable_auto_brightness", self.enable_auto_brightness, save_immediately=False
        )
        self._config.set("rapidocr.enable_denoise", self.enable_denoise, save_immediately=False)
        self._config.set(
            "rapidocr.enable_scanner_effect", self.enable_scanner_effect, save_immediately=False
        )
        self._config.set(
            "rapidocr.scanner_effect_strength",
            self.scanner_effect_strength,
            save_immediately=False,
        )
        self._config.set(
            "rapidocr.enable_border_clean", self.enable_border_clean, save_immediately=False
        )
        self._config.set(
            "rapidocr.enable_vintage_look", self.enable_vintage_look, save_immediately=False
        )
        self._config.set("rapidocr.vintage_bw", self.vintage_bw, save_immediately=False)
        self._config.set(
            "rapidocr.text_score_threshold", self.text_score_threshold, save_immediately=False
        )
        self._config.set("rapidocr.box_thresh", self.box_thresh, save_immediately=False)
        self._config.set("rapidocr.unclip_ratio", self.unclip_ratio, save_immediately=False)
        self._config.set("rapidocr.ocr_profile", self.ocr_profile, save_immediately=False)
        self._config.set(
            "rapidocr.detection_full_resolution",
            self.detection_full_resolution,
            save_immediately=False,
        )

    def _save_image_export_settings(self) -> None:
        self._config.set("image_export.format", self.image_export_format, save_immediately=False)
        self._config.set("image_export.quality", self.image_export_quality, save_immediately=False)
        self._config.set(
            "image_export.preserve_original",
            self.image_export_preserve_original,
            save_immediately=False,
        )
        self._config.set(
            "image_export.auto_detect_quality", self.auto_detect_quality, save_immediately=False
        )

    def _save_pdf_output_settings(self) -> None:
        self._config.set("output.convert_to_pdfa", self.convert_to_pdfa, save_immediately=False)
        self._config.set("output.max_file_size_mb", self.max_file_size_mb, save_immediately=False)

    def _save_bilevel_settings(self) -> None:
        self._config.set(
            "output.enable_bilevel_compression",
            self.enable_bilevel_compression,
            save_immediately=False,
        )
        self._config.set(
            "output.force_bilevel_compression",
            self.force_bilevel_compression,
            save_immediately=False,
        )

    def get_pdf_suffix(self) -> str:
        """Get the formatted PDF suffix with date elements if enabled

        Returns:
            The formatted suffix string for PDF files
        """
        # Start with the base suffix
        suffix = self.pdf_suffix or DEFAULT_SUFFIX

        # If date inclusion is not enabled, return just the suffix
        if not self.include_date:
            return suffix

        # Get current time
        now = time.localtime()

        # Initialize date components with position ordering
        date_components: list[tuple[int, str]] = []

        # Add date elements with their preferred order
        if self.include_year:
            date_components.append((
                self.date_format_order.get("year", 1),
                f"{now.tm_year}",
            ))
        if self.include_month:
            date_components.append((
                self.date_format_order.get("month", 2),
                f"{now.tm_mon:02d}",
            ))
        if self.include_day:
            date_components.append((
                self.date_format_order.get("day", 3),
                f"{now.tm_mday:02d}",
            ))

        # Sort components by their position value
        date_components.sort(key=lambda x: x[0])

        # Extract ordered date parts
        date_parts = [component[1] for component in date_components]

        # Add time separately (always comes last)
        if self.include_time:
            date_parts.append(f"{now.tm_hour:02d}{now.tm_min:02d}")

        # If we have date parts, add them to the suffix
        if date_parts:
            date_str = "-".join(date_parts)
            return f"{suffix}-{date_str}"

        # Otherwise just return the suffix
        return suffix

    def _save_selected_files(self) -> None:
        """Save the current list of selected files to the configuration file"""
        try:
            with open(SELECTED_FILE_PATH, "w", encoding="utf-8") as f:
                for file_path in self.selected_files:
                    f.write(f"{file_path}\n")
            logger.info(_("Saved {0} selected files").format(len(self.selected_files)))
        except Exception as e:
            logger.error(_("Error saving selected files: {0}").format(e))

    def _load_selected_files(self) -> None:
        """Load selected files from configuration"""
        # Initialize selected files as empty list to ensure it's always iterable
        self.selected_files = []
        self.pages_count = 0

        if not os.path.exists(SELECTED_FILE_PATH):
            return

        try:
            with open(SELECTED_FILE_PATH, encoding="utf-8") as f:
                file_lines = f.readlines()
                if file_lines:  # Check if there are any lines
                    self.selected_files = [line.strip() for line in file_lines if line.strip()]

            # Filter to only existing files
            self.selected_files = [f for f in self.selected_files if os.path.exists(f)]

        except Exception as e:
            logger.error(_("Error loading selected files: {0}").format(e))
            # Ensure selected_files is always a list
            self.selected_files = []

    # NOTE: Legacy _load_* methods removed - settings now loaded from JSON via ConfigManager

    def _initialize_destination_folder(self) -> None:
        """Initialize the destination folder path based on selected files"""
        if not self.selected_files:
            self.destination_folder = ""
            return

        first_file = self.selected_files[0]
        file_folder = os.path.dirname(first_file)

        # Check if folder is writable, if not use home directory
        if not os.access(file_folder, os.W_OK):
            file_folder = os.path.expanduser("~")

        # Set the destination folder
        self.destination_folder = file_folder

    def reset_processing_state(self, *, full: bool = False) -> None:
        """Reset processing-related state for a new OCR run.

        Args:
            full: If True, also clears the file queue (selected_files,
                  original_file_paths). Use ``full=True``
                  when the user cancels processing and returns to the
                  settings page to start from scratch.
        """
        # Clear results
        self.processed_files: list[str] = []
        self.comparison_results: list[Any] = []

        # Clear extracted text to free memory
        if hasattr(self, "extracted_text") and self.extracted_text:
            text_count = len(self.extracted_text)
            total_chars = sum(len(text) for text in self.extracted_text.values())
            self.extracted_text.clear()
            logger.info(
                f"Cleared {text_count} extracted texts ({total_chars} characters) from memory"
            )
        else:
            self.extracted_text: dict[str, str] = {}

        # Clear OCR boxes data
        if hasattr(self, "ocr_boxes") and self.ocr_boxes:
            box_count = len(self.ocr_boxes)
            self.ocr_boxes.clear()
            logger.info(f"Cleared {box_count} OCR boxes from memory")
        else:
            self.ocr_boxes: dict[str, list[Any]] = {}

        # Full reset also clears the input file queue
        if full:
            self.selected_files = []
            self.original_file_paths: dict[str, str] = {}

        logger.info(_("Processing state reset successfully"))

    def display_name(self, file_path: str) -> str:
        """Return a user-friendly display name for a queued file."""
        original = self.original_file_paths.get(file_path)
        return os.path.basename(original or file_path)

    def cleanup_temp_files(self, processed_files: list[str]) -> None:
        """Clean up temporary files after OCR processing

        Args:
            processed_files: List of processed output files
        """
        try:
            for output_file in processed_files:
                if not output_file or not os.path.exists(output_file):
                    continue

                # Clean up .temp directory for this file
                temp_dir = os.path.join(os.path.dirname(output_file), ".temp")
                if os.path.exists(temp_dir):
                    self._cleanup_temp_directory(temp_dir, output_file)

            # Clean up editor merge temp files (bigocr_merge_*.pdf)
            if self.original_file_paths:
                for temp_path, _original_path in list(self.original_file_paths.items()):
                    if os.path.exists(temp_path) and "bigocr_merge_" in os.path.basename(temp_path):
                        try:
                            os.remove(temp_path)
                            logger.info(f"Removed merge temp file: {os.path.basename(temp_path)}")
                        except OSError as e:
                            logger.warning(f"Could not remove merge temp file: {e}")
                self.original_file_paths.clear()

            logger.info(_("Temporary files cleanup completed"))

        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    def _cleanup_temp_directory(self, temp_dir: str, output_file: str) -> None:
        """Clean up specific temporary directory

        Args:
            temp_dir: Path to temporary directory
            output_file: Output file path to match temp files
        """
        import glob

        try:
            # Get base name for matching temp files
            base_name = os.path.basename(os.path.splitext(output_file)[0])

            # Find temp files related to this output file
            temp_pattern = os.path.join(temp_dir, f"temp_{base_name}*")
            temp_files = glob.glob(temp_pattern)

            # Remove matching temp files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    logger.info(f"Removed temporary file: {os.path.basename(temp_file)}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")

            # Try to remove temp directory if empty
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    logger.info("Removed empty temporary directory")
            except OSError as e:
                logger.debug("Could not remove temp directory: %s", e)

        except Exception as e:
            logger.error(f"Error cleaning temp directory {temp_dir}: {e}")

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values and save."""
        # Clear the config file
        self._config._config = self._config._get_default_config()
        self._config.save()

        # Disable delegation during re-init
        object.__setattr__(self, "_initialized", False)
        OcrSettings.__init__(self)

        logger.info("All settings have been reset to defaults")
