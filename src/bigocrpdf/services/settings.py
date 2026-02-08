"""
BigOcrPdf - Settings Service

This module handles configuration settings and file management for the application.
"""

import os
import time

from bigocrpdf.config import (
    CONFIG_DIR,
    SELECTED_FILE_PATH,
)
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

# Default settings constants
DEFAULT_LANGUAGE = "latin"  # RapidOCR uses script names, not tesseract codes
DEFAULT_SUFFIX = "ocr"
DEFAULT_DATE_FORMAT = {"year": 1, "month": 2, "day": 3}
DEFAULT_DPI = 300


class OcrSettings:
    """Class to manage OCR settings and configuration files"""

    def __init__(self) -> None:
        """Initialize OCR settings with default values"""
        # Get the centralized config manager
        self._config = get_config_manager()

        # File management
        self.selected_files: list[str] = []
        self.page_ranges: dict[
            str, tuple[int, int] | None
        ] = {}  # {file: (start, end) or None for all}
        self.destination_folder: str = ""
        self.processed_files: list[str] = []

        # OCR processing settings
        self.lang: str = DEFAULT_LANGUAGE
        self.save_in_same_folder: bool = False
        self.extracted_text: dict[str, str] = {}
        self.file_modifications: dict[str, list[dict]] = {}  # Store page states for each file
        self.original_file_paths: dict[str, str] = {}  # Map edited temp paths to original paths

        # Output filename settings
        self.pdf_suffix: str = DEFAULT_SUFFIX
        self.overwrite_existing: bool = False

        # Date inclusion settings
        self.include_date: bool = False
        self.include_year: bool = False
        self.include_month: bool = False
        self.include_day: bool = False
        self.include_time: bool = False
        self.date_format_order: dict[str, int] = DEFAULT_DATE_FORMAT.copy()

        # Text extraction settings
        self.save_txt: bool = False
        self.separate_txt_folder: bool = False
        self.txt_folder: str = ""

        # RapidOCR preprocessing settings (match reference defaults)
        self.dpi: int = DEFAULT_DPI
        self.ocr_language: str = DEFAULT_LANGUAGE
        # Color/Enhancement: OFF by default (PP-OCRv5 works best without)
        self.enable_preprocessing: bool = False  # Master switch OFF
        # Auto-detect: ON by default
        self.enable_auto_detect: bool = True
        # Geometric corrections: ON by default (reference CLI behavior)
        self.enable_deskew: bool = True
        self.enable_perspective_correction: bool = (
            True  # Perspective correction for photographed documents
        )
        self.enable_orientation_detection: bool = True
        # These only take effect if enable_preprocessing=True
        self.enable_auto_contrast: bool = False
        self.enable_auto_brightness: bool = False
        self.enable_denoise: bool = False
        self.enable_scanner_effect: bool = True
        self.scanner_effect_strength: float = 1.0
        self.enable_border_clean: bool = False
        self.enable_vintage_look: bool = False
        self.text_score_threshold: float = 0.3  # Lower threshold catches more text
        self.box_thresh: float = 0.5  # Detection box threshold
        self.unclip_ratio: float = (
            1.2  # Control text box size (1.2-2.0, lower=tighter crop=better recognition)
        )
        self.ocr_profile: str = "balanced"  # "fast", "balanced", "precise"

        # Image export settings
        self.image_export_format: str = "original"  # "original" or "jpeg"
        self.image_export_quality: int = 85  # JPEG quality (1-100)
        self.image_export_preserve_original: bool = True  # Keep original quality if possible
        self.auto_detect_quality: bool = True  # Auto-detect original image quality

        # PDF output settings
        self.convert_to_pdfa: bool = True  # Convert output to PDF/A-2b format
        self.max_file_size_mb: int = 0  # 0 = no limit; split output if exceeded

        # OCR behavior settings
        self.replace_existing_ocr: bool = True  # Replace existing OCR in PDFs

        # Parallel processing settings
        self.parallel_workers: int = 0  # 0 = auto (all CPU cores, low priority), 1 = sequential

        # Processing results (populated during OCR, cleared on reset)
        self.comparison_results: list = []
        self.ocr_boxes: dict[str, list] = {}

        # ODF export settings (loaded from config in load_settings)
        self.save_odf: bool = False
        self.odf_include_images: bool = True
        self.odf_use_formatting: bool = True

        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)

        # Load settings from JSON config manager
        self.load_settings()

    def load_settings(self) -> None:
        """Load all settings from JSON configuration"""
        # Load from JSON config manager
        # If language is not set (None), try to detect system language
        config_lang = self._config.get("ocr.language")

        if config_lang:
            # User has explicitly set a language
            self.lang = config_lang
        else:
            # No user setting - detect and save system language
            detected_lang = self._detect_default_language()
            self.lang = detected_lang
            # Save the detected language so it persists and is used by UI
            self._config.set("ocr.language", detected_lang, save_immediately=True)
            logger.info(f"Saved detected language to config: {detected_lang}")

        # Output settings
        self.pdf_suffix = self._config.get("output.suffix", DEFAULT_SUFFIX)
        self.overwrite_existing = self._config.get("output.overwrite_existing", False)
        self.save_in_same_folder = self._config.get("output.save_in_same_folder", False)
        self.destination_folder = self._config.get("output.destination_folder", "")

        # Date settings
        self.include_date = self._config.get("date.include_date", False)
        self.include_year = self._config.get("date.include_year", False)
        self.include_month = self._config.get("date.include_month", False)
        self.include_day = self._config.get("date.include_day", False)
        self.include_time = self._config.get("date.include_time", False)
        self.date_format_order = self._config.get("date.format_order", DEFAULT_DATE_FORMAT.copy())

        # Text extraction settings
        self.save_txt = self._config.get("text_extraction.save_txt", False)
        self.separate_txt_folder = self._config.get("text_extraction.separate_folder", False)
        self.txt_folder = self._config.get("text_extraction.txt_folder", "")

        self.file_modifications = {}

        # ODF export settings
        self.save_odf = self._config.get("odf_export.save_odf", False)
        self.odf_include_images = self._config.get("odf_export.include_images", True)
        self.odf_use_formatting = self._config.get("odf_export.use_formatting", True)

        # RapidOCR preprocessing settings (from JSON config, with reference defaults)
        self.dpi = self._config.get("rapidocr.dpi", DEFAULT_DPI)
        # Sync language: UI stores in self.lang (ocr.language), RapidOCR reads self.ocr_language
        self.ocr_language = self.lang
        # Color/Enhancement: OFF by default (PP-OCRv5 works best without)
        self.enable_preprocessing = self._config.get("rapidocr.enable_preprocessing", False)
        # Auto-detect: ON by default
        self.enable_auto_detect = self._config.get("rapidocr.enable_auto_detect", True)
        # Geometric corrections: ON by default (reference CLI behavior)
        self.enable_deskew = self._config.get("rapidocr.enable_deskew", True)
        self.enable_perspective_correction = self._config.get(
            "rapidocr.enable_perspective_correction", True
        )
        self.enable_orientation_detection = self._config.get(
            "rapidocr.enable_orientation_detection", True
        )
        # These only take effect if enable_preprocessing=True
        self.enable_auto_contrast = self._config.get("rapidocr.enable_auto_contrast", False)
        self.enable_auto_brightness = self._config.get("rapidocr.enable_auto_brightness", False)
        self.enable_denoise = self._config.get("rapidocr.enable_denoise", False)
        self.enable_scanner_effect = self._config.get("rapidocr.enable_scanner_effect", True)
        self.scanner_effect_strength = self._config.get("rapidocr.scanner_effect_strength", 1.0)
        self.enable_border_clean = self._config.get("rapidocr.enable_border_clean", False)
        self.enable_vintage_look = self._config.get("rapidocr.enable_vintage_look", False)
        self.text_score_threshold = self._config.get("rapidocr.text_score_threshold", 0.3)
        self.box_thresh = self._config.get("rapidocr.box_thresh", 0.5)
        self.unclip_ratio = self._config.get("rapidocr.unclip_ratio", 1.2)
        self.ocr_profile = self._config.get("rapidocr.ocr_profile", "balanced")

        # Image export settings
        self.image_export_format = self._config.get("image_export.format", "original")
        self.image_export_quality = self._config.get("image_export.quality", 85)
        self.image_export_preserve_original = self._config.get(
            "image_export.preserve_original", True
        )
        self.auto_detect_quality = self._config.get("image_export.auto_detect_quality", True)

        # PDF output settings
        self.convert_to_pdfa = self._config.get("output.convert_to_pdfa", True)
        self.max_file_size_mb = self._config.get("output.max_file_size_mb", 0)

        # OCR behavior settings
        self.replace_existing_ocr = self._config.get("ocr.replace_existing_ocr", True)

        # Load selected files from legacy file (file list not stored in JSON)
        self._load_selected_files()

        # Only initialize destination folder if it isn't already set
        if not self.destination_folder:
            self._initialize_destination_folder()

        logger.info("Settings loaded from JSON configuration")

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
        # Sync language: UI uses self.lang, RapidOCR uses self.ocr_language
        self.ocr_language = self.lang

        # OCR settings
        self._config.set("ocr.language", self.lang, save_immediately=False)

        # Output settings
        self._config.set("output.suffix", self.pdf_suffix, save_immediately=False)
        self._config.set(
            "output.overwrite_existing", self.overwrite_existing, save_immediately=False
        )
        self._config.set(
            "output.save_in_same_folder",
            self.save_in_same_folder,
            save_immediately=False,
        )
        self._config.set(
            "output.destination_folder", self.destination_folder, save_immediately=False
        )

        # Date settings
        self._config.set("date.include_date", self.include_date, save_immediately=False)
        self._config.set("date.include_year", self.include_year, save_immediately=False)
        self._config.set("date.include_month", self.include_month, save_immediately=False)
        self._config.set("date.include_day", self.include_day, save_immediately=False)
        self._config.set("date.include_time", self.include_time, save_immediately=False)
        self._config.set("date.format_order", self.date_format_order, save_immediately=False)

        # Text extraction settings
        self._config.set("text_extraction.save_txt", self.save_txt, save_immediately=False)
        self._config.set(
            "text_extraction.separate_folder",
            self.separate_txt_folder,
            save_immediately=False,
        )
        self._config.set("text_extraction.txt_folder", self.txt_folder, save_immediately=False)

        # Editor modifications
        if self.page_ranges:
            self._config.set("editor.page_ranges", self.page_ranges, save_immediately=False)

        # Don't save editor.modifications (modifications are transient)

        # ODF export settings
        self._config.set("odf_export.save_odf", self.save_odf, save_immediately=False)
        self._config.set(
            "odf_export.include_images", self.odf_include_images, save_immediately=False
        )
        self._config.set(
            "odf_export.use_formatting", self.odf_use_formatting, save_immediately=False
        )

        # RapidOCR preprocessing settings
        self._config.set("rapidocr.dpi", self.dpi, save_immediately=False)
        self._config.set("rapidocr.language", self.ocr_language, save_immediately=False)
        self._config.set(
            "rapidocr.enable_preprocessing", self.enable_preprocessing, save_immediately=False
        )
        self._config.set(
            "rapidocr.enable_auto_detect", self.enable_auto_detect, save_immediately=False
        )
        self._config.set("rapidocr.enable_deskew", self.enable_deskew, save_immediately=False)
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
            "rapidocr.scanner_effect_strength", self.scanner_effect_strength, save_immediately=False
        )
        self._config.set(
            "rapidocr.enable_border_clean", self.enable_border_clean, save_immediately=False
        )
        self._config.set(
            "rapidocr.enable_vintage_look", self.enable_vintage_look, save_immediately=False
        )
        self._config.set(
            "rapidocr.text_score_threshold", self.text_score_threshold, save_immediately=False
        )
        self._config.set("rapidocr.box_thresh", self.box_thresh, save_immediately=False)
        self._config.set("rapidocr.unclip_ratio", self.unclip_ratio, save_immediately=False)
        self._config.set("rapidocr.ocr_profile", self.ocr_profile, save_immediately=False)

        # Image export settings
        self._config.set("image_export.format", self.image_export_format, save_immediately=False)
        self._config.set("image_export.quality", self.image_export_quality, save_immediately=False)
        self._config.set(
            "image_export.preserve_original",
            self.image_export_preserve_original,
            save_immediately=False,
        )
        self._config.set(
            "image_export.auto_detect_quality",
            self.auto_detect_quality,
            save_immediately=False,
        )

        # PDF output settings
        self._config.set("output.convert_to_pdfa", self.convert_to_pdfa, save_immediately=False)
        self._config.set("output.max_file_size_mb", self.max_file_size_mb, save_immediately=False)

        # OCR behavior settings
        self._config.set(
            "ocr.replace_existing_ocr", self.replace_existing_ocr, save_immediately=False
        )

        # Save everything at once
        self._config.save()

        logger.debug("All settings saved to JSON configuration")

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
            date_components.append(
                (
                    self.date_format_order.get("year", 1),
                    f"{now.tm_year}",
                )
            )
        if self.include_month:
            date_components.append(
                (
                    self.date_format_order.get("month", 2),
                    f"{now.tm_mon:02d}",
                )
            )
        if self.include_day:
            date_components.append(
                (
                    self.date_format_order.get("day", 3),
                    f"{now.tm_mday:02d}",
                )
            )

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
        self.processed_files = []
        self.comparison_results = []

        # Clear extracted text to free memory
        if hasattr(self, "extracted_text") and self.extracted_text:
            text_count = len(self.extracted_text)
            total_chars = sum(len(text) for text in self.extracted_text.values())
            self.extracted_text.clear()
            logger.info(
                f"Cleared {text_count} extracted texts ({total_chars} characters) from memory"
            )
        else:
            self.extracted_text = {}

        # Clear OCR boxes data
        if hasattr(self, "ocr_boxes") and self.ocr_boxes:
            box_count = len(self.ocr_boxes)
            self.ocr_boxes.clear()
            logger.info(f"Cleared {box_count} OCR boxes from memory")
        else:
            self.ocr_boxes = {}

        # Full reset also clears the input file queue
        if full:
            self.selected_files = []
            self.original_file_paths = {}

        logger.info(_("Processing state reset successfully"))

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
            except Exception:
                pass  # Directory not empty or other issue, ignore

        except Exception as e:
            logger.error(f"Error cleaning temp directory {temp_dir}: {e}")

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values and save."""
        # Clear the config file
        self._config._config = self._config._get_default_config()
        self._config.save()

        # Re-initialize all attributes with defaults
        OcrSettings.__init__(self)

        logger.info("All settings have been reset to defaults")
