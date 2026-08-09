"""Application settings, queue state, and persistence."""

from __future__ import annotations

import json
import os
import stat
import time
from copy import deepcopy
from typing import Any

from bigocrpdf.config import (
    CONFIG_DIR,
    SELECTED_FILE_PATH,
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
    DEFAULT_PAGE_LAYOUT,
    DEFAULT_SCANNER_EFFECT_STRENGTH,
    DEFAULT_TEXT_SCORE_THRESHOLD,
    DEFAULT_UNCLIP_RATIO,
    DEFAULT_VINTAGE_BW,
    DEFAULT_WORKERS,
    OCRConfig,
)
from bigocrpdf.utils.config_manager import DEFAULT_CONFIG, get_config_manager
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger

DEFAULT_SUFFIX = "ocr"
DEFAULT_DATE_FORMAT = {"year": 1, "month": 2, "day": 3}


def _deduplicate_file_paths(file_paths: list[str]) -> list[str]:
    """Keep the first spelling of each filesystem identity."""
    unique_paths: list[str] = []
    seen: set[str] = set()
    for file_path in file_paths:
        identity = os.path.realpath(file_path)
        if identity in seen:
            continue
        seen.add(identity)
        unique_paths.append(file_path)
    return unique_paths


# Flat attributes remain the compatibility API, but each persisted preference is
# declared once instead of repeated across initialization, loading, and saving.
_PERSISTED_SETTINGS: dict[str, tuple[str, Any]] = {
    "replace_existing_ocr": ("ocr.replace_existing_ocr", False),
    "enhance_embedded_images": ("ocr.enhance_embedded_images", False),
    "pdf_suffix": ("output.suffix", DEFAULT_SUFFIX),
    "use_original_filename": ("output.use_original_filename", False),
    "overwrite_existing": ("output.overwrite_existing", False),
    "save_in_same_folder": (
        "output.save_in_same_folder",
        DEFAULT_CONFIG["output"]["save_in_same_folder"],
    ),
    "destination_folder": ("output.destination_folder", ""),
    "include_date": ("date.include_date", False),
    "include_year": ("date.include_year", False),
    "include_month": ("date.include_month", False),
    "include_day": ("date.include_day", False),
    "include_time": ("date.include_time", False),
    "date_format_order": ("date.format_order", DEFAULT_DATE_FORMAT),
    "save_txt": ("text_extraction.save_txt", False),
    "separate_txt_folder": ("text_extraction.separate_folder", False),
    "txt_folder": ("text_extraction.txt_folder", ""),
    "save_odf": ("odf_export.save_odf", False),
    "odf_include_images": ("odf_export.include_images", True),
    "odf_open_after_export": ("odf_export.open_after_export", False),
    "md_include_front_matter": ("md_export.include_front_matter", False),
    "md_open_after_export": ("md_export.open_after_export", False),
    "dpi": ("rapidocr.dpi", DEFAULT_DPI),
    "enable_preprocessing": ("rapidocr.enable_preprocessing", DEFAULT_ENABLE_PREPROCESSING),
    "enable_deskew": ("rapidocr.enable_deskew", DEFAULT_ENABLE_DESKEW),
    "enable_baseline_dewarp": (
        "rapidocr.enable_baseline_dewarp",
        DEFAULT_ENABLE_BASELINE_DEWARP,
    ),
    "enable_perspective_correction": (
        "rapidocr.enable_perspective_correction",
        DEFAULT_ENABLE_PERSPECTIVE_CORRECTION,
    ),
    "enable_orientation_detection": (
        "rapidocr.enable_orientation_detection",
        DEFAULT_ENABLE_ORIENTATION_DETECTION,
    ),
    "enable_auto_contrast": ("rapidocr.enable_auto_contrast", DEFAULT_ENABLE_AUTO_CONTRAST),
    "enable_auto_brightness": (
        "rapidocr.enable_auto_brightness",
        DEFAULT_ENABLE_AUTO_BRIGHTNESS,
    ),
    "enable_denoise": ("rapidocr.enable_denoise", DEFAULT_ENABLE_DENOISE),
    "enable_scanner_effect": (
        "rapidocr.enable_scanner_effect",
        DEFAULT_ENABLE_SCANNER_EFFECT,
    ),
    "scanner_effect_strength": (
        "rapidocr.scanner_effect_strength",
        DEFAULT_SCANNER_EFFECT_STRENGTH,
    ),
    "enable_border_clean": ("rapidocr.enable_border_clean", DEFAULT_ENABLE_BORDER_CLEAN),
    "enable_vintage_look": ("rapidocr.enable_vintage_look", DEFAULT_ENABLE_VINTAGE_LOOK),
    "vintage_bw": ("rapidocr.vintage_bw", DEFAULT_VINTAGE_BW),
    "text_score_threshold": (
        "rapidocr.text_score_threshold",
        DEFAULT_TEXT_SCORE_THRESHOLD,
    ),
    "box_thresh": ("rapidocr.box_thresh", DEFAULT_BOX_THRESH),
    "unclip_ratio": ("rapidocr.unclip_ratio", DEFAULT_UNCLIP_RATIO),
    "ocr_profile": ("rapidocr.ocr_profile", "balanced"),
    "detection_full_resolution": (
        "rapidocr.detection_full_resolution",
        DEFAULT_DETECTION_FULL_RESOLUTION,
    ),
    "image_export_format": ("image_export.format", DEFAULT_IMAGE_EXPORT_FORMAT),
    "image_export_quality": ("image_export.quality", DEFAULT_IMAGE_EXPORT_QUALITY),
    "image_export_preserve_original": ("image_export.preserve_original", True),
    "auto_detect_quality": ("image_export.auto_detect_quality", DEFAULT_AUTO_DETECT_QUALITY),
    "convert_to_pdfa": ("output.convert_to_pdfa", True),
    "max_file_size_mb": ("output.max_file_size_mb", DEFAULT_MAX_FILE_SIZE_MB),
    "page_layout": ("output.page_layout", DEFAULT_PAGE_LAYOUT),
    "enable_bilevel_compression": (
        "output.enable_bilevel_compression",
        DEFAULT_ENABLE_BILEVEL_COMPRESSION,
    ),
    "force_bilevel_compression": (
        "output.force_bilevel_compression",
        DEFAULT_FORCE_BILEVEL_COMPRESSION,
    ),
    "quick_start_mode": ("ui.quick_start_mode", True),
}

_ODF_SETTING_NAMES = ("save_odf", "odf_include_images", "odf_open_after_export")
_MD_SETTING_NAMES = ("md_include_front_matter", "md_open_after_export")


class OcrSettings:
    """Flat application settings and transient document state."""

    destination_folder: str
    save_in_same_folder: bool
    pdf_suffix: str
    use_original_filename: bool
    overwrite_existing: bool
    include_date: bool
    include_year: bool
    include_month: bool
    include_day: bool
    include_time: bool
    date_format_order: dict[str, int]
    save_txt: bool
    separate_txt_folder: bool
    txt_folder: str
    save_odf: bool
    odf_include_images: bool
    odf_open_after_export: bool
    md_include_front_matter: bool
    md_open_after_export: bool
    image_export_format: str
    image_export_quality: int
    image_export_preserve_original: bool
    auto_detect_quality: bool
    convert_to_pdfa: bool
    max_file_size_mb: int
    page_layout: str
    enable_bilevel_compression: bool
    force_bilevel_compression: bool
    dpi: int
    enable_preprocessing: bool
    enable_deskew: bool
    enable_baseline_dewarp: bool
    enable_perspective_correction: bool
    enable_orientation_detection: bool
    enable_auto_contrast: bool
    enable_auto_brightness: bool
    enable_denoise: bool
    enable_scanner_effect: bool
    scanner_effect_strength: float
    enable_border_clean: bool
    enable_vintage_look: bool
    vintage_bw: bool
    text_score_threshold: float
    box_thresh: float
    unclip_ratio: float
    ocr_profile: str
    detection_full_resolution: bool
    replace_existing_ocr: bool
    enhance_embedded_images: bool
    quick_start_mode: bool

    def __init__(self) -> None:
        self.selected_files: list[str] = []
        self.page_ranges: dict[str, tuple[int, int] | None] = {}
        self.processed_files: list[str] = []
        self.original_file_paths: dict[str, str] = {}
        self.file_modifications: dict[str, dict[str, Any]] = {}
        self.pages_count = 0

        self.ocr_language = DEFAULT_LANGUAGE
        self.parallel_workers = DEFAULT_WORKERS
        for attribute, (_key, default) in _PERSISTED_SETTINGS.items():
            setattr(self, attribute, deepcopy(default))

        self._config = get_config_manager()

        # Attributes that stay directly on OcrSettings (cross-cutting / transient)
        self.lang: str = DEFAULT_LANGUAGE
        self.extracted_text: dict[str, str] = {}
        self.comparison_results: list[Any] = []
        self.ocr_boxes: dict[str, list[Any]] = {}

        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)

        # Load settings from JSON config manager
        self.load_settings()

    def load_settings(self) -> None:
        """Load all settings from JSON configuration."""
        if not self._config.reload():
            raise OSError("Could not refresh settings from disk")
        stored_language = self._config.get("rapidocr.language")
        if not isinstance(stored_language, str) or not stored_language.strip():
            stored_language = self._config.get("ocr.language", DEFAULT_LANGUAGE)
        if not isinstance(stored_language, str) or not stored_language.strip():
            stored_language = DEFAULT_LANGUAGE
        self.lang = stored_language
        self.ocr_language = stored_language
        self._load_persisted_settings()
        self.file_modifications = {}

        # Load selected files from legacy file (file list not stored in JSON)
        self._load_selected_files()

        # Only initialize destination folder if it isn't already set
        if not self.destination_folder:
            self._initialize_destination_folder()

        logger.info("Settings loaded from JSON configuration")

    def _snapshot_ocr_config(self) -> OCRConfig:
        """Snapshot the persisted OCR preferences for one processing request."""
        return OCRConfig(
            language=self.ocr_language,
            dpi=self.dpi,
            enable_preprocessing=self.enable_preprocessing,
            enable_perspective_correction=self.enable_perspective_correction,
            enable_deskew=self.enable_deskew,
            enable_baseline_dewarp=self.enable_baseline_dewarp,
            enable_orientation_detection=self.enable_orientation_detection,
            enable_auto_contrast=self.enable_auto_contrast,
            enable_auto_brightness=self.enable_auto_brightness,
            enable_denoise=self.enable_denoise,
            enable_scanner_effect=self.enable_scanner_effect,
            scanner_effect_strength=self.scanner_effect_strength,
            enable_border_clean=self.enable_border_clean,
            enable_vintage_look=self.enable_vintage_look,
            vintage_bw=self.vintage_bw,
            text_score_threshold=self.text_score_threshold,
            box_thresh=self.box_thresh,
            unclip_ratio=self.unclip_ratio,
            detection_full_resolution=self.detection_full_resolution,
            convert_to_pdfa=self.convert_to_pdfa,
            max_file_size_mb=self.max_file_size_mb,
            page_layout=self.page_layout,
            enable_bilevel_compression=self.enable_bilevel_compression,
            force_bilevel_compression=self.force_bilevel_compression,
            image_export_format=self.image_export_format,
            image_export_quality=self.image_export_quality,
            auto_detect_quality=self.auto_detect_quality,
            workers=self.parallel_workers,
            replace_existing_ocr=self.replace_existing_ocr,
            enhance_embedded_images=self.enhance_embedded_images,
        )

    def _load_md_settings(self) -> None:
        self._load_persisted_settings(_MD_SETTING_NAMES)

    def _load_persisted_settings(self, attributes=None) -> None:
        for attribute in attributes or _PERSISTED_SETTINGS:
            key, default = _PERSISTED_SETTINGS[attribute]
            setattr(self, attribute, self._config.get(key, deepcopy(default)))

    def add_files(self, file_paths: list[str]) -> int:
        """Add files to the selected files list

        Args:
            file_paths: List of file paths to add

        Returns:
            Number of files successfully added
        """
        if not file_paths:
            return 0

        logger.info(
            ngettext(
                "Attempting to add {count} file",
                "Attempting to add {count} files",
                len(file_paths),
            ).format(count=len(file_paths))
        )

        # Filter and collect valid files (PDFs and Images)
        valid_files = self._filter_valid_files(file_paths)

        # Add valid files and update count
        if valid_files:
            previous_count = len(self.selected_files)
            self.selected_files.extend(valid_files)
            if not self._save_selected_files():
                del self.selected_files[previous_count:]
                return 0

            # Only initialize destination if it's not already set by the user
            if not self.destination_folder:
                self._initialize_destination_folder()

            logger.info(
                ngettext(
                    "Successfully added {count} file",
                    "Successfully added {count} files",
                    len(valid_files),
                ).format(count=len(valid_files))
            )
        else:
            logger.warning(_("No valid files were found to add"))

        return len(valid_files)

    def _add_generated_file(self, file_path: str, original_path: str) -> bool:
        """Queue a generated PDF and publish its display-name source atomically."""
        was_queued = any(
            os.path.realpath(queued_path) == os.path.realpath(file_path)
            for queued_path in self.selected_files
        )
        if self.add_files([file_path]) != 1:
            if not was_queued:
                from bigocrpdf.utils.temp_manager import remove_tracked_file

                remove_tracked_file(file_path)
            return False
        self.original_file_paths[file_path] = original_path
        return True

    def _remove_file(self, file_path: str) -> bool:
        """Remove a queued file and all state owned by that queue entry."""
        previous_state = self._snapshot_queue_state()
        was_generated = file_path in self.original_file_paths
        try:
            self.selected_files.remove(file_path)
        except ValueError:
            return False
        self.page_ranges.pop(file_path, None)
        self.file_modifications.pop(file_path, None)
        self.original_file_paths.pop(file_path, None)
        if not self._save_selected_files():
            self._restore_queue_state(previous_state)
            return False
        if was_generated:
            from bigocrpdf.utils.temp_manager import remove_tracked_file

            remove_tracked_file(file_path)
        return True

    def _replace_file(self, file_path: str, replacement_path: str) -> bool:
        """Replace an edited queue entry with its materialized PDF."""
        previous_state = self._snapshot_queue_state()
        try:
            index = self.selected_files.index(file_path)
        except ValueError:
            return False
        was_generated = file_path in self.original_file_paths
        self.selected_files[index] = replacement_path
        original_path = self.original_file_paths.pop(file_path, file_path)
        self.original_file_paths[replacement_path] = original_path
        self.page_ranges.pop(file_path, None)
        self.file_modifications.pop(file_path, None)
        if not self._save_selected_files():
            self._restore_queue_state(previous_state)
            return False
        if was_generated and replacement_path != file_path:
            from bigocrpdf.utils.temp_manager import remove_tracked_file

            remove_tracked_file(file_path)
        return True

    def _move_file(self, source_index: int, target_index: int) -> bool:
        """Move a queued file and persist the new order."""
        file_count = len(self.selected_files)
        if (
            source_index == target_index
            or not 0 <= source_index < file_count
            or not 0 <= target_index < file_count
        ):
            return False

        previous_order = self.selected_files.copy()
        file_path = self.selected_files.pop(source_index)
        self.selected_files.insert(target_index, file_path)
        if self._save_selected_files():
            return True

        self.selected_files[:] = previous_order
        return False

    def _clear_files(self) -> bool:
        """Clear the queue and all state owned by its entries."""
        had_queue_state = bool(
            self.selected_files
            or self.page_ranges
            or self.file_modifications
            or self.original_file_paths
        )
        if not had_queue_state:
            return False
        previous_state = self._snapshot_queue_state()
        generated_files = tuple(self.original_file_paths)
        self.selected_files.clear()
        self.page_ranges.clear()
        self.file_modifications.clear()
        self.original_file_paths.clear()
        if not self._save_selected_files():
            self._restore_queue_state(previous_state)
            return False
        if generated_files:
            from bigocrpdf.utils.temp_manager import remove_tracked_file

            for file_path in generated_files:
                remove_tracked_file(file_path)
        return True

    def _snapshot_queue_state(
        self,
    ) -> tuple[
        list[str],
        dict[str, tuple[int, int] | None],
        dict[str, dict[str, Any]],
        dict[str, str],
    ]:
        """Copy queue-owned state for rollback around durable publication."""
        return (
            list(self.selected_files),
            dict(self.page_ranges),
            dict(self.file_modifications),
            dict(self.original_file_paths),
        )

    def _restore_queue_state(
        self,
        state: tuple[
            list[str],
            dict[str, tuple[int, int] | None],
            dict[str, dict[str, Any]],
            dict[str, str],
        ],
    ) -> None:
        """Restore a queue snapshot without replacing externally observed containers."""
        selected_files, page_ranges, file_modifications, original_file_paths = state
        self.selected_files[:] = selected_files
        self.page_ranges.clear()
        self.page_ranges.update(page_ranges)
        self.file_modifications.clear()
        self.file_modifications.update(file_modifications)
        self.original_file_paths.clear()
        self.original_file_paths.update(original_file_paths)

    def _filter_valid_files(self, file_paths: list[str]) -> list[str]:
        """Filter a list of paths to only include valid files (PDF and Images)

        Args:
            file_paths: List of file paths to filter

        Returns:
            List of valid file paths
        """
        valid_files: list[str] = []
        queued_identities = {os.path.realpath(path) for path in self.selected_files}

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

            # Skip duplicate filesystem identities, including aliases in this batch.
            identity = os.path.realpath(file_path)
            if identity in queued_identities:
                logger.info(_("File already in list: {0}").format(file_path))
                continue

            # File is valid, add it
            logger.info(_("Adding valid file: {0}").format(file_path))
            valid_files.append(file_path)
            queued_identities.add(identity)

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
            if not self._save_all_settings():
                raise OSError("Could not persist settings")

            logger.info(_("Settings saved successfully"))

        except Exception as e:
            logger.error(_("Error saving settings: {0}").format(e))
            raise

    def _save_all_settings(self) -> bool:
        """Save all settings to JSON configuration."""
        self.ocr_language = self.lang
        self._config.set("rapidocr.language", self.ocr_language, save_immediately=False)
        self._save_persisted_settings()
        self._save_editor_settings()
        if not self._config.save():
            return False
        logger.debug("All settings saved to JSON configuration")
        return True

    def _save_editor_settings(self) -> None:
        if self.page_ranges:
            self._config.set("editor.page_ranges", self.page_ranges, save_immediately=False)

    def _save_odf_settings(self) -> None:
        self._save_persisted_settings(_ODF_SETTING_NAMES)

    def _save_md_settings(self) -> None:
        self._save_persisted_settings(_MD_SETTING_NAMES)

    def _save_persisted_settings(self, attributes=None) -> None:
        for attribute in attributes or _PERSISTED_SETTINGS:
            key, default = _PERSISTED_SETTINGS[attribute]
            value = getattr(self, attribute, deepcopy(default))
            self._config.set(key, value, save_immediately=False)

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

    def _save_selected_files(self) -> bool:
        """Publish the current selected-file list and report durable success."""
        try:
            from bigocrpdf.utils.durable_writes import write_text_atomically

            payload = {
                "version": 1,
                "selected_files": self.selected_files,
            }
            write_text_atomically(
                SELECTED_FILE_PATH,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            selected_count = len(self.selected_files)
            logger.info(
                ngettext(
                    "Saved {count} selected file",
                    "Saved {count} selected files",
                    selected_count,
                ).format(count=selected_count)
            )
        except Exception as e:
            logger.error(_("Error saving selected files: {0}").format(e))
            return False
        return True

    def _load_selected_files(self) -> None:
        """Load selected files from configuration"""
        # Initialize selected files as empty list to ensure it's always iterable
        self.selected_files = []
        self.pages_count = 0

        try:
            with open(
                SELECTED_FILE_PATH,
                encoding="utf-8",
                opener=lambda path, flags: os.open(
                    path,
                    flags
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                ),
            ) as selected_file:
                if not stat.S_ISREG(os.fstat(selected_file.fileno()).st_mode):
                    raise OSError("selected-file list is not a regular file")
                raw = selected_file.read()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # Backward compatibility with the legacy newline-delimited file.
                self.selected_files = [line.strip() for line in raw.splitlines() if line.strip()]
            else:
                if not isinstance(payload, dict) or payload.get("version") != 1:
                    raise ValueError("unsupported selected-file list format")
                stored_files = payload.get("selected_files")
                if not isinstance(stored_files, list) or not all(
                    isinstance(file_path, str) for file_path in stored_files
                ):
                    raise ValueError("invalid selected-file list")
                self.selected_files = stored_files

            # Filter to only existing files
            self.selected_files = _deduplicate_file_paths(
                [f for f in self.selected_files if os.path.exists(f)]
            )

        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
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
            full: If True, also clears the file queue and its per-file state.
                  Use ``full=True``
                  when the user cancels processing and returns to the
                  settings page to start from scratch.
        """
        # Clear results
        self.processed_files: list[str] = []
        self.comparison_results: list[Any] = []

        # Clear extracted text to free memory
        if self.extracted_text:
            text_count = len(self.extracted_text)
            total_chars = sum(len(text) for text in self.extracted_text.values())
            self.extracted_text.clear()
            logger.info(
                f"Cleared {text_count} extracted texts ({total_chars} characters) from memory"
            )

        # Clear OCR boxes data
        if self.ocr_boxes:
            box_count = len(self.ocr_boxes)
            self.ocr_boxes.clear()
            logger.info(f"Cleared {box_count} OCR boxes from memory")

        # Full reset also clears the input file queue
        if full:
            self._clear_files()

        logger.info(_("Processing state reset successfully"))

    def display_name(self, file_path: str) -> str:
        """Return a user-friendly display name for a queued file."""
        original = self.original_file_paths.get(file_path)
        return os.path.basename(original or file_path)

    def cleanup_temp_files(self, processed_files: list[str]) -> None:
        """Retain the public hook while queue ownership releases exact temp inputs."""
        del processed_files

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values and save."""
        if not self._config.reset_to_defaults():
            raise OSError("Could not reset settings")

        OcrSettings.__init__(self)

        logger.info("All settings have been reset to defaults")
