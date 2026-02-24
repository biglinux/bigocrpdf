"""Output file naming, destination, and export format configuration."""

from __future__ import annotations

DEFAULT_SUFFIX = "ocr"
DEFAULT_DATE_FORMAT = {"year": 1, "month": 2, "day": 3}


class OutputConfig:
    """Output file naming, destination, and export format configuration."""

    def __init__(self) -> None:
        # Destination
        self.destination_folder: str = ""
        self.save_in_same_folder: bool = False
        # Filename
        self.pdf_suffix: str = DEFAULT_SUFFIX
        self.use_original_filename: bool = False
        self.overwrite_existing: bool = False
        # Date elements
        self.include_date: bool = False
        self.include_year: bool = False
        self.include_month: bool = False
        self.include_day: bool = False
        self.include_time: bool = False
        self.date_format_order: dict[str, int] = DEFAULT_DATE_FORMAT.copy()
        # Text extraction
        self.save_txt: bool = False
        self.separate_txt_folder: bool = False
        self.txt_folder: str = ""
        # ODF export
        self.save_odf: bool = False
        self.odf_include_images: bool = True
        self.odf_use_formatting: bool = True
        # Image export
        self.image_export_format: str = "original"
        self.image_export_quality: int = 85
        self.image_export_preserve_original: bool = True
        self.auto_detect_quality: bool = True
        # PDF output
        self.convert_to_pdfa: bool = True
        self.max_file_size_mb: int = 0
