"""
BigOcrPdf - Settings Service

This module handles configuration settings and file management for the application.
"""

import os
import subprocess
from typing import List, Dict
from ..utils.logger import logger
from ..utils.i18n import _
from ..config import (
    SELECTED_FILE_PATH,
    LANG_FILE_PATH,
    QUALITY_FILE_PATH,
    ALIGN_FILE_PATH,
    SAVEFILE_PATH,
    SAME_FOLDER_PATH,
)


class OcrSettings:
    """Class to manage OCR settings and configuration files"""

    def __init__(self):
        """Initialize OCR settings with default values"""
        self.selected_files: List[str] = []
        self.pages_count: int = 0
        self.lang: str = "eng"  # Default language
        self.quality: str = "normal"  # Default quality
        self.align: str = "none"  # Default alignment
        self.destination_folder: str = ""
        self.processed_files: List[str] = []  # Track files processed by OCR
        self.save_in_same_folder: bool = (
            False  # Option to save in the same folder as original file
        )
        # Initialize extracted_text dictionary to store OCR results
        self.extracted_text: Dict[str, str] = {}

        # PDF output filename settings
        self.pdf_suffix: str = "ocr"  # Default suffix for output PDFs
        self.overwrite_existing: bool = False  # Whether to overwrite existing files
        self.include_date: bool = False  # Whether to include date in filenames
        self.include_year: bool = False  # Include year in filename
        self.include_month: bool = False  # Include month in filename
        self.include_day: bool = False  # Include day in filename
        self.include_time: bool = False  # Include time in filename

        # Text extraction settings
        self.save_txt: bool = False  # Whether to save text files automatically
        self.separate_txt_folder: bool = (
            False  # Whether to save text files to a separate folder
        )
        self.txt_folder: str = (
            ""  # Folder to save text files if separate folder is enabled
        )

        # Date format order preference (position for each component)
        self.date_format_order = {
            "year": 1,  # Default: Year first (e.g., YYYY-MM-DD)
            "month": 2,  # Default: Month second
            "day": 3,  # Default: Day third
        }

        # Load settings from config files
        self.load_settings()

    def load_settings(self) -> None:
        """Load settings from configuration files"""
        self._load_selected_files()
        self._load_language()
        self._load_quality()
        self._load_alignment()
        self._load_same_folder_option()
        self._load_pdf_output_settings()
        self._initialize_destination_folder()

    def add_files(self, file_paths: List[str]) -> int:
        """Add files to the selected files list

        Args:
            file_paths: List of file paths to add

        Returns:
            Number of files successfully added
        """
        if not file_paths:
            return 0

        logger.info(f"Attempting to add {len(file_paths)} files")

        # Filter valid PDF files
        valid_files = []
        for file_path in file_paths:
            logger.info(f"Checking file: {file_path}")

            # First check if file exists
            if not file_path:
                logger.warning("Empty file path provided")
                continue

            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                continue

            # Check if it's a PDF (more lenient - just check file extension)
            if not file_path.lower().endswith(".pdf"):
                logger.warning(f"Not a PDF file: {file_path}")
                continue

            # Avoid duplicates
            if file_path in self.selected_files:
                logger.info(f"File already in list: {file_path}")
                continue

            # Accept the file
            logger.info(f"Adding valid PDF: {file_path}")
            valid_files.append(file_path)

        # Add valid files and update count
        if valid_files:
            self.selected_files.extend(valid_files)
            self._count_pdf_pages(valid_files)
            self._save_selected_files()

            # Only initialize destination if it's not already set by the user
            if not self.destination_folder:
                self._initialize_destination_folder()

            logger.info(f"Successfully added {len(valid_files)} files")
        else:
            logger.warning("No valid PDF files were found to add")

        return len(valid_files)

    def remove_files(self, indices: List[int]) -> None:
        """Remove files at specified indices from the selected files list

        Args:
            indices: List of indices to remove
        """
        if not indices or not self.selected_files:
            return

        # Sort indices in reverse order to avoid index shifting during removal
        indices.sort(reverse=True)

        # Remove files by index
        removed_files = []
        for idx in indices:
            if 0 <= idx < len(self.selected_files):
                removed_files.append(self.selected_files.pop(idx))

        # Recalculate page count if files were removed
        if removed_files:
            self._recalculate_page_count()
            self._save_selected_files()
            self._initialize_destination_folder()  # Update destination if needed

    def clear_files(self) -> None:
        """Remove all files from the selection"""
        self.selected_files = []
        self.pages_count = 0
        self._save_selected_files()
        self._initialize_destination_folder()

    def save_settings(
        self,
        lang: str,
        quality: str,
        align: str,
        destination_folder: str,
        save_in_same_folder: bool = False,
    ) -> None:
        """Save current settings to configuration files"""
        try:
            # Update values
            self.lang = lang
            self.quality = quality
            self.align = align
            self.destination_folder = destination_folder
            self.save_in_same_folder = save_in_same_folder

            # Save to configuration files
            with open(LANG_FILE_PATH, "w") as f:
                f.write(self.lang)

            with open(QUALITY_FILE_PATH, "w") as f:
                f.write(self.quality)

            with open(ALIGN_FILE_PATH, "w") as f:
                f.write(self.align)

            with open(SAVEFILE_PATH, "w") as f:
                f.write(self.destination_folder)

            with open(SAME_FOLDER_PATH, "w") as f:
                f.write("true" if self.save_in_same_folder else "false")

            # Save PDF output settings
            try:
                pdf_settings_dir = os.path.dirname(SAVEFILE_PATH)

                # PDF suffix
                pdf_suffix_path = os.path.join(pdf_settings_dir, "pdf_suffix")
                with open(pdf_suffix_path, "w") as f:
                    f.write(self.pdf_suffix or "ocr")

                # Overwrite existing
                overwrite_path = os.path.join(pdf_settings_dir, "overwrite_existing")
                with open(overwrite_path, "w") as f:
                    f.write("true" if self.overwrite_existing else "false")

                # Date inclusion settings
                date_settings_path = os.path.join(pdf_settings_dir, "pdf_date_settings")
                date_settings = {
                    "include_date": self.include_date,
                    "include_year": self.include_year,
                    "include_month": self.include_month,
                    "include_day": self.include_day,
                    "include_time": self.include_time,
                }

                with open(date_settings_path, "w") as f:
                    for key, value in date_settings.items():
                        f.write(f"{key}={str(value).lower()}\n")

                # Save date format order
                order_settings_path = os.path.join(
                    pdf_settings_dir, "date_format_order"
                )
                with open(order_settings_path, "w") as f:
                    for key, value in self.date_format_order.items():
                        f.write(f"{key}={value}\n")

                logger.info("PDF output settings saved successfully")
            except Exception as e:
                logger.error(f"Error saving PDF output settings: {e}")

            logger.info("Settings saved successfully")

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            raise

    def _count_pdf_pages(self, file_paths: List[str]) -> None:
        """Count pages in PDF files and add to the total page count

        Args:
            file_paths: List of PDF file paths to count pages for
        """
        for file_path in file_paths:
            if file_path and file_path.lower().endswith(".pdf"):
                try:
                    result = subprocess.run(
                        ["pdfinfo", file_path],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    for line in result.stdout.split("\n"):
                        if line.startswith("Pages:"):
                            self.pages_count += int(line.split(":")[1].strip())
                except Exception as e:
                    logger.error(f"Error counting PDF pages: {e}")

    def _recalculate_page_count(self) -> None:
        """Recalculate the total page count for all selected files"""
        self.pages_count = 0
        self._count_pdf_pages(self.selected_files)

    def _save_selected_files(self) -> None:
        """Save the current list of selected files to the configuration file"""
        try:
            with open(SELECTED_FILE_PATH, "w") as f:
                for file_path in self.selected_files:
                    f.write(f"{file_path}\n")
            logger.info(f"Saved {len(self.selected_files)} selected files")
        except Exception as e:
            logger.error(f"Error saving selected files: {e}")

    def _load_selected_files(self) -> None:
        """Load selected files from configuration"""
        # Initialize selected files as empty list to ensure it's always iterable
        self.selected_files = []

        if not os.path.exists(SELECTED_FILE_PATH):
            return

        try:
            with open(SELECTED_FILE_PATH, "r") as f:
                file_lines = f.readlines()
                if file_lines:  # Check if there are any lines
                    self.selected_files = [line.strip() for line in file_lines]

                # Count pages in PDF files
                self.pages_count = 0
                for file_path in self.selected_files:
                    if file_path and file_path.lower().endswith(".pdf"):
                        try:
                            result = subprocess.run(
                                ["pdfinfo", file_path],
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            for line in result.stdout.split("\n"):
                                if line.startswith("Pages:"):
                                    self.pages_count += int(line.split(":")[1].strip())
                        except Exception as e:
                            logger.error(f"Error counting PDF pages: {e}")
        except Exception as e:
            logger.error(f"Error loading selected files: {e}")
            # Ensure selected_files is always a list
            self.selected_files = []

    def _load_language(self) -> None:
        """Load language setting from configuration"""
        if os.path.exists(LANG_FILE_PATH):
            try:
                with open(LANG_FILE_PATH, "r") as f:
                    self.lang = f.read().strip()
            except Exception as e:
                logger.error(f"Error loading language setting: {e}")
                # If not set or error, try to detect from system locale
                locale = os.environ.get("LANG", "").lower()
                if locale.startswith("pt"):
                    self.lang = "por"
                elif locale.startswith("es"):
                    self.lang = "spa"
                else:
                    self.lang = "eng"

    def _load_quality(self) -> None:
        """Load quality setting from configuration"""
        if os.path.exists(QUALITY_FILE_PATH):
            try:
                with open(QUALITY_FILE_PATH, "r") as f:
                    self.quality = f.read().strip()
            except Exception as e:
                logger.error(f"Error loading quality setting: {e}")
                self.quality = "normal"

    def _load_alignment(self) -> None:
        """Load alignment setting from configuration"""
        if os.path.exists(ALIGN_FILE_PATH):
            try:
                with open(ALIGN_FILE_PATH, "r") as f:
                    self.align = f.read().strip()
            except Exception as e:
                logger.error(f"Error loading alignment setting: {e}")
                self.align = "none"

    def _load_same_folder_option(self) -> None:
        """Load save in same folder option from configuration"""
        if os.path.exists(SAME_FOLDER_PATH):
            try:
                with open(SAME_FOLDER_PATH, "r") as f:
                    value = f.read().strip().lower()
                    self.save_in_same_folder = value == "true"
            except Exception as e:
                logger.error(f"Error loading same folder option: {e}")
                self.save_in_same_folder = False

    def _load_pdf_output_settings(self) -> None:
        """Load PDF output filename settings from configuration"""
        pdf_settings_dir = os.path.dirname(SAVEFILE_PATH)

        # Load PDF suffix
        pdf_suffix_path = os.path.join(pdf_settings_dir, "pdf_suffix")
        if os.path.exists(pdf_suffix_path):
            try:
                with open(pdf_suffix_path, "r") as f:
                    suffix = f.read().strip()
                    self.pdf_suffix = suffix if suffix else "ocr"
            except Exception as e:
                logger.error(f"Error loading PDF suffix setting: {e}")

        # Load overwrite setting
        overwrite_path = os.path.join(pdf_settings_dir, "overwrite_existing")
        if os.path.exists(overwrite_path):
            try:
                with open(overwrite_path, "r") as f:
                    value = f.read().strip().lower()
                    self.overwrite_existing = value == "true"
            except Exception as e:
                logger.error(f"Error loading overwrite setting: {e}")

        # Load date settings
        date_settings_path = os.path.join(pdf_settings_dir, "pdf_date_settings")
        if os.path.exists(date_settings_path):
            try:
                with open(date_settings_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue

                        key, value = line.split("=", 1)
                        value = value.lower() == "true"

                        if key == "include_date":
                            self.include_date = value
                        elif key == "include_year":
                            self.include_year = value
                        elif key == "include_month":
                            self.include_month = value
                        elif key == "include_day":
                            self.include_day = value
                        elif key == "include_time":
                            self.include_time = value
            except Exception as e:
                logger.error(f"Error loading PDF date settings: {e}")

        # Load date format order
        order_settings_path = os.path.join(pdf_settings_dir, "date_format_order")
        if os.path.exists(order_settings_path):
            try:
                with open(order_settings_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or "=" not in line:
                            continue

                        key, value = line.split("=", 1)
                        try:
                            position = int(value)
                            if key in ["year", "month", "day"]:
                                self.date_format_order[key] = position
                        except ValueError:
                            logger.error(f"Invalid position value for {key}: {value}")
            except Exception as e:
                logger.error(f"Error loading date format order settings: {e}")

    def _initialize_destination_folder(self) -> None:
        """Initialize the destination folder path based on selected files"""
        if not self.selected_files:
            return

        first_file = self.selected_files[0]
        file_folder = os.path.dirname(first_file)

        # Check if folder is writable, if not use home directory
        if not os.access(file_folder, os.W_OK):
            file_folder = os.path.expanduser("~")

        # For folders, we don't need to check for existing files
        # We'll just use the parent directory of the first file as the destination folder
        self.destination_folder = file_folder

    def get_pdf_suffix(self) -> str:
        """Get the formatted PDF suffix with date elements if enabled

        Returns:
            The formatted suffix string for PDF files
        """
        import time

        # Start with the base suffix
        suffix = self.pdf_suffix or "ocr"

        # If date inclusion is not enabled, return just the suffix
        if not self.include_date:
            return suffix

        # Get current time
        now = time.localtime()

        # Initialize date components with position ordering
        date_components = []

        # Check if we have a date format order preference
        order = getattr(self, "date_format_order", {"year": 1, "month": 2, "day": 3})

        # Add date elements with their preferred order
        if self.include_year:
            date_components.append((order.get("year", 1), f"{now.tm_year}"))
        if self.include_month:
            date_components.append((order.get("month", 2), f"{now.tm_mon:02d}"))
        if self.include_day:
            date_components.append((order.get("day", 3), f"{now.tm_mday:02d}"))

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
