"""
BigOcrPdf - Settings Service

This module handles configuration settings and file management for the application.
"""

import os
import subprocess
import time
from typing import List, Dict, Optional, Tuple, Any
from utils.logger import logger
from utils.i18n import _
from config import (
    SELECTED_FILE_PATH,
    LANG_FILE_PATH,
    QUALITY_FILE_PATH,
    ALIGN_FILE_PATH,
    SAVEFILE_PATH,
    SAME_FOLDER_PATH,
    CONFIG_DIR,
)

# Default settings constants
DEFAULT_LANGUAGE = "eng"
DEFAULT_QUALITY = "normal"
DEFAULT_ALIGNMENT = "alignrotate"
DEFAULT_SUFFIX = "ocr"
DEFAULT_DATE_FORMAT = {"year": 1, "month": 2, "day": 3}


class OcrSettings:
    """Class to manage OCR settings and configuration files"""

    def __init__(self):
        """Initialize OCR settings with default values"""
        # File management
        self.selected_files: List[str] = []
        self.pages_count: int = 0
        self.destination_folder: str = ""
        self.processed_files: List[str] = []
        
        # OCR processing settings
        self.lang: str = DEFAULT_LANGUAGE
        self.quality: str = DEFAULT_QUALITY
        self.align: str = DEFAULT_ALIGNMENT
        self.save_in_same_folder: bool = False
        self.extracted_text: Dict[str, str] = {}
        
        # Output filename settings
        self.pdf_suffix: str = DEFAULT_SUFFIX
        self.overwrite_existing: bool = False
        
        # Date inclusion settings
        self.include_date: bool = False
        self.include_year: bool = False
        self.include_month: bool = False
        self.include_day: bool = False
        self.include_time: bool = False
        self.date_format_order = DEFAULT_DATE_FORMAT.copy()
        
        # Text extraction settings
        self.save_txt: bool = False
        self.separate_txt_folder: bool = False
        self.txt_folder: str = ""
        
        # Ensure config directory exists
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # Ensure alignment configuration exists with default value
        if not os.path.exists(ALIGN_FILE_PATH):
            try:
                with open(ALIGN_FILE_PATH, "w") as f:
                    f.write(DEFAULT_ALIGNMENT)
            except Exception as e:
                logger.error(_("Error creating alignment setting file: {0}").format(e))
        
        # Load settings from config files
        self.load_settings()

    def load_settings(self) -> None:
        """Load all settings from configuration files"""
        self._load_selected_files()
        self._load_language()
        self._load_quality()
        self._load_alignment()
        self._load_same_folder_option()
        self._load_pdf_output_settings()
        
        # Only initialize destination folder if it isn't already set
        if not self.destination_folder:
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

        logger.info(_("Attempting to add {0} files").format(len(file_paths)))

        # Filter and collect valid PDF files
        valid_files = self._filter_valid_pdf_files(file_paths)

        # Add valid files and update count
        if valid_files:
            self.selected_files.extend(valid_files)
            self._count_pdf_pages(valid_files)
            self._save_selected_files()

            # Only initialize destination if it's not already set by the user
            if not self.destination_folder:
                self._initialize_destination_folder()

            logger.info(_("Successfully added {0} files").format(len(valid_files)))
        else:
            logger.warning(_("No valid PDF files were found to add"))

        return len(valid_files)

    def _filter_valid_pdf_files(self, file_paths: List[str]) -> List[str]:
        """Filter a list of paths to only include valid PDF files
        
        Args:
            file_paths: List of file paths to filter
            
        Returns:
            List of valid PDF file paths
        """
        valid_files = []
        
        for file_path in file_paths:
            # Skip empty paths
            if not file_path:
                logger.warning(_("Empty file path provided"))
                continue

            # Skip non-existent files
            if not os.path.exists(file_path):
                logger.warning(_("File does not exist: {0}").format(file_path))
                continue

            # Skip non-PDF files
            if not file_path.lower().endswith(".pdf"):
                logger.warning(_("Not a PDF file: {0}").format(file_path))
                continue

            # Skip duplicates
            if file_path in self.selected_files:
                logger.info(_("File already in list: {0}").format(file_path))
                continue

            # File is valid, add it
            logger.info(_("Adding valid PDF: {0}").format(file_path))
            valid_files.append(file_path)
            
        return valid_files

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
            
            # Update destination if needed
            if not self.selected_files:
                self.destination_folder = ""
            elif not self.destination_folder:
                self._initialize_destination_folder()

    def clear_files(self) -> None:
        """Remove all files from the selection"""
        self.selected_files = []
        self.pages_count = 0
        self._save_selected_files()
        self.destination_folder = ""

    def save_settings(
        self,
        lang: str,
        quality: str,
        align: str,
        destination_folder: str,
        save_in_same_folder: bool = False,
    ) -> None:
        """Save current settings to configuration files
        
        Args:
            lang: OCR language code
            quality: Quality setting (normal, economic, economicplus)
            align: Alignment setting (none, align, rotate, alignrotate)
            destination_folder: Path to save output files
            save_in_same_folder: Whether to save in same folder as original
        """
        try:
            # Update values
            self.lang = lang or DEFAULT_LANGUAGE
            self.quality = quality or DEFAULT_QUALITY
            self.align = align or DEFAULT_ALIGNMENT
            self.destination_folder = destination_folder
            self.save_in_same_folder = save_in_same_folder

            # Save core settings
            self._save_core_settings()
            
            # Save PDF output settings
            self._save_pdf_output_settings()

            logger.info(_("Settings saved successfully"))

        except Exception as e:
            logger.error(_("Error saving settings: {0}").format(e))
            raise

    def _save_core_settings(self) -> None:
        """Save core settings to configuration files"""
        # Create a settings dictionary for writing
        settings_data = {
            LANG_FILE_PATH: self.lang,
            QUALITY_FILE_PATH: self.quality,
            ALIGN_FILE_PATH: self.align,
            SAVEFILE_PATH: self.destination_folder,
            SAME_FOLDER_PATH: "true" if self.save_in_same_folder else "false"
        }
        
        # Write each setting to its file
        for filepath, value in settings_data.items():
            try:
                with open(filepath, "w") as f:
                    f.write(str(value))
            except Exception as e:
                logger.error(_("Error saving setting to {0}: {1}").format(
                    os.path.basename(filepath), e))

    def _save_pdf_output_settings(self) -> None:
        """Save PDF output settings to configuration files"""
        try:
            pdf_settings_dir = os.path.dirname(SAVEFILE_PATH)

            # PDF suffix
            pdf_suffix_path = os.path.join(pdf_settings_dir, "pdf_suffix")
            with open(pdf_suffix_path, "w") as f:
                f.write(self.pdf_suffix or DEFAULT_SUFFIX)

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

            logger.info(_("PDF output settings saved successfully"))
        except Exception as e:
            logger.error(_("Error saving PDF output settings: {0}").format(e))

    def get_pdf_count_and_pages(self) -> Tuple[int, int]:
        """Get count of selected PDFs and total pages
        
        Returns:
            Tuple containing (file_count, page_count)
        """
        return len(self.selected_files), self.pages_count

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
        date_components = []

        # Add date elements with their preferred order
        if self.include_year:
            date_components.append((self.date_format_order.get("year", 1), f"{now.tm_year}"))
        if self.include_month:
            date_components.append((self.date_format_order.get("month", 2), f"{now.tm_mon:02d}"))
        if self.include_day:
            date_components.append((self.date_format_order.get("day", 3), f"{now.tm_mday:02d}"))

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

    def get_pdf_page_count(self, file_path: str) -> Optional[int]:
        """Get page count for a single PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of pages or None if count failed
        """
        if not file_path or not os.path.exists(file_path):
            return None
            
        try:
            result = subprocess.run(
                ["pdfinfo", file_path],
                capture_output=True,
                text=True,
                check=False,
            )
            
            for line in result.stdout.split("\n"):
                if line.startswith("Pages:"):
                    return int(line.split(":")[1].strip())
                    
        except Exception as e:
            logger.error(_("Error getting page count for {0}: {1}").format(
                os.path.basename(file_path), e))
        
        return None

    def _count_pdf_pages(self, file_paths: List[str]) -> None:
        """Count pages in PDF files and add to the total page count

        Args:
            file_paths: List of PDF file paths to count pages for
        """
        for file_path in file_paths:
            if file_path and file_path.lower().endswith(".pdf"):
                page_count = self.get_pdf_page_count(file_path)
                if page_count:
                    self.pages_count += page_count

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
            with open(SELECTED_FILE_PATH, "r") as f:
                file_lines = f.readlines()
                if file_lines:  # Check if there are any lines
                    self.selected_files = [line.strip() for line in file_lines if line.strip()]

            # Filter to only existing files
            self.selected_files = [f for f in self.selected_files if os.path.exists(f)]
                    
            # Count pages in PDF files
            self._count_pdf_pages(self.selected_files)
            
        except Exception as e:
            logger.error(_("Error loading selected files: {0}").format(e))
            # Ensure selected_files is always a list
            self.selected_files = []

    def _load_setting_file(self, filepath: str, default_value: Any) -> Any:
        """Load a setting from a configuration file with error handling
        
        Args:
            filepath: Path to the configuration file
            default_value: Default value to use if file doesn't exist or error occurs
            
        Returns:
            The loaded setting value or default
        """
        if not os.path.exists(filepath):
            return default_value
            
        try:
            with open(filepath, "r") as f:
                value = f.read().strip()
                return value if value else default_value
        except Exception as e:
            logger.error(_("Error loading setting from {0}: {1}").format(
                os.path.basename(filepath), e))
            return default_value

    def _load_language(self) -> None:
        """Load language setting from configuration"""
        self.lang = self._load_setting_file(LANG_FILE_PATH, DEFAULT_LANGUAGE)
        
        # If failed to load, try to detect from system locale
        if not self.lang:
            locale = os.environ.get("LANG", "").lower()
            if locale.startswith("pt"):
                self.lang = "por"
            elif locale.startswith("es"):
                self.lang = "spa"
            else:
                self.lang = DEFAULT_LANGUAGE

    def _load_quality(self) -> None:
        """Load quality setting from configuration"""
        self.quality = self._load_setting_file(QUALITY_FILE_PATH, DEFAULT_QUALITY)

    def _load_alignment(self) -> None:
        """Load alignment setting from configuration"""
        self.align = self._load_setting_file(ALIGN_FILE_PATH, DEFAULT_ALIGNMENT)

    def _load_same_folder_option(self) -> None:
        """Load save in same folder option from configuration"""
        value = self._load_setting_file(SAME_FOLDER_PATH, "false")
        self.save_in_same_folder = value.lower() == "true"

    def _load_pdf_output_settings(self) -> None:
        """Load PDF output filename settings from configuration"""
        pdf_settings_dir = os.path.dirname(SAVEFILE_PATH)

        # Load PDF suffix
        pdf_suffix_path = os.path.join(pdf_settings_dir, "pdf_suffix")
        self.pdf_suffix = self._load_setting_file(pdf_suffix_path, DEFAULT_SUFFIX)

        # Load overwrite setting
        overwrite_path = os.path.join(pdf_settings_dir, "overwrite_existing")
        value = self._load_setting_file(overwrite_path, "false")
        self.overwrite_existing = value.lower() == "true"

        # Load date settings
        self._load_date_settings(pdf_settings_dir)
        
        # Load destination folder
        self.destination_folder = self._load_setting_file(SAVEFILE_PATH, "")

    def _load_date_settings(self, pdf_settings_dir: str) -> None:
        """Load date-related settings from configuration
        
        Args:
            pdf_settings_dir: Directory containing date settings files
        """
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
                logger.error(_("Error loading PDF date settings: {0}").format(e))

        # Load date format order
        self._load_date_format_order(pdf_settings_dir)
        
    def _load_date_format_order(self, pdf_settings_dir: str) -> None:
        """Load date format order settings
        
        Args:
            pdf_settings_dir: Directory containing date format order settings file
        """
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
                            logger.error(_("Invalid position value for {0}: {1}").format(key, value))
            except Exception as e:
                logger.error(_("Error loading date format order settings: {0}").format(e))
                # Reset to default in case of error
                self.date_format_order = DEFAULT_DATE_FORMAT.copy()

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
        
    def reset_processing_state(self) -> None:
        """Reset processing-related state for new OCR run"""
        # Clear processed files list
        self.processed_files = []
        
        # Clear extracted text to free memory
        if hasattr(self, 'extracted_text') and self.extracted_text:
            text_count = len(self.extracted_text)
            total_chars = sum(len(text) for text in self.extracted_text.values())
            self.extracted_text.clear()
            logger.info(f"Cleared {text_count} extracted texts ({total_chars} characters) from memory")
        else:
            self.extracted_text = {}
            
        logger.info(_("Processing state reset successfully"))
    
    def cleanup_temp_files(self, processed_files: List[str]) -> None:
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