"""
BigOcrPdf - OCR Processor Module

This module handles the OCR processing of PDF files using the OCRmyPDF API.
"""

from typing import List, Tuple, Dict, Optional, Callable, Any
import os
import subprocess
import logging

from services.settings import OcrSettings
from utils.logger import logger
from services.ocr_api import OcrQueue, configure_logging
from utils.i18n import _

# Processing constants
MAX_CONCURRENT_PROCESSES = 2
OCR_COMPRESSION_FORMATS = {
    "economic": "jpeg",
    "economicplus": "jpeg"
}
OCR_OPTIMIZATION_LEVELS = {
    "economic": 1,
    "economicplus": 1
}
DEFAULT_LANGUAGES = [
    ("por", "Portuguese"),
    ("eng", "English"),
    ("spa", "Spanish"),
]


class OcrProcessor:
    """Class to handle OCR processing tasks with stable progress tracking"""

    def __init__(self, settings: OcrSettings):
        """Initialize the OCR processor

        Args:
            settings: The OcrSettings object containing processing settings
        """
        self.settings = settings
        self.process_pid = None
        self.ocr_queue = None
        self.on_file_complete = None
        self.on_all_complete = None
        
        # Stable progress tracking - only real values
        self._is_processing = False
        self._processing_started = False

    def process_with_api(self) -> bool:
        """Process selected files using the OCRmyPDF Python API

        Returns:
            True if processing started successfully, False otherwise
        """
        try:
            if not self._validate_input_files():
                return False

            self._setup_processing()

            # Process each file - add to queue
            for i, file_path in enumerate(self.settings.selected_files):
                if not self._process_single_file(file_path, i):
                    continue

            # Start the OCR queue
            self.ocr_queue.start()
            self._is_processing = True
            self._processing_started = True
            
            logger.info(
                _("Started OCR processing for {0} files using Python API").format(
                    len(self.settings.selected_files)
                )
            )
            return True

        except Exception as e:
            logger.error(_("Error starting OCR processing: {0}").format(str(e)))
            return False

    def _validate_input_files(self) -> bool:
        """Validate that we have files to process"""
        if not self.settings.selected_files:
            logger.error(_("No files to process"))
            return False
        return True

    def _setup_processing(self) -> None:
        """Set up the OCR processing environment"""
        configure_logging()

        # Reset state
        self._is_processing = False
        self._processing_started = False

        # Create a new OCR queue
        self.ocr_queue = OcrQueue(max_concurrent=MAX_CONCURRENT_PROCESSES)

        # Set up callbacks
        self._register_internal_callbacks()

    def _register_internal_callbacks(self) -> None:
        """Register callbacks with the OCR queue"""
        if self.on_file_complete:
            self.ocr_queue.register_callback("file_complete", self.on_file_complete)
        if self.on_all_complete:
            self.ocr_queue.register_callback("all_complete", self._on_processing_complete)

        if not self.on_file_complete:
            logger.warning(_("No file_complete callback registered"))
        if not self.on_all_complete:
            logger.warning(_("No all_complete callback registered"))

    def _on_processing_complete(self) -> None:
        """Internal callback when all processing is complete"""
        self._is_processing = False
        if self.on_all_complete:
            self.on_all_complete()

    def _process_single_file(self, file_path: str, index: int) -> bool:
        """Process a single file with OCR"""
        if not file_path or not os.path.exists(file_path):
            logger.error(_("Error: File not found or invalid: {0}").format(file_path))
            return False

        output_file = self._get_output_file_path(file_path, index)
        if not output_file:
            return False

        self._track_output_file(output_file, index)
        options = self._create_ocr_options(file_path, output_file)

        self.ocr_queue.add_file(file_path, output_file, options)
        return True

    def _get_output_file_path(self, file_path: str, index: int) -> Optional[str]:
        """Determine the output file path for a processed file"""
        try:
            input_filename = os.path.basename(file_path)
            base_name = os.path.splitext(input_filename)[0]

            output_dir = self._get_output_directory(file_path)
            if not output_dir:
                logger.error(_("Could not determine output directory for {0}").format(file_path))
                return None

            output_file = self._create_output_file_path(output_dir, base_name, index)

            if os.path.exists(output_file) and not self.settings.overwrite_existing:
                output_file = self._generate_unique_filename(output_file)

            return output_file
        except Exception as e:
            logger.error(_("Error creating output path for {0}: {1}").format(file_path, e))
            return None

    def _get_output_directory(self, file_path: str) -> Optional[str]:
        """Determine the output directory for a processed file"""
        if self.settings.save_in_same_folder:
            return os.path.dirname(file_path)
        elif self.settings.destination_folder:
            os.makedirs(self.settings.destination_folder, exist_ok=True)
            return self.settings.destination_folder
        else:
            return os.path.dirname(file_path)

    def _create_output_file_path(self, output_dir: str, base_name: str, index: int) -> str:
        """Create the output file path based on settings"""
        use_original = getattr(self.settings, "use_original_filename", False)

        if use_original:
            return os.path.join(output_dir, f"{base_name}.pdf")
        else:
            suffix = self.settings.get_pdf_suffix() or "ocr"
            if index == 0:
                return os.path.join(output_dir, f"{base_name}-{suffix}.pdf")
            else:
                return os.path.join(output_dir, f"{base_name}-{suffix}-{index + 1}.pdf")

    def _track_output_file(self, output_file: str, index: int) -> None:
        """Track output files in settings"""
        if index == 0:
            self.settings.processed_files = []

        if output_file not in self.settings.processed_files:
            self.settings.processed_files.append(output_file)

    def _create_ocr_options(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Create OCR options dictionary for OCRmyPDF"""
        temp_dir = os.path.join(os.path.dirname(output_file), ".temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_sidecar = os.path.join(
            temp_dir,
            f"temp_{os.path.basename(os.path.splitext(output_file)[0])}.txt",
        )

        options = {
            "language": self.settings.lang,
            "force_ocr": True,
            "progress_bar": False,
            "sidecar": temp_sidecar,
        }

        self._add_text_extraction_options(options, output_file)
        self._add_quality_options(options)
        self._add_alignment_options(options)
        
        return options

    def _add_text_extraction_options(self, options: Dict[str, Any], output_file: str) -> None:
        """Add text extraction options to the options dictionary"""
        if hasattr(self.settings, "save_txt") and self.settings.save_txt:
            if (
                hasattr(self.settings, "separate_txt_folder")
                and self.settings.separate_txt_folder
                and self.settings.txt_folder
            ):
                txt_filename = (
                    os.path.basename(os.path.splitext(output_file)[0]) + ".txt"
                )
                sidecar_file = os.path.join(
                    self.settings.txt_folder, txt_filename
                )
                os.makedirs(self.settings.txt_folder, exist_ok=True)
            else:
                sidecar_file = os.path.splitext(output_file)[0] + ".txt"

            options["sidecar"] = sidecar_file

    def _add_quality_options(self, options: Dict[str, Any]) -> None:
        """Add quality-related options to the options dictionary"""
        if self.settings.quality in OCR_COMPRESSION_FORMATS:
            options["pdfa_image_compression"] = OCR_COMPRESSION_FORMATS[self.settings.quality]
            options["optimize"] = OCR_OPTIMIZATION_LEVELS[self.settings.quality]
            
            if self.settings.quality == "economicplus":
                options["oversample"] = 300

    def _add_alignment_options(self, options: Dict[str, Any]) -> None:
        """Add alignment-related options to the options dictionary"""
        if self.settings.align == "align":
            options["deskew"] = True
        elif self.settings.align == "rotate":
            options["rotate_pages"] = True
        elif self.settings.align == "alignrotate":
            options["deskew"] = True
            options["rotate_pages"] = True

    def get_available_ocr_languages(self) -> List[Tuple[str, str]]:
        """Get a list of available OCR languages from tesseract"""
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                check=False,
            )

            if not result or not result.stdout:
                logger.warning(_("tesseract --list-langs returned empty output"))
                return DEFAULT_LANGUAGES

            languages = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or "List of available languages" in line or "osd" in line:
                    continue

                if line == "por":
                    languages.append((line, "Portuguese"))
                elif line == "eng":
                    languages.append((line, "English"))
                elif line == "spa":
                    languages.append((line, "Spanish"))
                else:
                    languages.append((line, line))

            return languages if languages else DEFAULT_LANGUAGES
        except Exception as e:
            logger.error(_("Error getting OCR languages: {0}").format(e))
            return DEFAULT_LANGUAGES

    def get_progress(self) -> float:
        """Get the current OCR processing progress - stable tracking only

        Returns:
            Float between 0.0 and 1.0 representing completion percentage
        """
        # Use OCR queue for real progress tracking
        if self.ocr_queue:
            return self.ocr_queue.get_progress()

        # If no queue exists, return appropriate values
        if not self._processing_started:
            return 0.0
        elif not self._is_processing:
            return 1.0
        else:
            return 0.0

    def get_processed_count(self) -> int:
        """Get the number of files that have been processed so far"""
        if self.ocr_queue:
            return self.ocr_queue.get_processed_count()
        return 0

    def get_total_count(self) -> int:
        """Get the total number of files (processed + queued)"""
        if self.ocr_queue:
            return self.ocr_queue.get_total_count()
        
        # Fallback to settings if no queue
        return len(self.settings.selected_files) if self.settings.selected_files else 0
    
    def get_total_pages(self) -> int:
        """Get the total number of pages across all files"""
        if self.ocr_queue:
            return self.ocr_queue.get_total_pages()
        
        return getattr(self.settings, 'pages_count', 0)

    def get_processed_pages(self) -> int:
        """Get the number of pages processed so far"""
        if self.ocr_queue:
            return self.ocr_queue.get_processed_pages()
        
        return 0

    def get_current_file_info(self) -> Dict[str, Any]:
        """Get information about the currently processing file"""
        if self.ocr_queue:
            return self.ocr_queue.get_current_file_info()
        
        return {}

    def register_callbacks(self, on_file_complete: Optional[Callable] = None, 
                         on_all_complete: Optional[Callable] = None) -> None:
        """Register callbacks for OCR processing events"""
        self.on_file_complete = on_file_complete
        self.on_all_complete = on_all_complete

        if self.ocr_queue:
            if on_file_complete:
                self.ocr_queue.register_callback("file_complete", on_file_complete)
            if on_all_complete:
                self.ocr_queue.register_callback("all_complete", self._on_processing_complete)

    def remove_processed_file(self, input_file: str) -> None:
        """Remove a processed file from the selected files list"""
        if input_file in self.settings.selected_files:
            self.settings.selected_files.remove(input_file)
            logger.info(
                _("Removed processed file from queue: {0}").format(os.path.basename(input_file))
            )

    def _generate_unique_filename(self, file_path: str) -> str:
        """Generate a unique filename by appending a counter"""
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        counter = 1
        new_path = file_path

        while os.path.exists(new_path):
            new_path = os.path.join(dir_name, f"{name}-{counter}{ext}")
            counter += 1

        logger.info(
            _("Generated unique filename to avoid overwriting: {0}").format(os.path.basename(new_path))
        )
        return new_path
    
    def force_cleanup(self) -> None:
        """Force aggressive cleanup of all resources"""
        try:
            if self.ocr_queue:
                self.ocr_queue.stop()
                self.ocr_queue = None
                
            # Reset all state
            self._is_processing = False
            self._processing_started = False
            self.process_pid = None
            
            # Clear callbacks
            self.on_file_complete = None
            self.on_all_complete = None
            
            logger.info("OCR processor force cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")
            
    def get_processing_speed(self) -> float:
        """Get the current processing speed in files per minute"""
        if self.ocr_queue:
            return self.ocr_queue.get_processing_speed()
        return 0.0

    def get_estimated_time_remaining(self) -> int:
        """Get estimated time remaining in seconds"""
        if self.ocr_queue:
            return self.ocr_queue.get_estimated_time_remaining()
        return 0

    def is_processing(self) -> bool:
        """Check if processing is currently active"""
        return self._is_processing

    def has_started(self) -> bool:
        """Check if processing has been started"""
        return self._processing_started