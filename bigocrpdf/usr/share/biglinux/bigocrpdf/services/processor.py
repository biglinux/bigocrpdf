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
    """Class to handle OCR processing tasks"""

    def __init__(self, settings: OcrSettings):
        """Initialize the OCR processor

        Args:
            settings: The OcrSettings object containing processing settings
        """
        self.settings = settings
        self.process_pid = None
        self._progress = 0.0
        self._processed_files = 0
        self._total_files = 0
        self._last_progress_update = 0
        self.ocr_queue = None
        self.on_file_complete = None
        self.on_all_complete = None

    def process_with_api(self) -> bool:
        """Process selected files using the OCRmyPDF Python API

        Returns:
            True if processing started successfully, False otherwise
        """
        try:
            # Validate input files
            if not self._validate_input_files():
                return False

            # Configure processing environment
            self._setup_processing()

            # Process each file
            for i, file_path in enumerate(self.settings.selected_files):
                if not self._process_single_file(file_path, i):
                    continue

            # Start the OCR queue
            self.ocr_queue.start()
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
        """Validate that we have files to process
        
        Returns:
            True if files exist, False otherwise
        """
        if not self.settings.selected_files:
            logger.error(_("No files to process"))
            return False
        return True

    def _setup_processing(self) -> None:
        """Set up the OCR processing environment"""
        # Configure OCRmyPDF logging
        configure_logging()

        # Reset tracking variables
        self._processed_files = 0
        self._progress = 0.0

        # Create a new OCR queue
        self.ocr_queue = OcrQueue(max_concurrent=MAX_CONCURRENT_PROCESSES)

        # Set up callbacks
        self._register_internal_callbacks()

    def _register_internal_callbacks(self) -> None:
        """Register callbacks with the OCR queue"""
        if self.on_file_complete:
            self.ocr_queue.register_callback("file_complete", self.on_file_complete)
        if self.on_all_complete:
            self.ocr_queue.register_callback("all_complete", self.on_all_complete)

        # Log warning if callbacks are missing
        if not self.on_file_complete:
            logger.warning(_("No file_complete callback registered"))
        if not self.on_all_complete:
            logger.warning(_("No all_complete callback registered"))

    def _process_single_file(self, file_path: str, index: int) -> bool:
        """Process a single file with OCR
        
        Args:
            file_path: Path to the input file
            index: Index of the file in the selected files list
            
        Returns:
            True if file was added to queue, False otherwise
        """
        # Skip invalid files
        if not file_path or not os.path.exists(file_path):
            logger.error(_("Error: File not found or invalid: {0}").format(file_path))
            return False

        # Determine output file path
        output_file = self._get_output_file_path(file_path, index)
        if not output_file:
            return False

        # Track output files in settings
        self._track_output_file(output_file, index)

        # Create OCR options
        options = self._create_ocr_options(file_path, output_file)

        # Add file to OCR queue
        self.ocr_queue.add_file(file_path, output_file, options)
        return True

    def _get_output_file_path(self, file_path: str, index: int) -> Optional[str]:
        """Determine the output file path for a processed file
        
        Args:
            file_path: Input file path
            index: Index of the file in the processing queue
            
        Returns:
            Output file path or None if error occurred
        """
        try:
            # Get the base name for output file naming
            input_filename = os.path.basename(file_path)
            base_name = os.path.splitext(input_filename)[0]

            # Determine output directory
            output_dir = self._get_output_directory(file_path)
            if not output_dir:
                logger.error(_("Could not determine output directory for {0}").format(file_path))
                return None

            # Create output file path
            output_file = self._create_output_file_path(output_dir, base_name, index)

            # Check if file already exists and handle accordingly
            if os.path.exists(output_file) and not self.settings.overwrite_existing:
                # Generate a unique filename to avoid overwriting
                output_file = self._generate_unique_filename(output_file)

            return output_file
        except Exception as e:
            logger.error(_("Error creating output path for {0}: {1}").format(file_path, e))
            return None

    def _get_output_directory(self, file_path: str) -> Optional[str]:
        """Determine the output directory for a processed file
        
        Args:
            file_path: Input file path
            
        Returns:
            Output directory path or None if error occurred
        """
        # If save in same folder is enabled, use the directory of the original file
        if self.settings.save_in_same_folder:
            return os.path.dirname(file_path)
        elif self.settings.destination_folder:
            # Make sure destination directory exists
            os.makedirs(self.settings.destination_folder, exist_ok=True)
            return self.settings.destination_folder
        else:
            # Fallback to the original file's directory if no destination folder specified
            return os.path.dirname(file_path)

    def _create_output_file_path(self, output_dir: str, base_name: str, index: int) -> str:
        """Create the output file path based on settings
        
        Args:
            output_dir: Output directory path
            base_name: Base name of the input file (without extension)
            index: Index of the file in the processing queue
            
        Returns:
            Output file path
        """
        # Check if we should use the original filename
        use_original = getattr(self.settings, "use_original_filename", False)

        if use_original:
            # Use the original filename without adding any suffix
            return os.path.join(output_dir, f"{base_name}.pdf")
        else:
            # Get custom suffix if it exists, otherwise use default "ocr"
            suffix = self.settings.get_pdf_suffix() or "ocr"

            # Generate output filename with appropriate suffix
            if index == 0:
                return os.path.join(output_dir, f"{base_name}-{suffix}.pdf")
            else:
                return os.path.join(output_dir, f"{base_name}-{suffix}-{index + 1}.pdf")

    def _track_output_file(self, output_file: str, index: int) -> None:
        """Track output files in settings
        
        Args:
            output_file: Output file path
            index: Index of the file in the processing queue
        """
        # Clean up the processed files list before starting new processing
        if index == 0:
            # First file in batch, reset the processed files list
            self.settings.processed_files = []

        # Keep track of output files for the conclusion page
        if output_file not in self.settings.processed_files:
            self.settings.processed_files.append(output_file)

    def _create_ocr_options(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Create OCR options dictionary for OCRmyPDF
        
        Args:
            input_file: Input file path
            output_file: Output file path
            
        Returns:
            Dictionary of OCR options
        """
        # Create a temporary directory for extracted text if needed
        temp_dir = os.path.join(os.path.dirname(output_file), ".temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a temporary sidecar file for text extraction
        temp_sidecar = os.path.join(
            temp_dir,
            f"temp_{os.path.basename(os.path.splitext(output_file)[0])}.txt",
        )

        # Base options
        options = {
            "language": self.settings.lang,
            "force_ocr": True,
            "progress_bar": False,  # We'll handle progress display ourselves
            "sidecar": temp_sidecar,  # Always extract text for memory storage
        }

        # Add text extraction options
        self._add_text_extraction_options(options, output_file)
        
        # Add quality settings
        self._add_quality_options(options)
        
        # Add alignment settings
        self._add_alignment_options(options)
        
        return options

    def _add_text_extraction_options(self, options: Dict[str, Any], output_file: str) -> None:
        """Add text extraction options to the options dictionary
        
        Args:
            options: OCR options dictionary to modify
            output_file: Output file path
        """
        # If text files should be saved permanently, set up the proper location
        if hasattr(self.settings, "save_txt") and self.settings.save_txt:
            # Determine the location for the text file
            if (
                hasattr(self.settings, "separate_txt_folder")
                and self.settings.separate_txt_folder
                and self.settings.txt_folder
            ):
                # Save to a separate folder
                txt_filename = (
                    os.path.basename(os.path.splitext(output_file)[0]) + ".txt"
                )
                sidecar_file = os.path.join(
                    self.settings.txt_folder, txt_filename
                )
                # Ensure the directory exists
                os.makedirs(self.settings.txt_folder, exist_ok=True)
            else:
                # Save alongside the PDF
                sidecar_file = os.path.splitext(output_file)[0] + ".txt"

            # Update the sidecar option to save to the permanent location
            options["sidecar"] = sidecar_file

    def _add_quality_options(self, options: Dict[str, Any]) -> None:
        """Add quality-related options to the options dictionary
        
        Args:
            options: OCR options dictionary to modify
        """
        if self.settings.quality in OCR_COMPRESSION_FORMATS:
            options["pdfa_image_compression"] = OCR_COMPRESSION_FORMATS[self.settings.quality]
            options["optimize"] = OCR_OPTIMIZATION_LEVELS[self.settings.quality]
            
            # Add oversample for economicplus
            if self.settings.quality == "economicplus":
                options["oversample"] = 300

    def _add_alignment_options(self, options: Dict[str, Any]) -> None:
        """Add alignment-related options to the options dictionary
        
        Args:
            options: OCR options dictionary to modify
        """
        if self.settings.align == "align":
            options["deskew"] = True
        elif self.settings.align == "rotate":
            options["rotate_pages"] = True
        elif self.settings.align == "alignrotate":
            options["deskew"] = True
            options["rotate_pages"] = True

    def get_available_ocr_languages(self) -> List[Tuple[str, str]]:
        """Get a list of available OCR languages from tesseract

        Returns:
            A list of tuples containing (language_code, language_name)
        """
        try:
            # Run tesseract to list available languages
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
            # Process the output lines
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or "List of available languages" in line or "osd" in line:
                    continue

                # Map language codes to names
                if line == "por":
                    languages.append((line, "Portuguese"))
                elif line == "eng":
                    languages.append((line, "English"))
                elif line == "spa":
                    languages.append((line, "Spanish"))
                else:
                    languages.append((line, line))

            # Return default languages if we didn't find any
            return languages if languages else DEFAULT_LANGUAGES
        except Exception as e:
            logger.error(_("Error getting OCR languages: {0}").format(e))
            # Return default languages on error
            return DEFAULT_LANGUAGES

    def get_progress(self) -> float:
        """Get the current OCR processing progress

        Returns:
            Float between 0.0 and 1.0 representing completion percentage
        """
        # Use the OCR queue if available
        if self.ocr_queue:
            return self.ocr_queue.get_progress()

        # Initialize progress if it doesn't exist
        if not hasattr(self, "_progress"):
            self._progress = 0.0

        if not hasattr(self, "_processed_files"):
            self._processed_files = 0

        if not hasattr(self, "_total_files"):
            self._total_files = (
                len(self.settings.selected_files) if self.settings.selected_files else 0
            )

        # If progress is already at 100%, don't recalculate
        if self._progress >= 1.0:
            return self._progress

        # Simple check if process is still running
        if hasattr(self, "process_pid") and self.process_pid:
            try:
                # Check if process still exists
                os.kill(self.process_pid, 0)

                # If total files is known, use that for progress calculation
                if self._total_files > 0 and self._processed_files > 0:
                    self._progress = min(
                        0.99, self._processed_files / self._total_files
                    )

            except OSError:
                # Process is no longer running, set progress to complete
                self._progress = 1.0

        return self._progress

    def get_processed_count(self) -> int:
        """Get the number of files that have been processed so far

        Returns:
            Integer count of processed files
        """
        # Use the OCR queue if available
        if self.ocr_queue:
            return self.ocr_queue.get_processed_count()

        # Initialize tracking properties if they don't exist
        if not hasattr(self, "_progress"):
            self._progress = 0.0
        if not hasattr(self, "_processed_files"):
            self._processed_files = 0
        if not hasattr(self, "_total_files"):
            self._total_files = (
                len(self.settings.selected_files) if self.settings.selected_files else 0
            )

        # Simple heuristic for now - we'll just estimate based on the progress
        self._processed_files = int(self._progress * self._total_files)
        if self._processed_files < 1 and self._progress > 0.5:
            self._processed_files = 1

        return self._processed_files

    def get_total_count(self) -> int:
        """Get the total number of files (processed + queued)

        Returns:
            Total count of files in the OCR process
        """
        # If we have an active OCR queue, use its tracking
        if self.ocr_queue:
            # The OCR queue's get_total_count returns total files added for processing
            processed = self.get_processed_count()
            remaining = self.ocr_queue.count_remaining_files()
            return processed + remaining

        # Otherwise use the settings.selected_files count
        processed = self.get_processed_count()
        queued = len(self.settings.selected_files)
        return processed + queued

    def register_callbacks(self, on_file_complete: Optional[Callable] = None, 
                         on_all_complete: Optional[Callable] = None) -> None:
        """Register callbacks for OCR processing events

        Args:
            on_file_complete: Function to call when a file is completed
            on_all_complete: Function to call when all files are completed
        """
        self.on_file_complete = on_file_complete
        self.on_all_complete = on_all_complete

        # Register with the OCR queue if it already exists
        if self.ocr_queue:
            if on_file_complete:
                self.ocr_queue.register_callback("file_complete", on_file_complete)
            if on_all_complete:
                self.ocr_queue.register_callback("all_complete", on_all_complete)

    def remove_processed_file(self, input_file: str) -> None:
        """Remove a processed file from the selected files list

        Args:
            input_file: Path to the input file that was processed
        """
        if input_file in self.settings.selected_files:
            self.settings.selected_files.remove(input_file)
            logger.info(
                _("Removed processed file from queue: {0}").format(os.path.basename(input_file))
            )

    def _generate_unique_filename(self, file_path: str) -> str:
        """Generate a unique filename by appending a counter

        Args:
            file_path: Original file path

        Returns:
            A unique file path that doesn't exist on the filesystem
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)

        counter = 1
        new_path = file_path

        # Keep incrementing counter until we find a filename that doesn't exist
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
            # Stop OCR queue if exists
            if self.ocr_queue:
                self.ocr_queue.stop()
                self.ocr_queue = None
                
            # Reset all state
            self._progress = 0.0
            self._processed_files = 0
            self._total_files = 0
            self.process_pid = None
            
            # Clear callbacks
            self.on_file_complete = None
            self.on_all_complete = None
            
            logger.info("OCR processor force cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")