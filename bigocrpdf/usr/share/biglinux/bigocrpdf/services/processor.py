from typing import List, Tuple
import os
import subprocess

from .settings import OcrSettings
from ..utils.logger import logger
from .ocr_api import OcrQueue, configure_logging


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
            # Check if we have any files to process
            if not self.settings.selected_files:
                logger.error("No files to process")
                return False

            # Configure OCRmyPDF logging
            configure_logging()

            # Reset any tracking variables
            self._processed_files = 0
            self._progress = 0.0

            # Create a new OCR queue
            self.ocr_queue = OcrQueue(max_concurrent=2)  # Process up to 2 files at once

            # Set up callbacks
            if self.on_file_complete:
                self.ocr_queue.register_callback("file_complete", self.on_file_complete)
            if self.on_all_complete:
                self.ocr_queue.register_callback("all_complete", self.on_all_complete)

            # Make sure we always have callbacks set for consistency
            if not self.on_file_complete:
                logger.warning("No file_complete callback registered")
            if not self.on_all_complete:
                logger.warning("No all_complete callback registered")

            # Add each file to the queue
            for i, file_path in enumerate(self.settings.selected_files):
                if not file_path or not os.path.exists(file_path):
                    logger.error(f"Error: File not found or invalid: {file_path}")
                    continue

                # Get the base name for output file naming
                input_filename = os.path.basename(file_path)
                base_name = os.path.splitext(input_filename)[0]

                # Determine output file path
                # If save in same folder is enabled, use the directory of the original file
                if self.settings.save_in_same_folder:
                    output_dir = os.path.dirname(file_path)
                elif self.settings.destination_folder:
                    output_dir = self.settings.destination_folder
                else:
                    # Fallback to the original file's directory if no destination folder specified
                    output_dir = os.path.dirname(file_path)

                # Check if we should use the original filename
                use_original = getattr(self.settings, "use_original_filename", False)

                if use_original:
                    # Use the original filename without adding any suffix
                    output_file = os.path.join(output_dir, f"{base_name}.pdf")
                else:
                    # Get custom suffix if it exists, otherwise use default "ocr"
                    suffix = self.settings.get_pdf_suffix() or "ocr"

                    # Generate output filename with appropriate suffix
                    if i == 0:
                        output_file = os.path.join(
                            output_dir, f"{base_name}-{suffix}.pdf"
                        )
                    else:
                        output_file = os.path.join(
                            output_dir, f"{base_name}-{suffix}-{i + 1}.pdf"
                        )

                # Check if file already exists and generate a unique name if needed
                if os.path.exists(output_file) and not self.settings.overwrite_existing:
                    # Generate a unique filename to avoid overwriting
                    output_file = self._generate_unique_filename(output_file)

                # Clean up the processed files list before starting new processing
                if i == 0:
                    # First file in batch, reset the processed files list
                    self.settings.processed_files = []

                # Keep track of output files for the conclusion page
                if output_file not in self.settings.processed_files:
                    self.settings.processed_files.append(output_file)

                # Create options dictionary for OCRmyPDF

                # Always create a temporary sidecar file for text extraction to memory
                temp_dir = os.path.join(os.path.dirname(output_file), ".temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_sidecar = os.path.join(
                    temp_dir,
                    f"temp_{os.path.basename(os.path.splitext(output_file)[0])}.txt",
                )

                options = {
                    "language": self.settings.lang,
                    "force_ocr": True,
                    "progress_bar": False,  # We'll handle progress display ourselves
                    "sidecar": temp_sidecar,  # Always extract text for memory storage
                }

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

                        # Update the sidecar option to save to the permanent location
                        options["sidecar"] = sidecar_file
                    else:
                        # Save alongside the PDF
                        sidecar_file = os.path.splitext(output_file)[0] + ".txt"

                        # Update the sidecar option to save to the permanent location
                        options["sidecar"] = sidecar_file

                # Add quality settings
                if self.settings.quality == "economic":
                    options["pdfa_image_compression"] = "jpeg"
                    options["optimize"] = 1
                elif self.settings.quality == "economicplus":
                    options["pdfa_image_compression"] = "jpeg"
                    options["optimize"] = 1
                    options["oversample"] = 300

                # Add alignment settings
                if self.settings.align == "align":
                    options["deskew"] = True
                elif self.settings.align == "rotate":
                    options["rotate_pages"] = True
                elif self.settings.align == "alignrotate":
                    options["deskew"] = True
                    options["rotate_pages"] = True

                # Add file to queue
                self.ocr_queue.add_file(file_path, output_file, options)

            # Start processing
            self.ocr_queue.start()
            logger.info(
                f"Started OCR processing for {len(self.settings.selected_files)} files using Python API"
            )
            return True

        except Exception as e:
            logger.error(f"Error starting OCR processing: {str(e)}")
            return False

    def get_available_ocr_languages(self) -> List[Tuple[str, str]]:
        """Get a list of available OCR languages from tesseract

        Returns:
            A list of tuples containing (language_code, language_name)
        """
        # Always initialize with default languages in case of failure
        default_languages = [
            ("por", "Portuguese"),
            ("eng", "English"),
            ("spa", "Spanish"),
        ]

        try:
            # Run tesseract to list available languages
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                check=False,
            )

            if not result or not result.stdout:
                logger.warning("tesseract --list-langs returned empty output")
                return default_languages

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
            return languages if languages else default_languages
        except Exception as e:
            logger.error(f"Error getting OCR languages: {e}")
            # Return default languages on error
            return default_languages

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

    def register_callbacks(self, on_file_complete=None, on_all_complete=None):
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
                f"Removed processed file from queue: {os.path.basename(input_file)}"
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
            f"Generated unique filename to avoid overwriting: {os.path.basename(new_path)}"
        )
        return new_path
